#include <QApplication>
#include <QMainWindow>

#include <iostream>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>

typedef pcl::PointXYZ PointDepth;
typedef pcl::PointXYZRGB PointDepthColor;
typedef pcl::PointCloud<PointDepth> PointCloudDepth;
typedef pcl::PointCloud<PointDepthColor> PointCloudDepthColor;
typedef pcl::visualization::PCLVisualizer Visualizer;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointDepthColor> PointColor;
typedef Visualizer::Ptr VisualizerSharedPtr;
typedef PointCloudDepthColor::ConstPtr PointCloudDepthColorSharedConstPtr;
typedef PointCloudDepth::Ptr PointCloudDepthSharedPtr;
typedef PointCloudDepthColor::Ptr PointCloudDepthColorSharedPtr;

void CopyColorToDepth(PointCloudDepthColorSharedPtr cloud_in, PointCloudDepthSharedPtr cloud_out)
{
    for (int i = 0; i < cloud_in->size(); i++) {
        PointDepth point;
        point.x = cloud_in->at(i).r / 255.0;
        point.y = cloud_in->at(i).g / 255.0;
        point.z = cloud_in->at(i).b / 255.0;
        cloud_out->push_back(point);
    }
}

void ConvertToMeter (const float feat_x, const float feat_y, const float depth, float &X, float &Y, float &Z) {
    // reject invalid points
    if (depth <= 0) {
        X = 0;
        Y = 0;
        Z = 0;
        return;
    }

    // My kinect v2 intrinsic params
    float cx = 254.878f;
    float cy = 205.395f;
    float fx = 365.456f;
    float fy = 365.456f;

//    float fx = 525.0; // focal length x
//    float fy = 525.0; // focal length y
//    float cx = 512./2; // optical center x
//    float cy = 424./2; // optical center y
    float sclFactor = 1000.0;

    // Recall the camera projective projection model
    Z = depth / sclFactor;
    X = (feat_x - cx) * Z / fx;
    Y = (feat_y - cy) * Z / fy;

}

// Convert RGB-D to point cloud
void ConvertColorToCloud (PointCloudDepthColor &PointCloudColor, const cv::Mat &colorImg, const cv::Mat &depthImg) {
    for(int i=0; i<colorImg.rows; i++) {
        for(int j=0; j<colorImg.cols; j++) {
            float X, Y, Z;

            unsigned short depth = depthImg.at<unsigned short>(i, j);

            // Render the 3D values
            ConvertToMeter(i, j, depth, X, Y, Z);

            // Remove features which are out of Kinect senser range
            if (X > 5 || Y > 5 || Z == 0.0) {
                continue;
            }

            // Write out the colored 3D point
            float R = (float)colorImg.at<cv::Vec3b>(i,j)[0];
            float G = (float)colorImg.at<cv::Vec3b>(i,j)[1];
            float B = (float)colorImg.at<cv::Vec3b>(i,j)[2];

            // Push back the 3D point
            PointDepthColor pt;
            pt.x = X; pt.y = Y; pt.z = Z;
            pt.r = R; pt.g = G; pt.b = B;
            PointCloudColor.push_back(pt);
        }
    }
}

void ConvertFeaturesToCloud (PointCloudDepth &cloud_1, const cv::Mat &rgb_1, const cv::Mat &depth_1, std::vector<cv::KeyPoint> &keyPts_1,
                             PointCloudDepth &cloud_2, const cv::Mat &rgb_2, const cv::Mat &depth_2, std::vector<cv::KeyPoint> &keyPts_2,
                             std::vector<cv::DMatch> *matches) {
    for (int i = 0; i < matches->size(); i++) {
        // Get the index of matching pairs
        int idx_1 = (*matches)[i].queryIdx;
        int idx_2 = (*matches)[i].trainIdx;

        // Round feature positions
        int feat_1_x = (int) keyPts_1[idx_1].pt.y;
        int feat_1_y = (int) keyPts_1[idx_1].pt.x;
        int feat_2_x = (int) keyPts_2[idx_2].pt.y;
        int feat_2_y = (int) keyPts_2[idx_2].pt.x;

        unsigned short feat_1_z = depth_1.at<unsigned short>(feat_1_x, feat_1_y);
        unsigned short feat_2_z = depth_2.at<unsigned short>(feat_2_x, feat_2_y);

        // Convert image point (u,v)+depth to [X, Y, Z] 3D coordinate
        float X_1, Y_1, Z_1;
        float X_2, Y_2, Z_2;

        ConvertToMeter(feat_1_x, feat_1_y, feat_1_z, X_1, Y_1, Z_1);
        ConvertToMeter(feat_2_x, feat_2_y, feat_2_z, X_2, Y_2, Z_2);

        // Remove features which are out of Kinect senser range
        if (X_1 > 5 || Y_1 > 5 || Z_1 == 0.0 || X_2 > 5 || Y_2 > 5 || Z_2 == 0.0) {
            continue;
        }

        PointDepth point;

        point.x = X_1;
        point.y = Y_1;
        point.z = Z_1;
        cloud_1.push_back(point);

        point.x = X_2;
        point.y = Y_2;
        point.z = Z_2;
        cloud_2.push_back(point);
    }
}

VisualizerSharedPtr cloudViewer (PointCloudDepthColorSharedConstPtr cloud, std::string title) {
    // Open 3D viewer and add point cloud
    VisualizerSharedPtr viewer (new Visualizer(title));
    //viewer->setBackgroundColor(0, 0, 0);
    viewer->setCameraPosition(0.0, 0.0, -2.5, 0.0, 0.0, 0.0);
    pcl::visualization::PointCloudColorHandlerRGBField<PointDepthColor> handler(cloud);
    viewer->addPointCloud<PointDepthColor>(cloud, handler, "reference cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reference cloud");
    viewer->addCoordinateSystem (0.10);
    viewer->initCameraParameters ();
    return (viewer);
}

// ICP registration
void RegisterFrames(PointCloudDepthColorSharedPtr cloudReference, PointCloudDepthColorSharedPtr cloudSample,
                    PointCloudDepthSharedPtr matchReference, PointCloudDepthSharedPtr matchSample,
                    PointCloudDepthColorSharedPtr cloudFused, bool icpRefinement)
{
    // Define an icp regisration object
    pcl::IterativeClosestPoint<PointDepth, PointDepth> icp;
    icp.setInputCloud(matchSample);
    icp.setInputTarget(matchReference);
    icp.setRANSACOutlierRejectionThreshold(0.05);
    PointCloudDepth aligned;
    icp.align(aligned);

    // Estimate the rigid transformation
    Eigen::Matrix4f transMat = icp.getFinalTransformation();

    // Transform the sample cloud to reference cloud coordinate
    pcl::transformPointCloud(*cloudSample, *cloudFused, transMat);

    // Registration refinement using ICP with all points
    if (icpRefinement) {
        PointCloudDepthSharedPtr cloud_in (new PointCloudDepth);
        CopyColorToDepth(cloudFused, cloud_in);

        PointCloudDepthSharedPtr cloud_out(new PointCloudDepth);
        CopyColorToDepth(cloudReference, cloud_out);

        icp.setInputCloud(cloud_in);
        icp.setInputTarget(cloud_out);
        icp.align(aligned);
        transMat = icp.getFinalTransformation();
        pcl::transformPointCloud(*cloudFused, *cloudFused, transMat);
    }

    // Fuse with the reference cloud
    for (int i = 0; i < cloudReference->size(); i++) {
        cloudFused->push_back(cloudReference->at(i));
    }
}

int main (int argc, char *argv[])
{
    QApplication a (argc, argv);

    std::cout << "SE3D started" << std::endl;

    // Read images
    std::string commonPath = "../model_reconstruction/kinect_data";
//    std::string depth1File = commonPath + "/depth/frame_01.png";
//    std::string depth2File = commonPath + "/depth/frame_02.png";
//    std::string color1File = commonPath + "/rgb/frame_01.png";
//    std::string color2File = commonPath + "/rgb/frame_02.png";

    std::string depth1File = commonPath + "/depth/depth28.png";
    std::string depth2File = commonPath + "/depth/depth29.png";
    std::string color1File = commonPath + "/rgb/color28.png";
    std::string color2File = commonPath + "/rgb/color29.png";

    cv::Mat depthImg_1 = cv::imread(depth1File, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat colorImg_1 = cv::imread(color1File, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat depthImg_2 = cv::imread(depth2File, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat colorImg_2 = cv::imread(color2File, CV_LOAD_IMAGE_UNCHANGED);


    if( !depthImg_1.data || !colorImg_1.data || !depthImg_2.data || !colorImg_2.data ) {
        std::cerr << "Reading one or more images failed." << std::endl;
        return -1;
    }

    if (colorImg_1.channels() == 4) {
        cv::cvtColor(colorImg_1, colorImg_1, cv::COLOR_RGBA2BGR);
    }
    if (colorImg_2.channels() == 4) {
        cv::cvtColor(colorImg_2, colorImg_2, cv::COLOR_RGBA2BGR);
    }

    if (depthImg_1.depth() != CV_16U) {
        cv::cvtColor(depthImg_1, depthImg_1, cv::COLOR_RGB2GRAY);
        depthImg_1.convertTo(depthImg_1, CV_16U);
    }

    if (depthImg_2.depth() != CV_16U) {
        cv::cvtColor(depthImg_2, depthImg_2, cv::COLOR_RGB2GRAY);
        depthImg_2.convertTo(depthImg_2, CV_16U);
    }

    // Flip images
    cv::flip(colorImg_1, colorImg_1, 1);
    cv::flip(colorImg_2, colorImg_2, 1);
    cv::flip(depthImg_1, depthImg_1, 1);
    cv::flip(depthImg_2, depthImg_2, 1);

    // Convert to grey
    cv::Mat colorImgGrey_1;
    cv::Mat colorImgGrey_2;
    cv::cvtColor(colorImg_1, colorImgGrey_1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(colorImg_2, colorImgGrey_2, cv::COLOR_RGB2GRAY);

    // Detect keypoints with SURF
    cv::SURF detector;

    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;

    detector.detect(colorImgGrey_1, keypoints_1);
    detector.detect(colorImgGrey_2, keypoints_2);

    //-- Draw keypoints
    cv::Mat keypointsImg_1;
    cv::Mat keypointsImg_2;

    cv::drawKeypoints(colorImg_1, keypoints_1, keypointsImg_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::drawKeypoints(colorImg_2, keypoints_2, keypointsImg_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::namedWindow("Keypoints 1", CV_WINDOW_NORMAL);
    cv::imshow("Keypoints 1", keypointsImg_1);

    cv::namedWindow("Keypoints 2", CV_WINDOW_NORMAL);
    cv::imshow("Keypoints 2", keypointsImg_2);

    // Compute descriptors
    cv::SiftDescriptorExtractor extractor;

    cv::Mat descriptors_1;
    cv::Mat descriptors_2;

    extractor.compute(colorImgGrey_1, keypoints_1, descriptors_1);
    extractor.compute(colorImgGrey_2, keypoints_2, descriptors_2);

    std::cout<<"Descriptors type: "<<descriptors_1.type()<<std::endl;

    // Match descriptor vectors with FLANN (could test with brute force)
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> bulkMatches;
    matcher.match(descriptors_1, descriptors_2, bulkMatches);

    //    // Match descriptor vectors with brute force
    //    cv::BFMatcher matcher(cv::NORM_L2);
    //    std::vector< cv::DMatch > bulkMatches;
    //    matcher.match( descriptors_1, descriptors_2, bulkMatches);

    double maxDist = 0;
    double minDist = 10000;

    // Compute max and min distances between keypoints
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = bulkMatches[i].distance;
        if (dist < minDist) {
            minDist = dist;
        }
        if (dist > maxDist) {
            maxDist = dist;
        }
    }

    printf("Max dist : %f \n", maxDist);
    printf("Min dist : %f \n", minDist);

    // Keep only matches with small distances
    std::vector<cv::DMatch> matches;
;
    double margin = 0.02;
    for(int i = 0; i < descriptors_1.rows; i++) {
        int idx = bulkMatches[i].queryIdx;
        int feat_x = (int) keypoints_1[idx].pt.x;
        int feat_y = (int) keypoints_1[idx].pt.y;

        if (bulkMatches[i].distance <= cv::max(2*minDist, margin) && colorImgGrey_1.at<unsigned char>(feat_y, feat_x) != 0) {
            matches.push_back(bulkMatches[i]);
        }
    }

    // Draw matches
    cv::Mat matchesImg;
    cv::drawMatches(colorImg_1, keypoints_1, colorImg_2, keypoints_2,
                    matches, matchesImg, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Show detected matches
    cv::namedWindow("Matches", CV_WINDOW_NORMAL);
    cv::imshow("Matches", matchesImg);

    for( int i = 0; i < (int)matches.size(); i++ ) {
        printf( "Match #%d => Keypoint 1: %d & Keypoint 2: %d\n", i, matches[i].queryIdx, matches[i].trainIdx );
    }

    // Make point clouds
    PointCloudDepthColor::Ptr cloudColor_1(new PointCloudDepthColor);
    PointCloudDepthColor::Ptr cloudColor_2(new PointCloudDepthColor);

    ConvertColorToCloud(*cloudColor_1, colorImg_1, depthImg_1);
    ConvertColorToCloud(*cloudColor_2, colorImg_2, depthImg_2);

    // Load the 3d matching pairs
    PointCloudDepthSharedPtr cloudFeatures_1(new PointCloudDepth);
    PointCloudDepthSharedPtr cloudFeatures_2(new PointCloudDepth);

    ConvertFeaturesToCloud(*cloudFeatures_1, colorImg_1, depthImg_1, keypoints_1, *cloudFeatures_2, colorImg_2, depthImg_2, keypoints_2, &matches);

    // Visualize the point cloud
    VisualizerSharedPtr visualizer;

//    visualizer = cloudViewer(cloudColor_1, "SE3D - Model Reconstruction");

//    // Add another cloud
//    pcl::visualization::PointCloudColorHandlerRGBField<PointDepthColor> handler(cloudColor_2);
//    visualizer->addPointCloud<PointDepthColor> (cloudColor_2, handler);

//    // Visualize the features
//        pcl::visualization::PointCloudColorHandlerCustom<PointDepth> handlerFeatures_1(cloudFeatures_1, 255, 0, 0);
//        pcl::visualization::PointCloudColorHandlerCustom<PointDepth> handlerFeatures_2(cloudFeatures_2, 0, 255, 0);

//        visualizer->addPointCloud<pcl::PointXYZ> (cloudFeatures_1, handlerFeatures_1, "features 1");
//        visualizer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "features 1");

//        visualizer->addPointCloud<pcl::PointXYZ> (cloudFeatures_2, handlerFeatures_2, "features 2");
//        visualizer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "features 2");

//        while (!visualizer->wasStopped ())
//        {
//            visualizer->spinOnce(100);
//            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//        }

    // Registration with ICP
    PointCloudDepthColorSharedPtr cloudFused(new PointCloudDepthColor);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloudColor_1, *cloudColor_1, indices);
    pcl::removeNaNFromPointCloud(*cloudColor_2, *cloudColor_2, indices);
    pcl::removeNaNFromPointCloud(*cloudFeatures_1, *cloudFeatures_1, indices);
    pcl::removeNaNFromPointCloud(*cloudFeatures_2, *cloudFeatures_2, indices);

    RegisterFrames(cloudColor_1, cloudColor_2, cloudFeatures_1, cloudFeatures_2, cloudFused, false);

    visualizer = cloudViewer(cloudFused, "SE3D - Model Reconstruction");

    while (!visualizer->wasStopped ())
    {
        visualizer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }


    cv::waitKey(0);

    std::cout << "SE3D terminated" << std::endl;

    return 0;

    //a.exec ();
}
