#include "reconstructor.h"

se3d::Reconstructor::Reconstructor(cv::Mat color1, cv::Mat color2, cv::Mat depth1, cv::Mat depth2)
{
    isPairMode = true;

    params.center_x = P_CENTERX;
    params.center_y = P_CENTERY;
    params.focal_x = P_FOCALX;
    params.focal_y = P_FOCALY;
    params.scale = P_SCALE;

    originalColorMats.push_back(color1);
    originalColorMats.push_back(color2);

    colorMats.push_back(color1);
    colorMats.push_back(color2);

    depthMats.push_back(depth1);
    depthMats.push_back(depth2);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2 (new pcl::PointCloud<pcl::PointXYZRGB>);
    clouds.push_back(cloud_1);
    clouds.push_back(cloud_2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr featureCloud_1 (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr featureCloud_2 (new pcl::PointCloud<pcl::PointXYZ>);
    featuresClouds.push_back(featureCloud_1);
    featuresClouds.push_back(featureCloud_2);

    fusedCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
}

se3d::Reconstructor::~Reconstructor()
{

}

se3d::Reconstructor::Reconstructor()
{

}

void se3d::Reconstructor::start()
{
    if(!isValidImages()) {
        std::cerr << "Reading one or more images failed." << std::endl;
        return;
    }

    cv::Mat descriptors_1, descriptors_2;

    reformatImages();

    detectKeyPoints();

    computeFeatures(descriptors_1, descriptors_2);

    matchFeatures(descriptors_1, descriptors_2);

    convertMatToCloud(clouds[0], originalColorMats[0], depthMats[0]);
    convertMatToCloud(clouds[1], originalColorMats[1], depthMats[1]);

    convertFeaturesToCloud();

    registerClouds();
}

void se3d::Reconstructor::displayKeypoints()
{
    if (!isPairMode) {
        return;
    }

    cv::Mat keypointsMat_1;
    cv::Mat keypointsMat_2;

    cv::drawKeypoints(colorMats[0], keypoints_1, keypointsMat_1,
            cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::drawKeypoints(colorMats[1], keypoints_2, keypointsMat_2,
            cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    std::string title = "Image 1 - key points";
    cv::namedWindow(title, CV_WINDOW_NORMAL);
    cv::imshow(title, keypointsMat_1);

    title = "Image 2 - key points";
    cv::namedWindow(title, CV_WINDOW_NORMAL);
    cv::imshow(title, keypointsMat_2);

    cv::waitKey(0);
}

void se3d::Reconstructor::displayMatches(bool filtered)
{
    if (!isPairMode) {
        return;
    }
    cv::Mat matchesMat;
    std::vector<cv::DMatch> m;

    if (filtered) {
        m = filteredMatches;
    } else {
        m = matches;
    }

    cv::drawMatches(colorMats[0], keypoints_1, colorMats[1], keypoints_2,
                    m, matchesMat, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    std::string title = "Matches of Image 1 & Image 2";
    cv::namedWindow(title, CV_WINDOW_NORMAL);
    cv::imshow(title, matchesMat);
    cv::waitKey(0);

}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr se3d::Reconstructor::getColorClouds(int i)
{
    return clouds[i];
}

pcl::PointCloud<pcl::PointXYZ>::Ptr se3d::Reconstructor::getFeaturesCloud(int i)
{
    return featuresClouds[i];
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr se3d::Reconstructor::getRegisteredCloud()
{
    return fusedCloud;
}

bool se3d::Reconstructor::isValidImages()
{
    // Check color images
    for(std::vector<cv::Mat>::iterator it = colorMats.begin(); it != colorMats.end(); ++it) {
        if (!it->data || it->channels() < 3 || it->channels() > 4) {
            return false;
        }
     }

    // Check depth images
    for(std::vector<cv::Mat>::iterator it = depthMats.begin(); it != depthMats.end(); ++it) {
        if (!it->data) {
            return false;
        }
     }

    return true;
}

void se3d::Reconstructor::reformatImages()
{
    // For color images
    for(std::vector<cv::Mat>::iterator it = colorMats.begin(); it != colorMats.end(); ++it) {
        if (it->channels() == 4) {
            cv::cvtColor(*it, *it, cv::COLOR_RGBA2BGR);
        }
        cv::cvtColor(*it, *it, cv::COLOR_RGB2GRAY);
        cv::flip(*it, *it, 1);
     }

    // For depth images
    for(std::vector<cv::Mat>::iterator it = depthMats.begin(); it != depthMats.end(); ++it) {
        if (it->depth() != CV_16U) {
             cv::cvtColor(*it, *it, cv::COLOR_RGB2GRAY);
             it->convertTo(*it, CV_16U);
        }
        cv::flip(*it, *it, 1);
    }
}

void se3d::Reconstructor::detectKeyPoints()
{
    // Detect keypoints with SURF
    cv::SURF detector;

    detector.detect(colorMats[0], keypoints_1);
    detector.detect(colorMats[1], keypoints_2);

}

void se3d::Reconstructor::computeFeatures(cv::Mat &descriptors_1, cv::Mat &descriptors_2)
{
    cv::SiftDescriptorExtractor extractor;

    extractor.compute(colorMats[0], keypoints_1, descriptors_1);
    extractor.compute(colorMats[1], keypoints_2, descriptors_2);
}

void se3d::Reconstructor::matchFeatures(cv::Mat &descriptors_1, cv::Mat &descriptors_2)
{
    double maxDist = 0;
    double minDist = 10000;
    double margin = 0.02;

    cv::FlannBasedMatcher matcher;

    matcher.match(descriptors_1, descriptors_2, matches);

    // Compute max and min distances between keypoints
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) {
            minDist = dist;
        }
        if (dist > maxDist) {
            maxDist = dist;
        }
    }

    // Keep only matches with small distances
    bool isROI;
    for(int i = 0; i < descriptors_1.rows; i++) {
        int idx = matches[i].queryIdx;
        int feat_x = (int) keypoints_1[idx].pt.x;
        int feat_y = (int) keypoints_1[idx].pt.y;

        isROI = colorMats[0].at<unsigned char>(feat_y, feat_x) != 0;
        if (matches[i].distance <= cv::max(2*minDist, margin) && isROI) {
            filteredMatches.push_back(matches[i]);
        }
    }
}

void se3d::Reconstructor::convertMatToCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                            const cv::Mat colorM, const cv::Mat depthM)
{
    for(int i = 0; i < colorM.rows; i++) {
        for(int j = 0; j < colorM.cols; j++) {
            float X, Y, Z;

            unsigned short depth = depthM.at<unsigned short>(i, j);

            // Render the 3D values
            convertToMeter(i, j, depth, X, Y, Z);

            // Remove features which are out of Kinect senser range
            if (X > 5 || Y > 5 || Z == 0.0) {
                continue;
            }

            // Write out the colored 3D point
            float R = (float) colorM.at<cv::Vec3b>(i,j)[0];
            float G = (float) colorM.at<cv::Vec3b>(i,j)[1];
            float B = (float) colorM.at<cv::Vec3b>(i,j)[2];

            // Push back the 3D point
            pcl::PointXYZRGB point;
            point.x = X; point.y = Y; point.z = Z;
            point.r = R; point.g = G; point.b = B;
            (*cloud).push_back(point);
        }
    }
}

void se3d::Reconstructor::convertFeaturesToCloud()
{
    for (int i = 0; i < filteredMatches.size(); i++) {
        // Get the index of matching pairs
        int idx_1 = filteredMatches[i].queryIdx;
        int idx_2 = filteredMatches[i].trainIdx;

        // Round feature positions
        int feat_1_x = (int) keypoints_1[idx_1].pt.y;
        int feat_1_y = (int) keypoints_1[idx_1].pt.x;

        int feat_2_x = (int) keypoints_2[idx_2].pt.y;
        int feat_2_y = (int) keypoints_2[idx_2].pt.x;

        unsigned short feat_1_z = depthMats[0].at<unsigned short>(feat_1_x, feat_1_y);
        unsigned short feat_2_z = depthMats[1].at<unsigned short>(feat_2_x, feat_2_y);

        // Convert image point (u,v)+depth to [X, Y, Z] 3D coordinate
        float X_1, Y_1, Z_1;
        float X_2, Y_2, Z_2;

        convertToMeter(feat_1_x, feat_1_y, feat_1_z, X_1, Y_1, Z_1);
        convertToMeter(feat_2_x, feat_2_y, feat_2_z, X_2, Y_2, Z_2);

        // Remove features which are out of Kinect senser range
        if (X_1 > 5 || Y_1 > 5 || Z_1 == 0.0 || X_2 > 5 || Y_2 > 5 || Z_2 == 0.0) {
            continue;
        }

        pcl::PointXYZ point;

        point.x = X_1;
        point.y = Y_1;
        point.z = Z_1;
        (*featuresClouds[0]).push_back(point);

        point.x = X_2;
        point.y = Y_2;
        point.z = Z_2;
        (*featuresClouds[1]).push_back(point);
    }
}

void se3d::Reconstructor::registerClouds(float threshold, bool refine)
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    icp.setInputCloud(featuresClouds[1]);
    icp.setInputTarget(featuresClouds[0]);
    icp.setRANSACOutlierRejectionThreshold(threshold);

    pcl::PointCloud<pcl::PointXYZ> alignedCloud;
    icp.align(alignedCloud);

    // Estimate the rigid transformation
    Eigen::Matrix4f transformMat = icp.getFinalTransformation();

    // Transform the sample cloud to reference cloud coordinate
    pcl::transformPointCloud(*clouds[1], *fusedCloud, transformMat);

    // Registration refinement using ICP with all points
    if (refine) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

        copyCloudToXYZ(fusedCloud, cloud_in);
        copyCloudToXYZ(clouds[0], cloud_out);

        icp.setInputCloud(cloud_in);
        icp.setInputTarget(cloud_out);
        icp.align(alignedCloud);

        transformMat = icp.getFinalTransformation();

        pcl::transformPointCloud(*fusedCloud, *fusedCloud, transformMat);
    }

    // Fuse with the reference cloud
    for (int i = 0; i < clouds[0]->size(); i++) {
        fusedCloud->push_back(clouds[0]->at(i));
    }
}

void se3d::Reconstructor::convertToMeter(const float feat_x, const float feat_y, const float depth, float &X, float &Y, float &Z)
{
    if (depth > 0) {
        Z = depth / params.scale;
        X = (feat_x - params.center_x) * Z / params.focal_x;
        Y = (feat_y - params.center_y) * Z / params.focal_y;
    } else {
        Z = 0;
        X = 0;
        Y = 0;
    }
}

void se3d::Reconstructor::copyCloudToXYZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out)
{
    for (int i = 0; i < cloud_in->size(); i++) {
        pcl::PointXYZ point;
        point.x = cloud_in->at(i).r / 255.0;
        point.y = cloud_in->at(i).g / 255.0;
        point.z = cloud_in->at(i).b / 255.0;
        cloud_out->push_back(point);
    }
}
