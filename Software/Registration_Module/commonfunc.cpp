#include "commonfunc.h"
#include <opencv2/contrib/contrib.hpp>
#include <opencv/highgui.h>


// Feature detection function headers
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iostream>
#include <fstream>
commonFunc::commonFunc()
{
}

void commonFunc::dispDepthColor(cv::Mat &depthMap, cv::Mat &depthColor)
{
  double min;
  double max;

  // Get minimum and maximum values
  cv::minMaxIdx(depthMap, &min, &max);

  // Histogram Equalization
  cv::Mat adjMap;
  float scale = 255 / (max-min);
  depthMap.convertTo(adjMap,CV_8UC1, scale, -min*scale);

  // Convert to color map display
  cv::applyColorMap(adjMap, depthColor, cv::COLORMAP_JET);
}

void commonFunc::dispDepthGray(cv::Mat &depthMap, cv::Mat &depthGray)
{
  // normalize depth map to uint8 values
  cv::normalize(depthMap, depthGray, 0, 255, CV_MINMAX);
  depthGray.convertTo(depthGray, CV_8UC3);
}

// Convert color image to gray scale image
void commonFunc::rgb2gray(cv::Mat &rgbImg, cv::Mat &grayImg)
{
  cv::cvtColor(rgbImg, grayImg, cv::COLOR_RGB2GRAY);
}
// Flip image left to right
void commonFunc::flipImage(cv::Mat &rgbImg)
{
  cv::flip(rgbImg, rgbImg, 1);
}

////////////////////////////////////////////////////////////////
/////////////        Image feature detection       /////////////
////////////////////////////////////////////////////////////////
// Detect feature points on image
void commonFunc::featureExtraction
(cv::Mat &rgbImg, std::vector<cv::KeyPoint> &keyPts)
{
  std::cout<<"Start feature detection...\n";
  // Extract SIFT features
  cv::SiftFeatureDetector siftDetector(100, 5);
  cv::Mat grayImg; rgb2gray(rgbImg, grayImg);
  // Remember to convert RGB image to Grayscale image beforehand
  siftDetector.detect(grayImg, keyPts);

}

// Show detected keypoints on the color image
void commonFunc::showKeyPoints
(cv::Mat &rgbImg, std::vector<cv::KeyPoint> &keyPts){
  //-- Draw keypoints
  cv::Mat img_keyPts;
  drawKeypoints( rgbImg, keyPts, img_keyPts, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  //-- Show detected (drawn) keypoints
  cv::namedWindow("Keypoints", CV_WINDOW_NORMAL);
  cv::imshow("Keypoints", img_keyPts );
  cv::waitKey(0);
  std::cout<<"End feature detection...\n";
}

////////////////////////////////////////////////////////////////
///////////// Image feature detection and matching /////////////
////////////////////////////////////////////////////////////////
// This is a simplified version of the following codes
// https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html
//
// Understanding SIFT descriptor
// https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
//
// Distance metrics:
// https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html

// Detect and match features between two images
void commonFunc::featureMatching
(cv::Mat &rgb_1, std::vector<cv::KeyPoint> &keyPts_1,
 cv::Mat &rgb_2, std::vector<cv::KeyPoint> &keyPts_2,
 std::vector< cv::DMatch > *matches, bool robustMatch)
{
  std::cout<<"Start feature matching...\n";
    //-- Extract SURF features
    cv::SURF siftDetector;
    cv::Mat gray_1; rgb2gray(rgb_1, gray_1);
    cv::Mat gray_2; rgb2gray(rgb_2, gray_2);
    siftDetector.detect(gray_1, keyPts_1);
    siftDetector.detect(gray_2, keyPts_2);

  //-- Draw keypoints
  cv::Mat img_keyPts_1;
  cv::Mat img_keyPts_2;
  drawKeypoints( rgb_1, keyPts_1, img_keyPts_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
  drawKeypoints( rgb_2, keyPts_2, img_keyPts_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

  //-- Compute descriptor
  cv::SiftDescriptorExtractor siftDesExtractor;
  cv::Mat descriptors_1, descriptors_2;
  siftDesExtractor.compute( gray_1, keyPts_1, descriptors_1 );
  siftDesExtractor.compute( gray_2, keyPts_2, descriptors_2 );
  std::cout<<"descriptors type: "<<descriptors_1.type()<<std::endl;

  //-- Feature matching using descriptors
  cv::FlannBasedMatcher matcher;
  matcher.match( descriptors_1, descriptors_2, *matches );

  // SIFT descriptor has type 5: 32F
  //  +--------+----+----+----+----+------+------+------+------+
  //  |        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
  //  +--------+----+----+----+----+------+------+------+------+
  //  | CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
  //  | CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
  //  | CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
  //  | CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
  //  | CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
  //  | CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
  //  | CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
  //  +--------+----+----+----+----+------+------+------+------+
  std::cout<<"Matched descriptors: "<<matches->size()<<std::endl;

  if(robustMatch)
    {
      std::cout<<"\n Start outlier removal...\n";
      double max_dist = 0; double min_dist = 10000;
      //-- Quick calculation of max and min distances between keypoints
      for( int i = 0; i < descriptors_1.rows; i++ )
        { double dist = (*matches)[i].distance;
          if( dist < min_dist ) min_dist = dist;
          if( dist > max_dist ) max_dist = dist;
        }
      std::cout<<"max descriptor distance = "<<max_dist<<", min descriptor disance = " <<min_dist<<std::endl;
      // Only consider matches with small distances
      std::vector< cv::DMatch > good_matches;
      for( int i = 0; i < descriptors_1.rows; i++ )
        { if( (*matches)[i].distance <= cv::max(2*min_dist, 0.02) )
            { good_matches.push_back( (*matches)[i]); }
        }

      // Overwrite with the good matches.
      matches->clear();
      for(int i = 0; i<good_matches.size(); i++ )
        {
          matches->push_back( good_matches[i]);
        }
      std::cout<<"Good matches number: "<<matches->size()<<std::endl;
    }
  std::cout<<"End feature matching...\n";
}

// Draw the feature matches
void commonFunc::showFeatureMatches
(cv::Mat &rgb_1, std::vector<cv::KeyPoint> &keyPts_1,
 cv::Mat &rgb_2, std::vector<cv::KeyPoint> &keyPts_2,
 std::vector< cv::DMatch > *matches)
{
  //-- Show detected (drawn) matches
  cv::Mat img_matches;
  cv::drawMatches( rgb_1, keyPts_1, rgb_2, keyPts_2,
                   *matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Show detected matches
  cv::namedWindow("Feature Matches", CV_WINDOW_NORMAL);
  imshow( "Feature Matches", img_matches );
  cv::waitKey(0);
}

////////////////////////////////////////////////////////////////
/////////// RGB-D data to 3D Point Cloud Rendering /////////////
////////////////////////////////////////////////////////////////

void commonFunc::myDepth2meter(const float feat_x, const float feat_y, const float rawDisparity,
                             float &X, float &Y, float &Z)
{
  // reject invalid points
  if(rawDisparity <= 0)
    {
      X = 0; Y = 0; Z = 0; return;
    }

  float fx = 525.0; // focal length x
  float fy = 525.0; // focal length y
  float cx = 319.5; // optical center x
  float cy = 239.5; // optical center y
  float sclFactor = 5000.0;

  // Recall the camera projective projection model
  Z = rawDisparity / sclFactor;
  X = (feat_x - cx) * Z / fx;
  Y = (feat_y - cy) * Z / fy;
}

// https://openkinect.org/wiki/Imaging_Information
// minDistance = -10
// scaleFactor = .0021.
// These values were found by hand.
void commonFunc::depth2meter(const float feat_x, const float feat_y, const float rawDisparity,
                             float &X, float &Y, float &Z)
{
  float minDistance = -10;
  float scaleFactor = 0.0021;
  Z = 0.1236 * std::tan(rawDisparity / 2842.5 + 1.1863);
  X = (feat_x - 480 / 2) * (Z + minDistance) * scaleFactor;
  Y = (feat_y - 640 / 2) * (Z + minDistance) * scaleFactor;
}


////////////////////////////////////////////////////////////////
///////////// Kinect RGB-D to 3D Point Conversion  /////////////
////////////////////////////////////////////////////////////////
// Kinect's depth value is 16-bit, but only the low-12-bit store real distance

// Save feature points
void commonFunc::saveFeatures(const cv::Mat &rgb_1, const cv::Mat &depth_1, std::vector<cv::KeyPoint> &keyPts_1,
                              const cv::Mat &rgb_2, const cv::Mat &depth_2, std::vector<cv::KeyPoint> &keyPts_2,
                              std::vector< cv::DMatch > *matches)
{
  std::cout<<"type "<<depth_1.type()<<std::endl;
  // open file with name features.txt
  std::ofstream openFile_1 ("../QtOpenCVExample/kinect_data/features_1.txt", std::ios::out);
  std::ofstream openFile_2 ("../QtOpenCVExample/kinect_data/features_2.txt", std::ios::out);
  if (openFile_1.is_open() && openFile_2.is_open())
    {
      for(int i=0; i<matches->size(); i++)
        {
          // Get the index of matching pairs
          int idx_1 = (*matches)[i].queryIdx;
          int idx_2 = (*matches)[i].trainIdx;

          // Round feature positions
          int feat_1_x = (int) keyPts_1[idx_1].pt.y;
          int feat_1_y = (int) keyPts_1[idx_1].pt.x;
          int feat_2_x = (int) keyPts_2[idx_2].pt.y;
          int feat_2_y = (int) keyPts_2[idx_2].pt.x;

          // Be careful about the datatype
          unsigned short feat_1_depth = depth_1.at<unsigned short>(feat_1_x, feat_1_y);
          unsigned short feat_2_depth = depth_2.at<unsigned short>(feat_2_x, feat_2_y);
          float X_1, Y_1, Z_1;
          float X_2, Y_2, Z_2;

          // Convert image point (u,v)+depth to [X, Y, Z] 3D coordinate
          myDepth2meter(feat_1_x, feat_1_y, feat_1_depth, X_1, Y_1, Z_1);
          myDepth2meter(feat_2_x, feat_2_y, feat_2_depth, X_2, Y_2, Z_2);

          // Remove features which are out of Kinect senser range
          if(X_1>5 || Y_1 > 5 || Z_1 == 0.0 || X_2>5 || Y_2 > 5 || Z_2 == 0.0)
            {continue; }

          // Write out the key point data
          openFile_1 << X_1 <<" " << Y_1 << " "<< Z_1 << "\n";
          openFile_2 << X_2 <<" " << Y_2 << " "<< Z_2 << "\n";
        }
      openFile_1.close();
      openFile_2.close();
    }

}

// Convert rgb + depth images to colored point clouds
void commonFunc::rgbd2pointcloud(const cv::Mat &rgbImg, const cv::Mat &depthImg, const std::string &fileName)
{
  //
  std::ofstream openFile(fileName.c_str(), std::ios::out);
  if(openFile.is_open())
    {
      for(int i=0; i<rgbImg.rows; i++) // x
        {
          for(int j=0; j<rgbImg.cols; j++) // y
            {
              float X, Y, Z;
              unsigned short depth = depthImg.at<unsigned short>(i, j);
              // Render the 3D values
              myDepth2meter(i,j,depth, X, Y, Z);

              // Remove features which are out of Kinect senser range
              if(X>5 || Y > 5 || Z == 0.0){continue; }
              // Write out the colored 3D point
              openFile << X <<" " << Y << " "<< Z << " " <<(float)rgbImg.at<cv::Vec3b>(i,j)[0]
                       << " " <<(float)rgbImg.at<cv::Vec3b>(i,j)[1]<< " " <<(float)rgbImg.at<cv::Vec3b>(i,j)[2] <<"\n";
            }
        }
    }
  openFile.close();
}
