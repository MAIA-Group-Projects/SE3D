#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

namespace se3d
{
    // Default Kinect calibration params
    #define P_CENTERX   254.878f;
    #define P_CENTERY   205.395f;
    #define P_FOCALX    365.456f;
    #define P_FOCALY    365.456f;
    #define P_SCALE     1000.f;

    struct IntrinsicParams
    {
        float center_x;
        float center_y;
        float focal_x;
        float focal_y;
        float scale;
    };

    class Reconstructor
    {
    public:
        Reconstructor ();
        Reconstructor (cv::Mat , cv::Mat , cv::Mat , cv::Mat );
        ~Reconstructor ();
        void start();
        void displayKeypoints();
        void displayMatches (bool = true);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr getColorClouds (int );
        pcl::PointCloud<pcl::PointXYZ>::Ptr getFeaturesCloud (int );
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr getRegisteredCloud ();

    private:
        bool isPairMode;
        IntrinsicParams params;
        std::vector<cv::Mat> colorMats;
        std::vector<cv::Mat> depthMats;
        std::vector<cv::Mat> originalColorMats;
        std::vector<cv::KeyPoint> keypoints_1;
        std::vector<cv::KeyPoint> keypoints_2;
        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> filteredMatches;
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> featuresClouds;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr fusedCloud;

        bool isValidImages();
        void reformatImages();
        void detectKeyPoints ();
        void computeFeatures (cv::Mat& , cv::Mat& );
        void matchFeatures (cv::Mat& , cv::Mat& );
        void convertMatToCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr& , const cv::Mat , const cv::Mat );
        void convertFeaturesToCloud ();
        void registerClouds (float = .05f, bool = false);
        void convertToMeter (const float , const float , const float , float& , float& , float& );
        void copyCloudToXYZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr , pcl::PointCloud<pcl::PointXYZ>::Ptr );
    };
}

#endif // RECONSTRUCTOR_H
