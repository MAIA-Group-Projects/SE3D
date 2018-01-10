#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>

#include "recorder.h"
#include "reconstructor.h"

typedef pcl::PointXYZRGBA PointType;
typedef pcl::visualization::PCLVisualizer Visualizer;
typedef pcl::PointCloud<PointType>::ConstPtr SharedConstPtr;

class Utility
{
public:
    Utility();
    void printConsoleInstructions();
    void printRegistrationInstructions();
    void testPairRegistration(std::string , std::string , std::string , std::string );
    void testMultipleRegistration();
    void testRecorder();

    static void onKeyboardEvent (const pcl::visualization::KeyboardEvent& event, void* arg) {
        if (!event.keyDown()) {
            return;
        }
        std::string sym = event.getKeySym();
        if (sym == "q") {
            *(static_cast<int*>(arg)) = -1;
        } else if (sym == "space") {
            *(static_cast<int*>(arg)) = S_ALL;
        } else if (sym == "p") {
            *(static_cast<int*>(arg)) = S_CLOUD;
        } else if (sym == "c") {
            *(static_cast<int*>(arg)) = S_COLOR;
        } else if (sym == "d") {
            *(static_cast<int*>(arg)) = S_DEPTH;
        }
    }

    static pcl::visualization::PCLVisualizer::Ptr cloudViewer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                                               std::string cloudID, std::string title = "SE3D Viewer") {

        pcl::visualization::PCLVisualizer::Ptr viewer (new Visualizer(title));
        //viewer->setBackgroundColor(0, 0, 0);
        viewer->setCameraPosition(0.0, 0.0, -2.5, 0.0, 0.0, 0.0);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(cloud);
        viewer->addPointCloud<pcl::PointXYZRGB>(cloud);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reference cloud");
        viewer->addCoordinateSystem (0.10);
        viewer->initCameraParameters ();
        return (viewer);
    }
};

#endif // UTILITY_H
