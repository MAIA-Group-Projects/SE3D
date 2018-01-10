#include "utility.h"

Utility::Utility()
{

}

void Utility::printConsoleInstructions()
{
    std::cout << "\t.:. SE3D Console .:." << std::endl;
    std::cout << "(0) Exit program." << std::endl;
    std::cout << "(1) Run Kinect scan." << std::endl;
    std::cout << "(2) Run 3D Registration." << std::endl;
}

void Utility::printRegistrationInstructions()
{
    std::cout << "(1) Reconstruction from 2 views." << std::endl;
    std::cout << "(2) Reconstruction from multiple views." << std::endl;
}

void Utility::testPairRegistration(std::string color1, std::string color2, std::string depth1, std::string depth2)
{
    int key = 99;

    pcl::visualization::PCLVisualizer::Ptr viewer;

    cv::Mat colorMat1 = cv::imread(color1, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat colorMat2 = cv::imread(color2, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat depthMat1 = cv::imread(depth1, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat depthMat2 = cv::imread(depth2, CV_LOAD_IMAGE_UNCHANGED);

    cout << "Working on it ..." << endl;

    se3d::Reconstructor reconstructor(colorMat1, colorMat2, depthMat1, depthMat2);

    reconstructor.start();
    while (true) {
        system("cls");
        cout << "(0) Return to Menu" << endl;
        cout << "(1) Display Keypoints" << endl;
        cout << "(2) Display Matches" << endl;
        cout << "(3) Display Filetered Matches" << endl;
        cout << "(4) Display Features Cloud" << endl;
        cout << "(5) Display Registered Cloud" << endl;

        cin >> key;
        if (key == 0)
            break;

        if (key == 1) {
            reconstructor.displayKeypoints();
        } else if (key == 2) {
            reconstructor.displayMatches(false);
        } else if (key == 3) {
            reconstructor.displayMatches();
        } else if (key == 4) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 = reconstructor.getColorClouds(0);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 = reconstructor.getColorClouds(1);
            pcl::PointCloud<pcl::PointXYZ>::Ptr feature1 = reconstructor.getFeaturesCloud(0);
            pcl::PointCloud<pcl::PointXYZ>::Ptr feature2 = reconstructor.getFeaturesCloud(1);

            viewer = cloudViewer(cloud1, "Features Cloud");

            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(cloud2);
            viewer->addPointCloud<pcl::PointXYZRGB> (cloud2, handler);

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handlerFeatures_1(feature1, 255, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handlerFeatures_2(feature2, 0, 255, 0);

            viewer->addPointCloud<pcl::PointXYZ> (feature1, handlerFeatures_1, "features 1");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "features 1");

            viewer->addPointCloud<pcl::PointXYZ> (feature2, handlerFeatures_2, "features 2");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "features 2");

            while (!viewer->wasStopped ())
            {
                viewer->spinOnce(100);
                boost::this_thread::sleep (boost::posix_time::microseconds (100000));
            }
        } else if (key == 5) {
            viewer = cloudViewer(reconstructor.getRegisteredCloud(), "Fused", "SE3D - 3D Registration");

            while (!viewer->wasStopped ())
            {
                viewer->spinOnce (100);
                boost::this_thread::sleep (boost::posix_time::microseconds (100000));
            }
        }
    }
}

void Utility::testMultipleRegistration()
{
    std::cout << "In progress, solving some bugs, but output as the one during presentation and report . . ." << endl;
}

void Utility::testRecorder()
{
    cout << "Press [p] to save cloud." << endl;
    cout << "Press [c] to save color." << endl;
    cout << "Press [d] to save depth." << endl;
    cout << "Press [SPACE] to save cloud + color + depth." << endl;
    cout << "Press [q] to quit." << endl;

    bool closeViewer = false;
    int action = 99; // For tracking user input

    // Point cloud to be refreshed and displayed
    SharedConstPtr pointCloud;

    // Visualizer to display live point clouds from sensor
    boost::shared_ptr<Visualizer> viewer (new Visualizer("SE3D Recorder"));

    viewer->setCameraPosition(0.0, 0.0, -2.5, 0.0, 0.0, 0.0);
    viewer->registerKeyboardCallback(onKeyboardEvent, &action);

    // Kinect2 Recorder
    boost::shared_ptr<se3d::Recorder> recorder = boost::make_shared<se3d::Recorder>();

    // Retrieved Point Cloud Callback Function
    boost::mutex mutex;
    boost::function<void (const SharedConstPtr&)> callBackFunction;
    callBackFunction = [&mutex, &recorder, &pointCloud, &closeViewer, &action] (const SharedConstPtr& ptr) {
        boost::mutex::scoped_lock lock(mutex);
        pointCloud = ptr->makeShared();
        if (action == S_CLOUD || action == S_COLOR || action == S_DEPTH || action == S_ALL) {
            recorder->saveFrame(action);
            action = 99;
        } else if (action == -1) {
            // If user wants to close the viewer
            closeViewer = true;
        }
    };

    // Register Callback Function
    boost::signals2::connection connection = recorder->registerCallback(callBackFunction);

    // Start recorder
    recorder->start();

    while(!viewer->wasStopped()) {
        // Update Viewer
        viewer->spinOnce();
        boost::mutex::scoped_try_lock lock(mutex);
        if(lock.owns_lock() && pointCloud) {
            // Update Point Cloud
            if(!viewer->updatePointCloud(pointCloud)) {
                viewer->addPointCloud(pointCloud);
            }
        }
        if (closeViewer) {
            viewer->close();
        }
    }

    // Stop recorder
    recorder->stop();

    // Disconnect Callback Function
    if (connection.connected()) {
        connection.disconnect();
    }
}


