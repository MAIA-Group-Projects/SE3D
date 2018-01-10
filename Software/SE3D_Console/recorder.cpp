#include "recorder.h"

se3d::Recorder::Recorder() {
    sensor = nullptr;
    mapper = nullptr;
    colorSource = nullptr;
    colorReader = nullptr;
    depthSource = nullptr;
    depthReader = nullptr;
    result = S_OK;
    colorWidth = 1920;
    colorHeight = 1080;
    depthWidth = 512;
    depthHeight = 424;
    isStarted = false;
    isStopped = false;
    isSaveCloud = false;
    isSaveColor = false;
    isSaveDepth = false;
    saveCloudCount = 0;
    saveColorCount = 0;
    saveDepthCount = 0;
    signal_renderPointCloud = nullptr;

    // Create Sensor Instance
    result = GetDefaultKinectSensor(&sensor );
    if (FAILED(result ) ) {
        throw std::exception("Exception : GetDefaultKinectSensor()" );
    }

    // Open Sensor
    result = sensor->Open();
    if (FAILED(result ) ) {
        throw std::exception("Exception : IKinectSensor::Open()" );
    }

    // Retrieved Coordinate Mapper
    result = sensor->get_CoordinateMapper(&mapper );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IKinectSensor::get_CoordinateMapper()" );
    }

    // Retrieved Color Frame Source
    result = sensor->get_ColorFrameSource(&colorSource );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IKinectSensor::get_ColorFrameSource()" );
    }

    // Retrieved Depth Frame Source
    result = sensor->get_DepthFrameSource(&depthSource );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IKinectSensor::get_DepthFrameSource()" );
    }

    // Retrieved Color Frame Size
    IFrameDescription* colorDescription;
    result = colorSource->get_FrameDescription(&colorDescription );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IColorFrameSource::get_FrameDescription()" );
    }

    result = colorDescription->get_Width(&colorWidth );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IFrameDescription::get_Width()" );
    }

    result = colorDescription->get_Height(&colorHeight );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IFrameDescription::get_Height()" );
    }

    SafeRelease(colorDescription );

    // To Reserve Color Frame Buffer
    colorBuffer.resize(colorWidth * colorHeight );

    // Retrieved Depth Frame Size
    IFrameDescription* depthDescription;
    result = depthSource->get_FrameDescription(&depthDescription );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IDepthFrameSource::get_FrameDescription()" );
    }

    result = depthDescription->get_Width(&depthWidth );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IFrameDescription::get_Width()" );
    }

    result = depthDescription->get_Height(&depthHeight );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IFrameDescription::get_Height()" );
    }

    SafeRelease(depthDescription );

    // To Reserve Depth Frame Buffer
    depthBuffer.resize(depthWidth * depthHeight );

    // Create a signal
    signal_renderPointCloud = createSignal<ptrFunc_renderPointCloud>();
}

se3d::Recorder::~Recorder() throw()
{
    stop();

    disconnect_all_slots<ptrFunc_renderPointCloud>();

    thread.join();

    // End Processing
    if (sensor) {
        sensor->Close();
    }
    SafeRelease(sensor );
    SafeRelease(mapper );
    SafeRelease(colorSource );
    SafeRelease(colorReader );
    SafeRelease(depthSource );
    SafeRelease(depthReader );
}

void se3d::Recorder::start()
{
    // Open Color Frame Reader
    result = colorSource->OpenReader(&colorReader );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IColorFrameSource::OpenReader()" );
    }

    // Open Depth Frame Reader
    result = depthSource->OpenReader(&depthReader );
    if (FAILED(result ) ) {
        throw std::exception("Exception : IDepthFrameSource::OpenReader()" );
    }

    isStarted = true;

    thread = boost::thread(&Recorder::threadFunction, this );
}

void se3d::Recorder::stop()
{
    boost::unique_lock<boost::mutex> lock(mutex );

    isStopped = true;
    isStarted = false;

    lock.unlock();
}

bool se3d::Recorder::isRunning() const
{
    boost::unique_lock<boost::mutex> lock(mutex );

    return isStarted;

    lock.unlock();
}

std::string se3d::Recorder::getName() const
{
    return std::string("SE3D::Recrder .:. Kinect Interface");
}

float se3d::Recorder::getFramesPerSecond() const
{
    return 0.f;
}

void se3d::Recorder::saveFrame(UINT16 code)
{
    switch(code) {
        case S_CLOUD:
            isSaveCloud = true;
            break;
        case S_DEPTH:
            isSaveDepth = true;
            break;
        case S_COLOR:
            isSaveColor = true;
            break;
        case S_ALL:
            isSaveCloud = true;
            isSaveColor = true;
            isSaveDepth = true;
            break;
        default:
            throw std::exception("Exception : invalid argument" );
            break;
    }
}

void se3d::Recorder::writeCloudFile(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud)
{
    std::stringstream stream;
    stream << "cloud" << saveCloudCount << ".pcd";
    std::string filename = stream.str();
    if (pcl::io::savePCDFile(filename, *cloud, true) == 0) {
        saveCloudCount++;
        std::cout << "Saved: " << filename << std::endl;
    } else {
        PCL_ERROR("Problem saving %s.\n", filename.c_str());
    }
}

void se3d::Recorder::writeColorFile(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud)
{
    if (!cloud->empty()) {
        unsigned int rows = cloud->height;
        unsigned int cols = cloud->width;

        cv::Mat colorImg = cv::Mat(cloud->height, cloud->width, CV_8UC3);

        #pragma omp parallel for
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                pcl::PointXYZRGBA point = cloud->at( x, y );
                // Set color image matrix ('a' is for transparency)
                colorImg.at<cv::Vec3b>( y, x )[0] = point.b;
                colorImg.at<cv::Vec3b>( y, x )[1] = point.g;
                colorImg.at<cv::Vec3b>( y, x )[2] = point.r;
            }
        }

        std::stringstream stream;
        stream << "color" << saveColorCount << ".png";
        std::string filename = stream.str();
        if (cv::imwrite(filename, colorImg)) {
            saveColorCount++;
            std::cout << "Saved: " << filename << std::endl;
        } else {
            std::cout << "Saving failed: " << filename << std::endl;
        }

    }
}

void se3d::Recorder::writeDepthFile(UINT16 *data)
{
    cv::Mat depthImg = cv::Mat(depthHeight, depthWidth, CV_16UC1, const_cast<unsigned short *>(data));

    std::stringstream stream;
    stream << "depth" << saveDepthCount << ".png";
    std::string filename = stream.str();
    if (cv::imwrite(filename, depthImg)) {
        saveDepthCount++;
        std::cout << "Saved: " << filename << std::endl;
    } else {
        std::cout << "Saving failed: " << filename << std::endl;
    }
}

void se3d::Recorder::threadFunction()
{
    while(!isStopped ) {
        boost::unique_lock<boost::mutex> lock(mutex );

        // Acquire Latest Color Frame
        IColorFrame* colorFrame = nullptr;
        result = colorReader->AcquireLatestFrame(&colorFrame );
        if (SUCCEEDED(result ) ) {
            // Retrieved Color Data
            result = colorFrame->CopyConvertedFrameDataToArray(colorBuffer.size() * sizeof(RGBQUAD ), reinterpret_cast<BYTE*>(&colorBuffer[0] ), ColorImageFormat::ColorImageFormat_Bgra );
            if (FAILED(result ) ) {
                throw std::exception("Exception : IColorFrame::CopyConvertedFrameDataToArray()" );
            }
        }
        SafeRelease(colorFrame );

        // Acquire Latest Depth Frame
        IDepthFrame* depthFrame = nullptr;
        result = depthReader->AcquireLatestFrame(&depthFrame );
        if (SUCCEEDED(result ) ) {
            // Retrieved Depth Data
            result = depthFrame->CopyFrameDataToArray(depthBuffer.size(), &depthBuffer[0] );
            if (FAILED(result ) ) {
                throw std::exception("Exception : IDepthFrame::CopyFrameDataToArray()" );
            }
        }
        SafeRelease(depthFrame );


        lock.unlock();

        if (signal_renderPointCloud->num_slots() > 0 ) {
            signal_renderPointCloud->operator()(renderPointCloud(&colorBuffer[0], &depthBuffer[0]));
        }
    }
}


pcl::PointCloud<pcl::PointXYZRGBA>::Ptr se3d::Recorder::renderPointCloud(RGBQUAD* colorBuffer, UINT16* depthBuffer)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>() );

    cloud->width = static_cast<uint32_t>(depthWidth);
    cloud->height = static_cast<uint32_t>(depthHeight);
    cloud->is_dense = false;

    cloud->points.resize(cloud->height * cloud->width );

    pcl::PointXYZRGBA* pt = &cloud->points[0];
    for(int y = 0; y < depthHeight; y++ ) {
        for(int x = 0; x < depthWidth; x++, pt++ ) {
            float Zmin = 0.1;
            float Zmax = 1.5;
            //float Zmin = 0.1;
            //float Zmax = 1.1;

            pcl::PointXYZRGBA point;

            DepthSpacePoint depthSpacePoint = { static_cast<float>(x ), static_cast<float>(y ) };
            UINT16 depth = depthBuffer[y * depthWidth + x];

            // Coordinate Mapping Depth to Color Space, and Setting PointCloud RGBA
            ColorSpacePoint colorSpacePoint = { 0.0f, 0.0f };
            mapper->MapDepthPointToColorSpace(depthSpacePoint, depth, &colorSpacePoint );

            // Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZ
            CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
            mapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint );

            int colorX = static_cast<int>(std::floor(colorSpacePoint.X + 0.5f ) );
            int colorY = static_cast<int>(std::floor(colorSpacePoint.Y + 0.5f ) );

            if ((cameraSpacePoint.Z > Zmin)  && (cameraSpacePoint.Z < Zmax) && (0 <= colorX) && (colorX < colorWidth ) && (0 <= colorY ) && (colorY < colorHeight ) ) {
                RGBQUAD color = colorBuffer[colorY * colorWidth + colorX];
                point.b = color.rgbBlue;
                point.g = color.rgbGreen;
                point.r = color.rgbRed;
                point.a = color.rgbReserved;

                point.x = cameraSpacePoint.X;
                point.y = cameraSpacePoint.Y;
                point.z = cameraSpacePoint.Z;
            }
            *pt = point;
        }
    }

    if (isSaveCloud) {
        writeCloudFile(cloud);
        isSaveCloud = false;
    }

    if (isSaveColor) {
        writeColorFile(cloud);
        isSaveColor = false;
    }

    if (isSaveDepth) {
        writeDepthFile(depthBuffer);
        isSaveDepth = false;
    }

    return cloud;
}
