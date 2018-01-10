#ifndef RECORDER_H
#define RECORDER_H

#define NOMINMAX
#include <Windows.h>
#include <Kinect.h>

#include <pcl/io/boost.h>
#include <pcl/io/grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

namespace se3d
{
    #define S_CLOUD 0    // To save as point cloud
    #define S_DEPTH 1    // To save depth frame
    #define S_COLOR 2    // To save color frame
    #define S_ALL   3    // To save cloud + depth + color

    template<class Interface>
    inline void SafeRelease(Interface *& IRelease )
    {
        if (IRelease != NULL ) {
            IRelease->Release();
            IRelease = NULL;
        }
    }

    class Recorder : public pcl::Grabber
    {
    public:
        Recorder ();
        virtual ~Recorder () throw ();
        virtual void start ();
        virtual void stop ();
        virtual bool isRunning () const;
        virtual std::string getName () const;
        virtual float getFramesPerSecond () const;
        void saveFrame (UINT16 = S_CLOUD);
        typedef void (ptrFunc_renderPointCloud) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA>>& );

    protected:
        bool isStopped;
        bool isStarted;
        bool isSaveCloud;
        bool isSaveColor;
        bool isSaveDepth;
        unsigned int saveCloudCount;
        unsigned int saveColorCount;
        unsigned int saveDepthCount;
        int colorWidth;
        int colorHeight;
        int depthWidth;
        int depthHeight;
        std::vector<RGBQUAD> colorBuffer;
        std::vector<UINT16> depthBuffer;
        HRESULT result;
        IKinectSensor* sensor;
        ICoordinateMapper* mapper;
        IColorFrameSource* colorSource;
        IColorFrameReader* colorReader;
        IDepthFrameSource* depthSource;
        IDepthFrameReader* depthReader;
        boost::signals2::signal<ptrFunc_renderPointCloud>* signal_renderPointCloud;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr renderPointCloud(RGBQUAD* , UINT16* );
        boost::thread thread;
        mutable boost::mutex mutex;
        void threadFunction ();

    private:
        void writeCloudFile (const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& );
        void writeColorFile (const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& );
        void writeDepthFile (UINT16* );
    };
}

#endif // RECORDER_H
