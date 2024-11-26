#include <iostream>

// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include "cylinder_segmentation.h"

// implementation based on https://github.com/isl-org/Open3D/blob/5f4985b6cff8ebd066b27ae6fb994f94a05d01e4/cpp/open3d/geometry/PointCloudFactory.cpp#L76
void generatePointCloudFromRGBDImage(
    const cv::Mat& rgbImage,
    const cv::Mat& depthImage,
    pcl::PointCloud<pcl::PointXYZRGB>& cloud,
//    pcl::PointCloud<pcl::PointXYZ>& cloud,
    std::vector<std::vector<float>> intrinsics)
{
    auto fx = intrinsics[0][0];
    auto fy = intrinsics[1][1];
    auto cx = intrinsics[0][2];
    auto cy = intrinsics[1][2];

    auto focal_length = std::make_pair(fx, fy);
    auto principal_point = std::make_pair(cx, cy);

    // double scale = 1.0;
    int num_valid_pixels = 0;
    for(int i = 0; i < depthImage.rows; ++i)
        for(int j = 0; j < depthImage.cols; ++j)
            if(depthImage.at<uint8_t>(i,j) > 0)
                num_valid_pixels++;

    cloud.resize(num_valid_pixels);

    int cnt = 0;
    for (int i = 0; i < depthImage.rows; ++i) {
        for (int j = 0; j < depthImage.cols; ++j) {
            auto depthValue = static_cast<float>(depthImage.at<uint8_t>(i, j)) /* * scale*/;
            if (depthValue > 0) {
                pcl::PointXYZRGB point;
//                pcl::PointXYZ point;

                point.z = depthValue;
                point.x = (j - principal_point.first) * point.z / focal_length.first;
                point.y = (i - principal_point.second) * point.z / focal_length.second;

                cv::Vec3b color = rgbImage.at<cv::Vec3b>(i, j);
                point.r = color[2];
                point.g = color[1];
                point.b = color[0];
                cloud[cnt++] = point;
            }
        }
    }

    Eigen::Matrix4f rotate {
        {1, 0, 0, 0},
        {0, -1, 0, 0},
        {0, 0, -1, 0},
        {0, 0, 0, 1}
    };
    pcl::transformPointCloud(cloud, cloud, rotate);
}

void segmentFromFile(const std::string& file_name)
{
    auto pointclouds = cylinderSegmentationFromFile(file_name);
    pcl::io::savePCDFileBinaryCompressed("/home/pawel/luxonis/cylinder-detection/pointclouds_segmentation_offline/pointcloud2.pcd", *pointclouds[0]);
    pcl::io::savePCDFileBinaryCompressed("/home/pawel/luxonis/cylinder-detection/pointclouds_segmentation_offline/pointcloud2_plane.pcd", *pointclouds[1]);
    pcl::io::savePCDFileBinaryCompressed("/home/pawel/luxonis/cylinder-detection/pointclouds_segmentation_offline/pointcloud2_cylinder.pcd", *pointclouds[2]);
}

int main() {
//     uncomment to perform segmentation from stored pcd file
//     segmentFromFile("/home/pawel/luxonis/cylinder-detection/pointclouds_segmentation/pointcloud2.pcd");
//     return 0;
    dai::Pipeline pipeline;

    // Define sources and outputs
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto depth = pipeline.create<dai::node::StereoDepth>();
    auto xoutDisparity = pipeline.create<dai::node::XLinkOut>();
    auto xoutRGB = pipeline.create<dai::node::XLinkOut>();

    xoutDisparity->setStreamName("disparity");
    xoutRGB->setStreamName("color");

    // Properties
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoLeft->setCamera("left");
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setCamera("right");

    // Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
    depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    // Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
    depth->setLeftRightCheck(true);
    depth->setExtendedDisparity(false);
    depth->setSubpixel(false);

    // Linking
    monoLeft->out.link(depth->left);
    monoRight->out.link(depth->right);
    depth->disparity.link(xoutDisparity->input);

    auto config = depth->initialConfig.get();
    config.postProcessing.speckleFilter.enable = false;
    config.postProcessing.speckleFilter.speckleRange = 50;
    config.postProcessing.temporalFilter.enable = true;
    config.postProcessing.spatialFilter.enable = true;
    config.postProcessing.spatialFilter.holeFillingRadius = 2;
    config.postProcessing.spatialFilter.numIterations = 1;
    config.postProcessing.thresholdFilter.minRange = 400;
    config.postProcessing.thresholdFilter.maxRange = 15000;
    config.postProcessing.decimationFilter.decimationFactor = 1;
    depth->initialConfig.set(config);

    auto camRGB = pipeline.create<dai::node::ColorCamera>();
    camRGB->setResolution(dai::ColorCameraProperties::SensorResolution::THE_800_P);
    camRGB->setIspScale(1, 3);
    camRGB->setColorOrder(dai::ColorCameraProperties::ColorOrder::RGB);
    camRGB->initialControl.setManualFocus(130);
    depth->setDepthAlign(dai::CameraBoardSocket::CAM_A);
    camRGB->isp.link(xoutRGB->input);
    // Connect to device and start pipeline
    dai::Device device(pipeline);

    device.setIrLaserDotProjectorIntensity(0.7);
    device.setIrFloodLightIntensity(0.1);

    // Output queue will be used to get the disparity frames from the outputs defined above
    auto qDisparity = device.getOutputQueue("disparity", 4, false);
    auto qRGB = device.getOutputQueue("color", 4, false);

    auto width = camRGB->getIspWidth();
    auto height = camRGB->getIspHeight();
    auto intrinsics = device.readCalibration().getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, width, height);

    auto counter = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    while(true) {
        auto depthImg = qDisparity->get<dai::ImgFrame>();
        auto rgbImg = qRGB->get<dai::ImgFrame>();

        // get img and depth from camera
        auto frameRGB = rgbImg->getCvFrame();
        auto frameDepth = depthImg->getCvFrame();
        frameDepth.convertTo(frameDepth, CV_8UC1, 255 / depth->initialConfig.getMaxDisparity());

        // because performing segmentation on every cycle was working too slow on my machine I decided to generate the pointcloud and process it only once per run to test and gather the data
        // for visualizing the pcds after the run I used https://imagetostl.com/view-pcd-online
        // I left the problem of optimizing the code, so it is possible to detect cylinders without delay for the future, focusing on developing working segmentation first
        if(counter++ == 180)
        {
            // convert img and depth to pointcloud
            generatePointCloudFromRGBDImage(frameRGB, frameDepth, *pointcloud, intrinsics);
            // process pointcloud to discover cylinder existing in the field of view
            auto pointclouds = cylinderSegmentation(pointcloud);

            // store pointclouds for further visualization
            pcl::io::savePCDFileBinaryCompressed("/home/pawel/luxonis/cylinder-detection/pointclouds_segmentation/pointcloud3.pcd", *pointcloud);
            pcl::io::savePCDFileBinaryCompressed("/home/pawel/luxonis/cylinder-detection/pointclouds_segmentation/pointcloud3_plane.pcd", *pointclouds.first);
            pcl::io::savePCDFileBinaryCompressed("/home/pawel/luxonis/cylinder-detection/pointclouds_segmentation/pointcloud3_cylinder.pcd", *pointclouds.second);
        }
        std::cout<<counter<<std::endl;

        // visualize camera view
        cv::imshow("depth", frameDepth);
        cv::applyColorMap(frameDepth, frameDepth, cv::COLORMAP_JET);
        cv::imshow("depthColor", frameDepth);
        cv::imshow("color", frameRGB);
        pointcloud->clear();

        cv::waitKey(1);
    }
}
