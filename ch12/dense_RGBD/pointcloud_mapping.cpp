#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    vector<Eigen::Isometry3d> poses;         // 相机位姿

    ifstream fin("./data/pose.txt");
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("./data/%s/%d.%s"); //图像文件格式
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // 使用-1读取原始图像

        double data[7] = {0};
        for (int i = 0; i < 7; i++) {
            fin >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "正在将图像转换为点云..." << endl;

    // 定义点云使用的格式：这里用的是XYZRGB
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    // 新建一个点云
	// 使用智能指针创建指向PointCloud对象的指针，用new分类内存实例化PointCloud
    PointCloud::Ptr pointCloud(new PointCloud);
    for (int i = 0; i < 5; i++) {
        PointCloud::Ptr current(new PointCloud);
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
				// 解引用指针得到(u-col,v-row)
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
				// 相机坐标系下空间点
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
				// 世界坐标系下空间点 T为相机坐标系到空间坐标系变换矩阵
                Eigen::Vector3d pointWorld = T * point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
				// BGR
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];
                current->points.push_back(p);
            }
        // depth filter and statistical removal
		// 创建临时点云
        PointCloud::Ptr tmp(new PointCloud);
		// 创建进行统计滤波的对象
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
		// 计算窗口内点云的平均距离，通过和阈值判断删除离群点
		// 设置平均邻域窗口大小为50
		// 减小参数，保留点云增加；增大参数，保留的点云减少
        statistical_filter.setMeanK(50);
		// 设置标准差倍数阈值(判断离群点阈值)为1.0
		// 减小参数阈值降低，删除的外点增加，保留点数目减少
        statistical_filter.setStddevMulThresh(1.0);
		// 设置滤波输入点云
        statistical_filter.setInputCloud(current);
		// 进行滤波并将结果存储到tmp
        statistical_filter.filter(*tmp);
		// 运算符重载，将滤波后的点云添加到最终结果
        (*pointCloud) += *tmp;
    }

    pointCloud->is_dense = false;
	int numTotal = pointCloud->size();
    cout << "点云共有" << numTotal << "个点" << endl;

    // voxel filter 体素降采样滤波器
    pcl::VoxelGrid<PointT> voxel_filter;
	// 体素网格分辨率 0.03*0.03*0.03 一个立方体晶格存放一个点
    double resolution = 0.03;
    voxel_filter.setLeafSize(resolution, resolution, resolution);       // resolution
	// 创建临时点云
	PointCloud::Ptr tmp(new PointCloud);
	// 设置待下采样点云
    voxel_filter.setInputCloud(pointCloud);
	// 滤波并保存结果
    voxel_filter.filter(*tmp);
	// 交换tmp和pointCloud内容
    tmp->swap(*pointCloud);

	int numFilter = pointCloud->size();
    cout << "滤波之后，点云共有" << numFilter << "个点" << endl;
	cout << "点云数量缩减了" << numTotal / numFilter << "倍" << endl;

    pcl::io::savePCDFileBinary("map" + (to_string(resolution)) +".pcd", *pointCloud);
	// 保存未使用降采样滤波器的结果
	//	pcl::io::savePCDFileBinary("map_novoxel.pcd", *pointCloud);
    return 0;
}