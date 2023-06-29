//
// Created by gaoxiang on 19-4-25.
//

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/surfel_smoothing.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/impl/mls.hpp>

#include <vector>
#include <iostream>
// typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointXYZRGBNormal SurfelT;
typedef pcl::PointCloud<SurfelT> SurfelCloud;
typedef pcl::PointCloud<SurfelT>::Ptr SurfelCloudPtr;

SurfelCloudPtr reconstructSurface(
        const PointCloudPtr &input, float radius, int polynomial_order) {
	// 创建MLS方法对象，确定输入、输出
    pcl::MovingLeastSquares<PointT, SurfelT> mls;
	// 创建KD树对象进行最近邻搜索，指针为tree，使用new进行内存分配
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	// 设置最近邻搜索方法，传入KD树对象的指针tree
    mls.setSearchMethod(tree);
	// 设置搜索半径
    mls.setSearchRadius(radius);
	// 设置需要计算法线信息
    mls.setComputeNormals(true);
	// 设置高斯权重函数的方差参数，计算每个点的权重
    mls.setSqrGaussParam(radius * radius);
	/*  已经弃用
    mls.setPolynomialFit(polynomial_order > 1);*/
	// 设置拟合多项式最高阶数
    mls.setPolynomialOrder(polynomial_order);
	// 设置输入点云
    mls.setInputCloud(input);
	// 创建输出对象指针
    SurfelCloudPtr output(new SurfelCloud);
	// 执行处理并将结果保存到输出
    mls.process(*output);
    return (output);
}

pcl::PolygonMeshPtr triangulateMesh(const SurfelCloudPtr &surfels) {
    // Create search tree*
    pcl::search::KdTree<SurfelT>::Ptr tree(new pcl::search::KdTree<SurfelT>);
	// 设置输入点云数据
    tree->setInputCloud(surfels);

    // Initialize objects
	// 初始化gp3算法对象
    pcl::GreedyProjectionTriangulation<SurfelT> gp3;
	// 创建指针保存三角化结果
    pcl::PolygonMeshPtr triangles(new pcl::PolygonMesh);

    // Set the maximum distance between connected points (maximum edge length)
	// 设置最大搜索半径，参数小产生更多小型三角面片，参数大产生更大三角面片
    gp3.setSearchRadius(0.05);

    // Set typical values for the parameters
    gp3.setMu(2.5);
	// 设置每个点的最大最近邻点数目，参数影响构建三角面片的点密度，值越大越密集
    gp3.setMaximumNearestNeighbors(100);
	// 设置最大表面角度，超过该角度的法线不连续
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
	// 最小三角形面片角度
    gp3.setMinimumAngle(M_PI / 18); // 10 degrees
	// 最大三角形面片角度
    gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
	// 设置法线一致性标志，提高三角化准确性
    gp3.setNormalConsistency(true);

    // Get result
    gp3.setInputCloud(surfels);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(*triangles);

    return triangles;
}

int main(int argc, char **argv) {

    // Load the points
    PointCloudPtr cloud(new PointCloud);
    if (argc == 0 || pcl::io::loadPCDFile(argv[1], *cloud)) {
        cout << "failed to load point cloud!";
        return 1;
    }
    cout << "point cloud loaded, points: " << cloud->points.size() << endl;

    // Compute surface elements
    cout << "computing normals ... " << endl;
    double mls_radius = 0.05, polynomial_order = 2;
    auto surfels = reconstructSurface(cloud, mls_radius, polynomial_order);

	/*
	 * 对处理后的点云进行可视化显示*/
	cout << "display clouds ... " << endl;
	pcl::visualization::PCLVisualizer viewer;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr clouds(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	for(auto it = surfels->begin(); it != surfels->end(); it++)
		clouds->push_back(*it);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(clouds);
	viewer.addPointCloud<pcl::PointXYZRGBNormal>(clouds, rgb, "clouds");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "clouds");
	viewer.spin();

    // Compute a greedy surface triangulation
    cout << "computing mesh ... " << endl;
    pcl::PolygonMeshPtr mesh = triangulateMesh(surfels);

    cout << "display mesh ... " << endl;
    pcl::visualization::PCLVisualizer vis;
    vis.addPolylineFromPolygonMesh(*mesh, "mesh frame");
    vis.addPolygonMesh(*mesh, "mesh");
    vis.resetCamera();
    vis.spin();

}