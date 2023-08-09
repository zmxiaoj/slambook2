//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

namespace myslam {

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulation(const std::vector<SE3> &poses,
                   const std::vector<Vec3> points, Vec3 &pt_world) {
	// poses 左右目相机图像位姿
	// 4x4
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i)
	{
		// [R|t]_{3x4}
        Mat34 m = poses[i].matrix3x4();
		// 从(2*i,0)开始选择一个1x4的矩阵块
		// x*P^3T-P^1T
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
		// y*P^3T-P^2T
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
	// 对A进行SVD分解
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
	// 取出SVD分解后最小奇异值对应的右奇异向量 4
	// 归一化
	// 取出前3个分量
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();
	// 判断SVD求解奇异值的质量
    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // 解质量不好，放弃
        return true;
		// 返回值与判断条件矛盾,修改后程序不能运行
	    // return false;
    }
    return false;
	// return true;
}

// converters
inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
