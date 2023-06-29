//
// Created by zmxj on 23-3-6.
//
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./distorted.png";
int main(int argc, char **argv)
{
	double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
	// 内参
	double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

	const double alpha0 = 0;
	const double alpha1 = 1;

	cv::Mat map1, map2;

	cv::Mat image = cv::imread(image_file, 0);
	int rows = image.rows, cols = image.cols;
	cv::Size imageSize(cols, rows);

	cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);
	cv::Mat image_undistort_filter = cv::Mat(rows, cols, CV_8UC1);
	cv::Mat image_undistort_func_uni = cv::Mat(rows, cols, CV_8UC1);
	cv::Mat image_undistort_func_map = cv::Mat(rows, cols, CV_8UC1);

	for (int v = 0; v < rows; v++)
	{
		for (int u = 0; u < cols; u++)
		{
			double x = (u - cx) / fx, y = (v - cy) / fy;
			double r = sqrt(x * x + y * y);
			double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
			double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
			double u_distorted = fx * x_distorted + cx;
			double v_distorted = fy * y_distorted + cy;

			// 判断畸变后的像素坐标是否还在像素矩阵内

			if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows)
			{
				//	cout << "u_distorted : " << u_distorted << " v_distorted: " << v_distorted << endl;
				//	cout << "u_distorted(int) : " << (int)u_distorted << " v_distorted(int) : " << (int)v_distorted << endl;

				/*	if (image.at<uchar>((int) v_distorted, (int) u_distorted) <= 128)
					{
						image_undistort.at<uchar>(v, u) = 255;
					}
					else
					{
						image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
					}*/
				image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);

				image_undistort_filter.at<uchar>(v, u) = (image.at<uchar>((int) v_distorted, (int) u_distorted)
				        + image.at<uchar>(((int) v_distorted) + 1, (int) u_distorted)
						+ image.at<uchar>((int) v_distorted, ((int) u_distorted) + 1)
						+ image.at<uchar>(((int) v_distorted) + 1, ((int) u_distorted) + 1))/4;

				//	cout << "I : " << (int )(image_undistort.at<uchar>(v, u)) << endl;
			}
			else
			{
				image_undistort.at<uchar>(v, u) = 0;
				// 取均值后的图像
				image_undistort_filter.at<uchar>(v, u) = 0;
			}
		}
	}
	cv::imshow("distorted", image);
	cv::imshow("undistorted", image_undistort);

	cv::imshow("undistorted_filter", image_undistort_filter);

	cv::Mat cameraMatrix = (cv::Mat_<double >(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
	cv::Mat distCoeffs = (cv::Mat_<double >(4, 1) << k1, k2, p1, p2);
	cv::undistort(image, image_undistort_func_uni, cameraMatrix, distCoeffs);
	cv::imshow("undistorted based on opencv undistort()", image_undistort_func_uni);

	cv::Mat newCameraMatrix1 = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha1, imageSize, 0);
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix1, imageSize, CV_8UC1, map1, map2);
	cv::remap(image, image_undistort_func_map, map1, map2, cv::INTER_LINEAR);
	cv::imshow("undistort based on opencv remap() alpha1", image_undistort_func_map);

	cv::undistort(image, image_undistort_func_uni, cameraMatrix, distCoeffs, newCameraMatrix1);
	cv::imshow("undistorted based on opencv undistort() newCameraMatrix1", image_undistort_func_uni);

	cv::Mat newCameraMatrix0 = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha0, imageSize, 0);
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix0, imageSize, CV_8UC1, map1, map2);
	cv::remap(image, image_undistort_func_map, map1, map2, cv::INTER_LINEAR);
	cv::imshow("undistort based on opencv remap() alpha0", image_undistort_func_map);

	cv::undistort(image, image_undistort_func_uni, cameraMatrix, distCoeffs, newCameraMatrix0);
	cv::imshow("undistorted based on opencv undistort() newCameraMatrix0", image_undistort_func_uni);

	cv::waitKey(0);
	return 0;
}