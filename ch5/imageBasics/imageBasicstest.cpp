//
// Created by zmxj on 23-3-6.
//
#include <iostream>
#include <chrono>

using namespace std;

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>

int main(int argc, char **argv)
{
    cv::Mat image;
    image = cv::imread(argv[1]);

    if (image.data == nullptr)
    { //数据不存在,可能是文件不存在
        cerr << "文件" << argv[1] << "不存在." << endl;
        return 0;
    }

    cout << "width:" << image.cols << "height:" << image.rows << endl;
    cv::imshow("Image", image);
    cv::waitKey(0);

    if(image.type() != CV_8UC1 && image.type() != CV_8UC3)
    {
        cout << "Wrong image type!" << endl;
        return 0;
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (size_t y = 0; y < image.rows; y++)
    {
		uchar *row_ptr = image.ptr<uchar>(y);
	    for (size_t x = 0; x < image.rows; x++)
	    {
			uchar *data_ptr = &row_ptr[x * image.channels()];

		    for (int c = 0; c != image.channels() ; c++)
		    {
				uchar data = data_ptr[c];
		    }
	    }
    }
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration < double >>(t2 - t1);
	cout << "Time_used : " << time_used.count() << "s" << endl;

	cv::Mat image_clone = image.clone();
	image_clone(cv::Rect(0, 0, 300, 300)).setTo(25);
	cv::imshow("Image", image);
	cv::imshow("Image_clone", image_clone);
	cv::waitKey(0);

	cv::destroyAllWindows();
    return 0;
}