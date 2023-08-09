//
// Created by gaoxiang on 19-5-4.
//

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

DEFINE_string(config_file, "./config/default.yaml", "config file path");

int main(int argc, char **argv) {
	// 用来解析命令行参数，默认argv是程序运行时的当前路径，因此此处应该是保证能够寻找到配置文件
    google::ParseCommandLineFlags(&argc, &argv, true);
	// 创建vo类，断言初始化成功，运行
    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(FLAGS_config_file));
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}
