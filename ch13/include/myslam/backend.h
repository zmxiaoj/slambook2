//
// Created by gaoxiang on 19-5-2.
//

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
class Map;

/**
 * 后端
 * 有单独优化线程，在Map更新时启动优化
 * Map更新由前端触发
 */ 
class Backend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    /// 构造函数中启动优化线程并挂起
    Backend();

    // 设置左右目的相机，用于获得内外参
    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        cam_left_ = left;
        cam_right_ = right;
    }

    /// 设置地图，让backend自己的地图指针指向当前的地图，而不是对当前地图进行修改，不需要锁
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    /// 触发地图更新，启动优化(notify)
    void UpdateMap();

    /// 关闭后端线程
    void Stop();

   private:
    /// 后端线程
    void BackendLoop();

    /// 对给定关键帧和路标点进行优化
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    std::shared_ptr<Map> map_;
    std::thread backend_thread_;
    std::mutex data_mutex_;
    // 条件变量
    std::condition_variable map_update_;
    // std::atomic 是模板类，一个模板类型为 T 的原子对象中封装了一个类型为 T 的值
    // 原子类型对象不同线程同时访问不会产生数据竞争
    std::atomic<bool> backend_running_;

    Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
};

}  // namespace myslam

#endif  // MYSLAM_BACKEND_H