//
// Created by gaoxiang on 19-5-2.
//

#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

namespace myslam {

	Backend::Backend() {
		// 原子类型变量用store写入，用load读取
	    backend_running_.store(true);
		// 创建一个线程，线程执行的函数是BackendLoop，并将this绑定到函数，
		// 即这是this指向的类的成员函数
	    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
	}

	void Backend::UpdateMap() {
	    std::unique_lock<std::mutex> lock(data_mutex_);
		// 唤醒1个正在等待的线程
	    map_update_.notify_one();
	}

	void Backend::Stop() {
		// backend_running标志为false，唤醒一个wait的线程，等待后端线程结束
		// 后端结束时最后一次更新地图
	    backend_running_.store(false);
	    map_update_.notify_one();
		// 调用join()，当前线程会被阻塞，直到完成
	    backend_thread_.join();
	}

	void Backend::BackendLoop() {
		// load读取backend_running的值
		// 实际上当后端在运行时，这是一个死循环函数，但是会等待前端的激活
		// 即前端激活一次，就运行此函数，进行一次后端优化
	    while (backend_running_.load()) {
	        std::unique_lock<std::mutex> lock(data_mutex_);
	        map_update_.wait(lock);

	        /// 后端仅优化激活的Frames和Landmarks
	        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
	        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
	        Optimize(active_kfs, active_landmarks);
	    }
	}
	// keyframes是map类中的哈希表，左边编号右边关键帧，lanmarks同理
	void Backend::Optimize(Map::KeyframesType &keyframes,
	                       Map::LandmarksType &landmarks) {
	    // setup g2o
	    typedef g2o::BlockSolver_6_3 BlockSolverType;
	    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
	        LinearSolverType;
	    auto solver = new g2o::OptimizationAlgorithmLevenberg(
	        g2o::make_unique<BlockSolverType>(
	            g2o::make_unique<LinearSolverType>()));
	    g2o::SparseOptimizer optimizer;
	    optimizer.setAlgorithm(solver);

	    // pose 顶点，使用Keyframe id
	    std::map<unsigned long, VertexPose *> vertices;
	    unsigned long max_kf_id = 0;
	    for (auto &keyframe : keyframes) {
	        auto kf = keyframe.second;
	        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
	        vertex_pose->setId(kf->keyframe_id_);
	        vertex_pose->setEstimate(kf->Pose());
	        optimizer.addVertex(vertex_pose);
	        if (kf->keyframe_id_ > max_kf_id) {
	            max_kf_id = kf->keyframe_id_;
	        }

	        vertices.insert({kf->keyframe_id_, vertex_pose});
	    }

	    // 路标顶点，使用路标id索引
	    std::map<unsigned long, VertexXYZ *> vertices_landmarks;

	    // K 和左右外参
	    Mat33 K = cam_left_->K();
	    SE3 left_ext = cam_left_->pose();
	    SE3 right_ext = cam_right_->pose();

	    // edges
	    int index = 1;
	    double chi2_th = 5.991;  // robust kernel 阈值
	    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

	    for (auto &landmark : landmarks) {
		    // 路标点、特征点异常值跳过
	        if (landmark.second->is_outlier_) continue;
	        unsigned long landmark_id = landmark.second->id_;
	        auto observations = landmark.second->GetObs();
	        for (auto &obs : observations) {
	            if (obs.lock() == nullptr) continue;
	            auto feat = obs.lock();
	            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

	            auto frame = feat->frame_.lock();
	            EdgeProjection *edge = nullptr;
	            if (feat->is_on_left_image_) {
	                edge = new EdgeProjection(K, left_ext);
	            } else {
	                edge = new EdgeProjection(K, right_ext);
	            }

	            // 如果landmark还没有被加入优化，则新加一个顶点
	            if (vertices_landmarks.find(landmark_id) ==
	                vertices_landmarks.end()) {
	                VertexXYZ *v = new VertexXYZ;
	                v->setEstimate(landmark.second->Pos());
	                v->setId(landmark_id + max_kf_id + 1);
	                v->setMarginalized(true);
	                vertices_landmarks.insert({landmark_id, v});
	                optimizer.addVertex(v);
	            }
				// 连接顶点（相机位姿与路标点），测量值为关键点坐标，信息矩阵设为单位阵，huber核
		        edge->setId(index);
	            edge->setVertex(0, vertices.at(frame->keyframe_id_));    // pose
	            edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
	            edge->setMeasurement(toVec2(feat->position_.pt));
	            edge->setInformation(Mat22::Identity());
	            auto rk = new g2o::RobustKernelHuber();
	            rk->setDelta(chi2_th);
	            edge->setRobustKernel(rk);
	            edges_and_features.insert({edge, feat});

	            optimizer.addEdge(edge);

	            index++;
	        }
	    }

	    // do optimization and eliminate the outliers
	    optimizer.initializeOptimization();
	    optimizer.optimize(10);

	    int cnt_outlier = 0, cnt_inlier = 0;
	    int iteration = 0;
	    while (iteration < 5) {
	        cnt_outlier = 0;
	        cnt_inlier = 0;
	        // determine if we want to adjust the outlier threshold
	        for (auto &ef : edges_and_features) {
	            if (ef.first->chi2() > chi2_th) {
	                cnt_outlier++;
	            } else {
	                cnt_inlier++;
	            }
	        }
	        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
	        if (inlier_ratio > 0.5) {
	            break;
	        } else {
	            chi2_th *= 2;
	            iteration++;
	        }
	    }

	    for (auto &ef : edges_and_features) {
	        if (ef.first->chi2() > chi2_th) {
	            ef.second->is_outlier_ = true;
	            // remove the observation
	            ef.second->map_point_.lock()->RemoveObservation(ef.second);
	        } else {
	            ef.second->is_outlier_ = false;
	        }
	    }

	    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
	              << cnt_inlier;

	    // Set pose and lanrmark position
		// 修正相机位姿和路标点坐标
	    for (auto &v : vertices) {
	        keyframes.at(v.first)->SetPose(v.second->estimate());
	    }
	    for (auto &v : vertices_landmarks) {
	        landmarks.at(v.first)->SetPos(v.second->estimate());
	    }
	}

}  // namespace myslam