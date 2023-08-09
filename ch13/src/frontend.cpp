//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

	Frontend::Frontend() {
	    gftt_ =
	        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
	    num_features_init_ = Config::Get<int>("num_features_init");
	    num_features_ = Config::Get<int>("num_features");
	}

	bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
	    current_frame_ = frame;

	    switch (status_) {
	        case FrontendStatus::INITING:
	            StereoInit();
	            break;
	        case FrontendStatus::TRACKING_GOOD:
	        case FrontendStatus::TRACKING_BAD:
	            Track();
	            break;
	        case FrontendStatus::LOST:
	            Reset();
	            break;
	    }

	    last_frame_ = current_frame_;
	    return true;
	}

	bool Frontend::Track() {
	    if (last_frame_) {
			// 计算当前帧的位姿
	        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
	    }
		// 跟踪前一帧特征
	    int num_track_last = TrackLastFrame();
		// 计算当前帧的内点数
	    tracking_inliers_ = EstimateCurrentPose();

		// 根据内点数判断跟踪状态
	    if (tracking_inliers_ > num_features_tracking_) {
	        // tracking good
	        status_ = FrontendStatus::TRACKING_GOOD;
	    }
		else if (tracking_inliers_ > num_features_tracking_bad_) {
	        // tracking bad
	        status_ = FrontendStatus::TRACKING_BAD;
	    }
		else {
	        // lost
	        status_ = FrontendStatus::LOST;
	    }
		// 插入关键帧
	    InsertKeyframe();
		// 计算当前帧相对上一帧的相对运动
	    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();
		// 将当前帧加入可视化程序
	    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
	    return true;
	}

	bool Frontend::InsertKeyframe() {
	    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
	        // still have enough features, don't insert keyframe
	        return false;
	    }
	    // current frame is a new keyframe
	    current_frame_->SetKeyFrame();
	    map_->InsertKeyFrame(current_frame_);

	    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
	              << current_frame_->keyframe_id_;

	    SetObservationsForKeyFrame();
	    DetectFeatures();  // detect new features

	    // track in right image
		// 找到特征在右图的对应
	    FindFeaturesInRight();
	    // triangulate map points
		// 三角化
	    TriangulateNewPoints();
	    // update backend because we have a new keyframe
	    backend_->UpdateMap();

	    if (viewer_) viewer_->UpdateMap();

	    return true;
	}

	void Frontend::SetObservationsForKeyFrame() {
	    for (auto &feat : current_frame_->features_left_)
		{
	        auto mp = feat->map_point_.lock();
	        if (mp)
				mp->AddObservation(feat);
	    }
	}

	int Frontend::TriangulateNewPoints() {
	    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
	    SE3 current_pose_Twc = current_frame_->Pose().inverse();
	    int cnt_triangulated_pts = 0;
	    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i)
		{
	        if (current_frame_->features_left_[i]->map_point_.expired() &&
	            current_frame_->features_right_[i] != nullptr) {
	            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
	            std::vector<Vec3> points{
	                camera_left_->pixel2camera(
	                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
	                         current_frame_->features_left_[i]->position_.pt.y)),
	                camera_right_->pixel2camera(
	                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
	                         current_frame_->features_right_[i]->position_.pt.y))};
	            Vec3 pworld = Vec3::Zero();

				// 判断条件是三角化成功
	            if (triangulation(poses, points, pworld) && pworld[2] > 0)
				{
	                auto new_map_point = MapPoint::CreateNewMappoint();
	                pworld = current_pose_Twc * pworld;
	                new_map_point->SetPos(pworld);
	                new_map_point->AddObservation(
	                    current_frame_->features_left_[i]);
	                new_map_point->AddObservation(
	                    current_frame_->features_right_[i]);

	                current_frame_->features_left_[i]->map_point_ = new_map_point;
	                current_frame_->features_right_[i]->map_point_ = new_map_point;
	                map_->InsertMapPoint(new_map_point);
	                cnt_triangulated_pts++;
	            }
	        }
	    }
	    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
	    return cnt_triangulated_pts;
	}

	int Frontend::EstimateCurrentPose() {
	    // setup g2o
	    typedef g2o::BlockSolver_6_3 BlockSolverType;
	    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
	        LinearSolverType;
	    auto solver = new g2o::OptimizationAlgorithmLevenberg(
	        g2o::make_unique<BlockSolverType>(
	            g2o::make_unique<LinearSolverType>()));
	    g2o::SparseOptimizer optimizer;
	    optimizer.setAlgorithm(solver);

	    // vertex
	    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
	    vertex_pose->setId(0);
	    vertex_pose->setEstimate(current_frame_->Pose());
	    optimizer.addVertex(vertex_pose);

	    // K
	    Mat33 K = camera_left_->K();

	    // edges
	    int index = 1;
	    std::vector<EdgeProjectionPoseOnly *> edges;
	    std::vector<Feature::Ptr> features;
		// 遍历当前帧的特征
	    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i)
		{
			// 对于每个特征对应的地图点设置为特征的edge(约束)
	        auto mp = current_frame_->features_left_[i]->map_point_.lock();
	        if (mp)
			{
	            features.push_back(current_frame_->features_left_[i]);
	            EdgeProjectionPoseOnly *edge =
	                new EdgeProjectionPoseOnly(mp->pos_, K);
	            edge->setId(index);
	            edge->setVertex(0, vertex_pose);
	            edge->setMeasurement(
	                toVec2(current_frame_->features_left_[i]->position_.pt));
	            edge->setInformation(Eigen::Matrix2d::Identity());
	            edge->setRobustKernel(new g2o::RobustKernelHuber);
	            edges.push_back(edge);
	            optimizer.addEdge(edge);
	            index++;
	        }
	    }

	    // estimate the Pose the determine the outliers
		// 卡方检测(chi-square)阈值
	    const double chi2_th = 5.991;
	    int cnt_outlier = 0;
	    for (int iteration = 0; iteration < 4; ++iteration)
		{
	        vertex_pose->setEstimate(current_frame_->Pose());
	        optimizer.initializeOptimization();
	        optimizer.optimize(10);
	        cnt_outlier = 0;

	        // count the outliers
			// 遍历所有边
	        for (size_t i = 0; i < edges.size(); ++i)
			{
	            auto e = edges[i];
	            if (features[i]->is_outlier_)
				{
	                e->computeError();
	            }
	            if (e->chi2() > chi2_th)
				{
	                features[i]->is_outlier_ = true;
					// 设置边的优化级别 1
					// 边会被视为异常值，并且在优化过程中被鲁棒地处理
	                e->setLevel(1);
	                cnt_outlier++;
	            }
				else
				{
	                features[i]->is_outlier_ = false;
					// 设置边的优化级别 0
	                e->setLevel(0);
	            };
				// 取消鲁棒核函数
	            if (iteration == 2)
				{
	                e->setRobustKernel(nullptr);
	            }
	        }
	    }

	    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
	              << features.size() - cnt_outlier;
	    // Set pose and outlier
	    current_frame_->SetPose(vertex_pose->estimate());

	    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

	    for (auto &feat : features) {
	        if (feat->is_outlier_) {
	            feat->map_point_.reset();
	            feat->is_outlier_ = false;  // maybe we can still use it in future
	        }
	    }
		// 返回内点数
	    return features.size() - cnt_outlier;
	}

	int Frontend::TrackLastFrame() {
	    // use LK flow to estimate points in the right image
		// 使用LK光流估计点的位置
	    std::vector<cv::Point2f> kps_last, kps_current;
		// 遍历前一帧左目图像全部关键点
	    for (auto &kp : last_frame_->features_left_)
		{
			// 判断弱指针指向的对象是否有效
			// 关键点对应的地图点
			// 若有效根据地图点计算关键点像素坐标投影，否则直接记录关键点信息
	        if (kp->map_point_.lock())
			{
	            // use project point
	            auto mp = kp->map_point_.lock();
				// 空间点转换为像素坐标
	            auto px =
	                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
	            kps_last.push_back(kp->position_.pt);
	            kps_current.push_back(cv::Point2f(px[0], px[1]));
	        }
			else
			{
	            kps_last.push_back(kp->position_.pt);
	            kps_current.push_back(kp->position_.pt);
	        }
	    }

		// 储存特征点光流跟踪的状态
	    std::vector<uchar> status;
		// 储存特征点光流跟踪的误差
	    Mat error;
		// maxLevel(3) -> 4层图像金字塔 0 -> 不使用图像金字塔
		// criteria 迭代搜索算法的终止条件 maxCount-最大迭代次数 或 epsilon-收敛误差
	    cv::calcOpticalFlowPyrLK(
	        last_frame_->left_img_, current_frame_->left_img_, kps_last,
	        kps_current, status, error, cv::Size(11, 11), 3,
	        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
	                         0.01),
	        cv::OPTFLOW_USE_INITIAL_FLOW);

	    int num_good_pts = 0;
		// 遍历全部特征点
	    for (size_t i = 0; i < status.size(); ++i) {
			// 为跟踪状态
	        if (status[i]) {
				// 创建新的关键点对象 特征点像素为中心 7为半径
	            cv::KeyPoint kp(kps_current[i], 7);
				// new feature对象
	            Feature::Ptr feature(new Feature(current_frame_, kp));
				// 将前一帧的地图点信息赋值给feature
	            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
				// 将新的feature加入当前帧的特征点列表中
	            current_frame_->features_left_.push_back(feature);
	            num_good_pts++;
	        }
	    }

	    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
	    return num_good_pts;
	}

	bool Frontend::StereoInit() {
	    int num_features_left = DetectFeatures();
	    int num_coor_features = FindFeaturesInRight();
	    if (num_coor_features < num_features_init_) {
	        return false;
	    }

	    bool build_map_success = BuildInitMap();
	    if (build_map_success) {
	        status_ = FrontendStatus::TRACKING_GOOD;
	        if (viewer_) {
	            viewer_->AddCurrentFrame(current_frame_);
	            viewer_->UpdateMap();
	        }
	        return true;
	    }
	    return false;
	}

	int Frontend::DetectFeatures() {
	    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
	    for (auto &feat : current_frame_->features_left_)
		{
		/*	cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
		                  feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);*/
		// for opencv4
	        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
	                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
	    }

	    std::vector<cv::KeyPoint> keypoints;
	    gftt_->detect(current_frame_->left_img_, keypoints, mask);
	    int cnt_detected = 0;
	    for (auto &kp : keypoints)
		{
	        current_frame_->features_left_.push_back(
	            Feature::Ptr(new Feature(current_frame_, kp)));
	        cnt_detected++;
	    }

	    LOG(INFO) << "Detect " << cnt_detected << " new features";
	    return cnt_detected;
	}

	int Frontend::FindFeaturesInRight() {
	    // use LK flow to estimate points in the right image
	    std::vector<cv::Point2f> kps_left, kps_right;
	    for (auto &kp : current_frame_->features_left_)
		{
	        kps_left.push_back(kp->position_.pt);
	        auto mp = kp->map_point_.lock();
	        if (mp)
			{
	            // use projected points as initial guess
	            auto px =
	                camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
	            kps_right.push_back(cv::Point2f(px[0], px[1]));
	        }
			else
			{
	            // use same pixel in left iamge
	            kps_right.push_back(kp->position_.pt);
	        }
	    }

	    std::vector<uchar> status;
	    Mat error;
	    cv::calcOpticalFlowPyrLK(
	        current_frame_->left_img_, current_frame_->right_img_, kps_left,
	        kps_right, status, error, cv::Size(11, 11), 3,
	        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
	                         0.01),
	        cv::OPTFLOW_USE_INITIAL_FLOW);

	    int num_good_pts = 0;
	    for (size_t i = 0; i < status.size(); ++i) {
	        if (status[i])
			{
	            cv::KeyPoint kp(kps_right[i], 7);
	            Feature::Ptr feat(new Feature(current_frame_, kp));
	            feat->is_on_left_image_ = false;
	            current_frame_->features_right_.push_back(feat);
	            num_good_pts++;
	        }
			else
			{
	            current_frame_->features_right_.push_back(nullptr);
	        }
	    }
	    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
	    return num_good_pts;
	}

	bool Frontend::BuildInitMap() {
	    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
	    size_t cnt_init_landmarks = 0;
	    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
	        if (current_frame_->features_right_[i] == nullptr) continue;
	        // create map point from triangulation
	        std::vector<Vec3> points{
	            camera_left_->pixel2camera(
	                Vec2(current_frame_->features_left_[i]->position_.pt.x,
	                     current_frame_->features_left_[i]->position_.pt.y)),
	            camera_right_->pixel2camera(
	                Vec2(current_frame_->features_right_[i]->position_.pt.x,
	                     current_frame_->features_right_[i]->position_.pt.y))};
	        Vec3 pworld = Vec3::Zero();

	        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
	            auto new_map_point = MapPoint::CreateNewMappoint();
	            new_map_point->SetPos(pworld);
	            new_map_point->AddObservation(current_frame_->features_left_[i]);
	            new_map_point->AddObservation(current_frame_->features_right_[i]);
	            current_frame_->features_left_[i]->map_point_ = new_map_point;
	            current_frame_->features_right_[i]->map_point_ = new_map_point;
	            cnt_init_landmarks++;
	            map_->InsertMapPoint(new_map_point);
	        }
	    }
	    current_frame_->SetKeyFrame();
	    map_->InsertKeyFrame(current_frame_);
	    backend_->UpdateMap();

	    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
	              << " map points";

	    return true;
	}

	bool Frontend::Reset() {
	    LOG(INFO) << "Reset is not implemented. ";
	    return true;
	}

}  // namespace myslam