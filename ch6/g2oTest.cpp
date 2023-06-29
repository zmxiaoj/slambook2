//
// Created by zmxj on 23-3-21.
//
#include <iostream>
#include <g2o/core/base_vertex.h>//顶点类型
#include <g2o/core/base_unary_edge.h>//一元边类型
#include <g2o/core/block_solver.h>//求解器的实现,主要来自choldmod, csparse
#include <g2o/core/optimization_algorithm_levenberg.h>//列文伯格－马夸尔特
#include <g2o/core/optimization_algorithm_gauss_newton.h>//高斯牛顿法
#include <g2o/core/optimization_algorithm_dogleg.h>//Dogleg（狗腿方法）
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>//矩阵库
#include <opencv2/core/core.hpp>//opencv2
#include <cmath>//数学库
#include <chrono>//时间库

using namespace std;

class CurveFittingVertex: public g2o::BaseVertex<4, Eigen::Vector4d>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	virtual void setToOriginImpl()
	{
		_estimate << 0, 0, 0, 0;
	}

	virtual void oplusImpl(const double* update)
	{
		_estimate += Eigen::Vector4d(update);
	}
	virtual bool read(istream &in) {}
	virtual bool write(ostream &out) const {}

};

class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	double _x;

	CurveFittingEdge(double x): BaseUnaryEdge(), _x(x) {}

	void computeError()
	{
		const CurveFittingVertex *v = static_cast<const CurveFittingVertex*>(_vertices[0]);

		const Eigen::Vector4d abcd = v->estimate();

		_error(0, 0) = _measurement - exp(abcd(0, 0) * pow(_x, 3)
				+ abcd(1, 0) * pow(_x, 2) + abcd(2, 0) * _x + abcd(3, 0));
	}
	virtual bool read(istream &in) {}
	virtual bool write(ostream &out) const {}
};

int main()
{
	double a=3.5, b=1.6, c=0.3, d=7.8;
	int N=100;
	double w_sigma=1.0;
	cv::RNG rng;

	vector<double> x_data, y_data;

	cout<<"generating data: "<<endl;
	for (int i=0; i<N; i++)
	{
		double x = i/100.0;
		x_data.push_back (x);
		y_data.push_back (exp(a*x*x*x + b*x*x + c*x + d) + rng.gaussian (w_sigma));
		cout<<x_data[i]<<"\t"<<y_data[i]<<endl;
	}
	typedef g2o::BlockSolver<g2o::BlockSolverTraits<4, 1>> Block;

	unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverDense<Block::PoseMatrixType>());
	unique_ptr<Block> solver_ptr (new Block(move(linearSolver)));

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(move(solver_ptr));
	//g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( std::move(solver_ptr));
	//g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg(std::move(solver_ptr));
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(true);
	CurveFittingVertex* v = new CurveFittingVertex();
	v->setEstimate(Eigen::Vector4d(0, 0, 0, 0));
	v->setId(0);
	optimizer.addVertex(v);

	for (int i=0; i<N; i++)
	{
		//新建边带入观测数据
		CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
		edge->setId(i);
		//设置连接的顶点，注意使用方式
		//这里第一个参数表示边连接的第几个节点(从0开始)，第二个参数是该节点的指针
		edge->setVertex(0, v);
		//观测数值
		edge->setMeasurement(y_data[i]);
		//信息矩阵：协方差矩阵之逆，这里各边权重相同。这里Eigen的Matrix其实也是模板类
		edge->setInformation(Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma));
		optimizer.addEdge(edge);
	}
	//执行优化
	cout<<"start optimization"<<endl;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时

	//初始化优化器
	optimizer.initializeOptimization();
	//优化次数
	optimizer.optimize(100);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//结束计时
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
	cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
	//输出优化值
	Eigen::Vector4d abc_estimate = v->estimate();
	cout<<"estimated model: "<<abc_estimate.transpose()<<endl;

	return 0;
}