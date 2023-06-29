//
// Created by zmxj on 23-3-16.
//

#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core.hpp>

using namespace std;

struct myStruct
{
public:
	double x_, y_;

	myStruct(double x, double y)
	{
		x_ = x;
		y_ = y;
	}

	double operator+(double z)
	{
		return x_ * 2 + y_ * 2 + z * 2;
	}

	template<typename T>
	bool operator()(T &x, T &y) const
	{
		x += x_;
		y += y_;
		return true;
	}


};

template <typename T>
T add(T a, T b)
{
	T sum = a + b;
	return sum;
}

struct CostFunctor
{
	template<typename T>
	bool operator()(const T* const x, T* residual)const
	{
		residual[0] = 10.0 - x[0];
		return true;
	}
};

struct CostFunctor1
{
	double x_ob, y_ob;

	CostFunctor1(double x, double y)
	{
		x_ob = x;
		y_ob = y;
	}

	template<typename T>
	bool operator()(const T* const params, T* residual)const
	{
		residual[0] = y_ob - (params[0] * pow(x_ob, 3) + params[1] * pow(x_ob, 2) + params[2] * x_ob + params[3]);
		return true;
	}
};

vector<double> xs, ys;

void generateDate()
{
	cv::RNG rng;
	double w_sigma = 5.0;
	for (int i = 0; i < 100; i++)
	{
		double x = i;
		double y = 3.5 * pow(x, 3.0) + 1.6 * pow(x, 2.0) + 0.3 * x + 7.8;
		xs.push_back(x);
		ys.push_back(y + rng.gaussian(w_sigma));
	}
	for (int i = 0; i < xs.size(); i++)
	{
		cout << "x: " << xs[i] << " y: " << ys[i] << endl;
	}
}

int main(int argc, char** argv)
{
	myStruct* s1;
	s1 = new myStruct(1.0, 2.1);

	cout << s1->x_ << endl;
	cout << s1->y_ << endl;

	double sum1 = add<double>(2.3, 3.1);
	cout << "double " << sum1 << endl;
	int sum2 = add<int>(1, 7);
	cout << "int " << sum2 << endl;

	double res1 = s1->operator+(2);
	cout << "res1: " << res1 << endl;
	double res2 = *s1 + 3;
	cout << "res2: " << res2 << endl;

	myStruct* s2 = new myStruct(2, 4);
	double x = 1.0, y = 3.0;
	s2->operator()<double>(x, y);
	cout << "x: " << x << "y: " << y << endl;

	double initial_v = 5.0;
	double v = initial_v;
	ceres::Problem problem;

	ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
	problem.AddResidualBlock(cost_function, nullptr, &v);

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	cout << summary.FullReport() << endl;
	cout << "initial_v: " << initial_v << "-> " << "v: " << v << endl;

	cout << endl << endl;

	generateDate();

	double params[4] = {1.0};

	ceres::Problem problem1;
	for (int i = 0; i < xs.size(); i++)
	{
		ceres::CostFunction *cost_function1 = new
				ceres::AutoDiffCostFunction<CostFunctor1, 1, 4>(new CostFunctor1(xs[i], ys[i]));
		problem1.AddResidualBlock(cost_function1, NULL, params);
	}
	ceres::Solver::Options options1;
	options1.linear_solver_type = ceres::DENSE_QR;
	options1.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary1;

	ceres::Solve(options1, &problem1, &summary1);

	cout << summary1.FullReport() << endl;

	cout << "p0: " << params[0] << endl;
	cout << "p1: " << params[1] << endl;
	cout << "p2: " << params[2] << endl;
	cout << "p3: " << params[3] << endl;

	return 0;
}