//
// Created by zmxj on 23-3-15.
//
#include<iostream>
#include<ctime>
#include <cmath>
#include <complex>
/*                线性方程组Ax = b的解法 ( 直接法（1,2,3,4,5）+迭代法(6) )  其中只有2 3方法不要求方程组个数与变量个数相等   */

//包含Eigen头文件
//#include <Eigen/Dense>
#include<Eigen/Core>
#include<Eigen/Geometry>
#include <Eigen/Eigenvalues>

//下面这两个宏的数值一样的时候 方法1 4 5 6才能正常工作
#define MATRIX_SIZE 3   //方程组个数
#define MATRIX_SIZE_ 3  //变量个数
//using namespace std;
typedef  Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE_>  Mat_A;
typedef  Eigen::Matrix<double, MATRIX_SIZE, 1>              Mat_B;

//Jacobi迭代法的一步求和计算
double Jacobi_sum(Mat_A   &A,Mat_B   &x_k,int i);

//迭代不收敛的话 解向量是0
Mat_B Jacobi(Mat_A   &A,Mat_B   &b,  int &iteration_num, double &accuracy );

int main(int argc,char **argv)
{
	//设置输出小数点后3位
	std::cout.precision(3);
	//设置变量
	Eigen::Matrix<double,MATRIX_SIZE, MATRIX_SIZE_> matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE_);
	Eigen::Matrix<double ,MATRIX_SIZE,1 > v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE,1);

	//测试用例
	matrix_NN << 10,3,1,2,-10,3,1,3,10;
	v_Nd <<14,-5,14;

	//设置解变量
	Eigen::Matrix<double,MATRIX_SIZE_,1>x;

	//时间变量
	clock_t tim_stt = clock();

/*1、求逆法      很可能没有解 仅仅针对方阵才能计算*/
#if (MATRIX_SIZE == MATRIX_SIZE_)
	x = matrix_NN.inverse() * v_Nd;
	std::cout<<"直接法所用时间和解为："<< 1000*(clock() - tim_stt)/(double)CLOCKS_PER_SEC
	         <<"MS"<< std::endl << x.transpose() << std::endl;
#else
	std::cout<<"直接法不能解!(提示:直接法中方程组的个数必须与变量个数相同，需要设置MATRIX_SIZE == MATRIX_SIZE_)"<<std::endl;
#endif

/*2、QR分解解方程组 适合非方阵和方阵 当方程组有解时的出的是真解，若方程组无解得出的是近似解*/
	tim_stt = clock();
	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
	std::cout<<"QR分解所用时间和解为："<<1000*(clock() - tim_stt)/(double)CLOCKS_PER_SEC
	         << "MS" << std::endl << x.transpose() << std::endl;

/*3、最小二乘法 适合非方阵和方阵，方程组有解时得出真解，否则是最小二乘解(在求解过程中可以用QR分解 分解最小二成的系数矩阵) */
	tim_stt = clock();
	x = (matrix_NN.transpose() * matrix_NN ).inverse() * (matrix_NN.transpose() * v_Nd);
	std::cout<<"最小二乘法所用时间和解为:"<< 1000*(clock() - tim_stt)/(double)CLOCKS_PER_SEC
	         << "MS" << std::endl  << x.transpose() << std::endl;

/*4、LU分解方法    只能为方阵（满足分解的条件才行）    */
#if (MATRIX_SIZE == MATRIX_SIZE_)
	tim_stt = clock();
	x = matrix_NN.lu().solve(v_Nd);
	std::cout<< "LU分解方法所用时间和解为:" << 1000*(clock() - tim_stt)/(double)CLOCKS_PER_SEC
	         << "MS" << std::endl << x.transpose() << std::endl;
#else
	std::cout<<"LU分解法不能解!(提示:直接法中方程组的个数必须与变量个数相同，需要设置MATRIX_SIZE == MATRIX_SIZE_)"<<std::endl;
#endif

/*5、Cholesky 分解方法  只能为方阵 (结果与其他的方法差好多)*/
#if (MATRIX_SIZE == MATRIX_SIZE_)
	tim_stt = clock();
	x = matrix_NN.llt().solve(v_Nd);
	std::cout<< "Cholesky 分解方法所用时间和解为:" << 1000*(clock() - tim_stt)/(double)CLOCKS_PER_SEC
	         << "MS"<< std::endl<< x.transpose()<<std::endl;
#else
	std::cout<< "Cholesky法不能解!(提示:直接法中方程组的个数必须与变量个数相同，需要设置MATRIX_SIZE == MATRIX_SIZE_)"<<std::endl;
#endif

/*6、Jacobi迭代法   */
#if (MATRIX_SIZE == MATRIX_SIZE_)
	int Iteration_num = 10 ;
	double Accuracy =0.01;
	tim_stt = clock();
	x= Jacobi(matrix_NN,v_Nd,Iteration_num,Accuracy);
	std::cout<< "Jacobi 迭代法所用时间和解为:" << 1000*(clock() - tim_stt)/(double)CLOCKS_PER_SEC
	         << "MS"<< std::endl<< x.transpose()<<std::endl;
#else
	std::cout<<"LU分解法不能解!(提示:直接法中方程组的个数必须与变量个数相同，需要设置MATRIX_SIZE == MATRIX_SIZE_)"<<std::endl;
#endif

	return 0;
}

//迭代不收敛的话 解向量是0
Mat_B Jacobi(Mat_A  &A,Mat_B  &b, int &iteration_num, double &accuracy )
{
	Mat_B x_k = Eigen::MatrixXd::Zero(MATRIX_SIZE_,1);//迭代的初始值
	Mat_B x_k1;         //迭代一次的解向量
	int k,i;            //i,k是迭代算法的循环次数的临时变量
	double temp;        //每迭代一次解向量的每一维变化的模值
	double R=0;         //迭代一次后，解向量每一维变化的模的最大值
	int isFlag = 0;     //迭代要求的次数后，是否满足精度要求

	//判断Jacobi是否收敛
	Mat_A D;            //D矩阵
	Mat_A L_U;          //L+U
	Mat_A temp2 = A;    //临时矩阵获得A矩阵除去对角线后的矩阵
	Mat_A B;            //Jacobi算法的迭代矩阵
	Eigen::MatrixXcd EV;//获取矩阵特征值
	double maxev=0.0;   //最大模的特征值
	int flag = 0;       //判断迭代算法是否收敛的标志 1表示Jacobi算法不一定能收敛到真值

	std::cout<<std::endl<<"欢迎进入Jacobi迭代算法！"<<std::endl;
	//對A矩陣進行分解 求取迭代矩陣 再次求取譜半徑 判斷Jacobi迭代算法是否收斂
	for(int l=0 ;l < MATRIX_SIZE;l++)
	{
		D(l,l) = A(l,l);
		temp2(l,l) = 0;
		if(D(l,l) == 0)
		{
			std::cout<<"迭代矩阵不可求"<<std::endl;
			flag =1;
			break;
		}
	}
	L_U = -temp2;
	B = D.inverse()*L_U;

	//求取特征值
	Eigen::EigenSolver<Mat_A>es(B);
	EV = es.eigenvalues();
//    cout<<"迭代矩阵特征值为:"<<EV << endl;

	//求取矩陣的特征值 然後獲取模最大的特徵值 即爲譜半徑
	for(int index = 0;index< MATRIX_SIZE;index++)
	{
		maxev = ( maxev > __complex_abs(EV(index)) )?maxev:(__complex_abs(EV(index)));
	}
	std::cout<< "Jacobi迭代矩阵的谱半径为："<< maxev<<std::endl;

	//谱半径大于1 迭代法则发散
	if(maxev >= 1)
	{
		std::cout<<"Jacobi迭代算法不收敛！"<<std::endl;
		flag =1;
	}

	//迭代法收敛则进行迭代的计算
	if (flag == 0 )
	{
		std::cout<<"Jacobi迭代算法谱半径小于1,该算法收敛"<<std::endl;
		std::cout<<"Jacobi迭代法迭代次数和精度： "<< std::endl << iteration_num<<" "<<accuracy<<std::endl;

		//迭代计算
		for( k = 0 ;k < iteration_num ; k++ )
		{
			for(i = 0;i< MATRIX_SIZE_ ; i++)
			{
				x_k1(i) = x_k(i) + ( b(i) - Jacobi_sum(A,x_k,i) )/A(i,i);
				temp = fabs( x_k1(i) - x_k(i) );
				if( fabs( x_k1(i) - x_k(i) ) > R )
					R = temp;
			}

			//判断进度是否达到精度要求 达到进度要求后 自动退出
			if( R < accuracy )
			{
				std::cout <<"Jacobi迭代算法迭代"<< k << "次达到精度要求."<< std::endl;
				isFlag = 1;
				break;
			}

			//清零R，交换迭代解
			R = 0;
			x_k = x_k1;
		}
		if( !isFlag )
			std::cout << std::endl <<"迭代"<<iteration_num<<"次后仍然未达到精度要求，若不满意该解，请再次运行加大循环次数！"<< std::endl;
		return x_k1;
	}
	//否则返回0
	return  x_k;
}

//Jacobi迭代法的一步求和计算
double Jacobi_sum(Mat_A  &A,Mat_B &x_k,int i)
{
	double sum;
	for(int j = 0; j< MATRIX_SIZE_;j++)
	{
		sum += A(i,j)*x_k(j);
	}
	return sum;
}