#ifndef CONTROLLER_H
#define CONTROLLER_H
#include <opencv.hpp>
#include <iostream>
#include <fstream>
#include <qdebug.h>
#include <exception>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;
using namespace std;

#define MAX_IMAGE_NUM (66)
#define IMAGE_TYPE_NUM (4)
#define PIXEL_COUNT (32*32)
#define IMAGE_SIZE (256*256)
#define FRACTAL_SIZE (8)
#define NAME_LEN (128)
#define GrayScale (255)
#define GLCM_DIS 3  //灰度共生矩阵的统计距离
#define GLCM_CLASS 16 //计算灰度共生矩阵的图像灰度值等级化
#define GLCM_ANGLE_HORIZATION 0  //水平
#define GLCM_ANGLE_VERTICAL   1	 //垂直
#define GLCM_ANGLE_DIGONAL    2  //对角

class Controller
{
public:
	Controller();
	~Controller();
	void test();

private:
	enum type {FLAIR,T1,T1C,T2};
	
	//图像预处理
	void preprocess();
	//提取颜色特征
	void extractColor(int);
	void extractColorFractal(int);
	//提取梯度特征
	void extractGradient(int);
	//提取分形特征 
	void extractFractal(int);
	//提取纹理特征（灰度共生矩阵，Gabor小波）
	void extractTexture(int);
	//提取边缘特征然后计算距离？
	void extractEdge(int);
	//提取傅里叶频谱特征
	void extractSpectrum(int);
	//归一化函数
	void normmat(Mat);
	//计算灰度共生矩阵
	double calGLCM(int,int,int,int,int);
	//机器学习
	Mat generateTestcase(int, int);
	void generateFractalResults(int);
	void generateResults(int);
	//help function
	string type2string(int);

	IplImage *src[MAX_IMAGE_NUM][IMAGE_TYPE_NUM*PIXEL_COUNT];
	Mat colors;
	Mat gradients;
	Mat spectrums;
	Mat textures;
	Mat fractals;
	Mat results;
	Mat fractalRet;

	//for debug
	ofstream ofs;
};
#endif//CONTROLLER_H