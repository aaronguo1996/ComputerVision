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
#define GLCM_DIS 3  //�Ҷȹ��������ͳ�ƾ���
#define GLCM_CLASS 16 //����Ҷȹ��������ͼ��Ҷ�ֵ�ȼ���
#define GLCM_ANGLE_HORIZATION 0  //ˮƽ
#define GLCM_ANGLE_VERTICAL   1	 //��ֱ
#define GLCM_ANGLE_DIGONAL    2  //�Խ�

class Controller
{
public:
	Controller();
	~Controller();
	void test();

private:
	enum type {FLAIR,T1,T1C,T2};
	
	//ͼ��Ԥ����
	void preprocess();
	//��ȡ��ɫ����
	void extractColor(int);
	void extractColorFractal(int);
	//��ȡ�ݶ�����
	void extractGradient(int);
	//��ȡ�������� 
	void extractFractal(int);
	//��ȡ�����������Ҷȹ�������GaborС����
	void extractTexture(int);
	//��ȡ��Ե����Ȼ�������룿
	void extractEdge(int);
	//��ȡ����ҶƵ������
	void extractSpectrum(int);
	//��һ������
	void normmat(Mat);
	//����Ҷȹ�������
	double calGLCM(int,int,int,int,int);
	//����ѧϰ
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