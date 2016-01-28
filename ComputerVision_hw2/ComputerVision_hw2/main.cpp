#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
using namespace cv;

#define GrayScale (255)
#define LN5 (0.6931)
#define pi (3.1415)
#define BLOCK 8
#define cvQueryHistValue_1D( hist, idx0 ) ((float)cvGetReal1D( (hist)->bins, (idx0)))

int otsuThreshold(IplImage *frame)
{
	int width = frame->width;
	int height = frame->height;
	int i, j, pixelSum = width * height, threshold = 0;
	int pixelCount[GrayScale];
	float pixelPro[GrayScale];
	uchar* data = (uchar*)frame->imageData;

	for (i = 0; i < GrayScale; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			pixelCount[(int)data[i * width + j]]++;
		}
	}

	//����ÿ������������ͼ���еı���
	for (i = 0; i < GrayScale; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//�����Ҷȼ�[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u,
		deltaTmp, deltaMax = 0;
	for (i = 0; i < GrayScale; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
		for (j = 0; j < GrayScale; j++)
		{
			if (j <= i)   //��������
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //ǰ������
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		//cout << "u0: " << u0 << endl << "u1: " << u1 << endl;
		u = u0tmp + u1tmp;
		deltaTmp =
			w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
		if (deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	return threshold;
}

int fuzzyCompactnessThreshold(IplImage *frame)
{
	int width = frame->width;
	int height = frame->height;
	int pixelCount[GrayScale];
	float pixelPro[GrayScale];
	float *pixel = new float[width*height];
	int i, j, graymin = 0,graymax = 0,pixelSum = width * height, threshold = 0;
	uchar* data = (uchar*)frame->imageData;
	graymin = graymax = (int)data[0];

	for (i = 0; i < GrayScale; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (graymin > (int)data[i * width + j]) graymin = (int)data[i * width + j];
			if (graymax < (int)data[i * width + j]) graymax = (int)data[i * width + j];
			pixelCount[(int)data[i * width + j]]++;
		}
	}

	//����ÿ������������ͼ���еı���
	for (i = 0; i < GrayScale; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//�����Ҷȼ�[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u,au,pu,c,
		deltaTmp, deltaMax = 0;
	c = 1.0 / (graymax - graymin);
	for (i = 0; i < GrayScale; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = au = pu = 0;
		for (j = 0; j < GrayScale; j++)
		{
			if (j <= i)   //��������
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //ǰ������
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		//cout << "u0: " << u0 << endl << "u1: " << u1 << endl;
		//return 255;
		for (int m = 0; m < width; m++)
		{
			for (int n = 0; n < height; n++)
			{
				if ((int)data[m*height + n] > i)
				{
					pixel[m*height + n] = exp(-c*abs(data[m * height + n] - u1));
				}
				else
				{
					pixel[m*height + n] = exp(-c*abs(data[m * height + n] - u0));
				}
				au += pixel[m*height + n];
				//cout << "pixel: " << pixel[m*height + n]<<endl;
			}
		}
		//cout << "au: " << au <<endl;
		for (int m = 0; m < width; m++)
		{
			for (int n = 0; n < height - 1; n++)
			{
				pu += abs((int)data[m * height + n] - (int)data[m * height + n + 1]);
			}
		}
		for (int n = 0; n < height; n++)
		{
			for (int m = 0; m < width - 1; m++)
			{
				pu += abs((int)data[m * height + n] - (int)data[(m + 1) * height + n]);
			}
		}
		//cout << pu << endl;
		deltaTmp = au / (pu*pu);
		//cout << au << endl << pu << endl << deltaTmp << endl;
		if (deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	//cout << c << endl;
	return threshold;
}

int linearIndicesOfFuzzinessThreshold(IplImage *frame)
{
	int width = frame->width;
	int height = frame->height;
	int pixelCount[GrayScale];
	float pixelPro[GrayScale];
	float *pixel = new float[width*height];
	int i, j, graymin = 0, graymax = 0, pixelSum = width * height, threshold = 0;
	uchar* data = (uchar*)frame->imageData;
	graymin = graymax = (int)data[0];

	for (i = 0; i < GrayScale; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (graymin >(int)data[i * width + j]) graymin = (int)data[i * width + j];
			if (graymax < (int)data[i * width + j]) graymax = (int)data[i * width + j];
			pixelCount[(int)data[i * width + j]]++;
		}
	}

	//����ÿ������������ͼ���еı���
	for (i = 0; i < GrayScale; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//�����Ҷȼ�[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u, li, c,
		deltaTmp, deltaMin;
	c = 1.0 / (graymax - graymin);
	for (i = 0; i < GrayScale; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = li = 0;
		for (j = 0; j < GrayScale; j++)
		{
			if (j <= i)   //��������
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //ǰ������
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		//cout << "u0: " << u0 << endl << "u1: " << u1 << endl;
		//return 255;
		for (int m = 0; m < width; m++)
		{
			for (int n = 0; n < height; n++)
			{
				if ((int)data[m*height + n] > i)
				{
					pixel[m*height + n] = exp(-c*LN5*abs(data[m * height + n] - u1));
				}
				else
				{
					pixel[m*height + n] = exp(-c*LN5*abs(data[m * height + n] - u0));
				}
				if (pixel[m*height + n] <= 0.5)
				{
					li += abs(pixel[m*height + n]);
				}
				else
				{
					li += abs(pixel[m*height + n] - 1);
				}
				//cout << "pixel: " << pixel[m*height + n]<<endl;
				
			}
		}
		//cout << "li: " << li << endl;
		//cout << pu << endl;
		deltaTmp = li * 2 / (width * height);
		if (i == 0) deltaMin = deltaTmp;
		//cout << deltaTmp << endl;
		if (deltaTmp < deltaMin)
		{
			deltaMin = deltaTmp;
			threshold = i;
			//cout << "threshold: "<<threshold << endl;
		}
	}
	//cout << c << endl;
	return threshold;
}

int quadraticIndicesOfFuzzinessThreshold(IplImage *frame)
{
	int width = frame->width;
	int height = frame->height;
	int pixelCount[GrayScale];
	float pixelPro[GrayScale];
	float *pixel = new float[width*height];
	int i, j, graymin = 0, graymax = 0, pixelSum = width * height, threshold = 0;
	uchar* data = (uchar*)frame->imageData;
	graymin = graymax = (int)data[0];

	for (i = 0; i < GrayScale; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (graymin >(int)data[i * width + j]) graymin = (int)data[i * width + j];
			if (graymax < (int)data[i * width + j]) graymax = (int)data[i * width + j];
			pixelCount[(int)data[i * width + j]]++;
		}
	}

	//����ÿ������������ͼ���еı���
	for (i = 0; i < GrayScale; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//�����Ҷȼ�[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u, qi, c,
		deltaTmp, deltaMin;
	c = 1.0 / (graymax - graymin);
	for (i = 0; i < GrayScale; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = qi = 0;
		for (j = 0; j < GrayScale; j++)
		{
			if (j <= i)   //��������
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //ǰ������
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		//cout << "u0: " << u0 << endl << "u1: " << u1 << endl;
		//return 255;
		for (int m = 0; m < width; m++)
		{
			for (int n = 0; n < height; n++)
			{
				if ((int)data[m*height + n] > i)
				{
					pixel[m*height + n] = exp(-c*LN5*abs(data[m * height + n] - u1));
				}
				else
				{
					pixel[m*height + n] = exp(-c*LN5*abs(data[m * height + n] - u0));
				}
				if (pixel[m*height + n] <= 0.5)
				{
					qi += pow(pixel[m*height + n],2);
				}
				else
				{
					qi += pow((pixel[m*height + n] - 1),2);
				}
				//cout << "pixel: " << pixel[m*height + n]<<endl;

			}
		}
		//cout << "li: " << li << endl;
		//cout << pu << endl;
		deltaTmp = sqrt(qi) * 2 / sqrt(width * height);
		if (i == 0) deltaMin = deltaTmp;
		//cout << deltaTmp << endl;
		if (deltaTmp < deltaMin)
		{
			deltaMin = deltaTmp;
			threshold = i;
			//cout << "threshold: "<<threshold << endl;
		}
	}
	//cout << c << endl;
	return threshold;
}

int fuzzySimilarityThreshold(IplImage *frame)
{
	int width = frame->width;
	int height = frame->height;
	int pixelCount[GrayScale];
	float pixelPro[GrayScale];
	float *pixel = new float[width*height];
	int i, j, graymin = 0, graymax = 0, pixelSum = width * height, threshold = 0;
	uchar* data = (uchar*)frame->imageData;
	graymin = graymax = (int)data[0];

	for (i = 0; i < GrayScale; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (graymin >(int)data[i * width + j]) graymin = (int)data[i * width + j];
			if (graymax < (int)data[i * width + j]) graymax = (int)data[i * width + j];
			pixelCount[(int)data[i * width + j]]++;
		}
	}

	//����ÿ������������ͼ���еı���
	for (i = 0; i < GrayScale; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//�����Ҷȼ�[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u, s, c,
		deltaTmp, deltaMax = 0;
	c = 1.0 / (graymax - graymin);
	for (i = 0; i < GrayScale; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = s = 0;
		for (j = 0; j < GrayScale; j++)
		{
			if (j <= i)   //��������
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //ǰ������
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		//cout << "u0: " << u0 << endl << "u1: " << u1 << endl;
		//return 255;
		for (int m = 0; m < width; m++)
		{
			for (int n = 0; n < height; n++)
			{
				if ((int)data[m*height + n] > i)
				{
					pixel[m*height + n] = exp(-c*abs(data[m * height + n] - u1));
				}
				else
				{
					pixel[m*height + n] = exp(-c*abs(data[m * height + n] - u0));
				}
				s += pixel[m*height + n];
				//cout << "pixel: " << pixel[m*height + n]<<endl;

			}
		}
		//cout << "li: " << li << endl;
		//cout << pu << endl;
		deltaTmp = s / (width * height);
		if (i == 0) deltaMax = deltaTmp;
		//cout << deltaTmp << endl;
		if (deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
			//cout << "threshold: "<<threshold << endl;
		}
	}
	//cout << c << endl;
	return threshold;

}

int fuzzyDivergenceThreshold(IplImage *frame)
{
	int width = frame->width;
	int height = frame->height;
	int pixelCount[GrayScale];
	float pixelPro[GrayScale];
	float *pixel = new float[width*height];
	int i, j, graymin = 0, graymax = 0, pixelSum = width * height, threshold = 0;
	uchar* data = (uchar*)frame->imageData;
	graymin = graymax = (int)data[0];

	for (i = 0; i < GrayScale; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (graymin >(int)data[i * width + j]) graymin = (int)data[i * width + j];
			if (graymax < (int)data[i * width + j]) graymax = (int)data[i * width + j];
			pixelCount[(int)data[i * width + j]]++;
		}
	}

	//����ÿ������������ͼ���еı���
	for (i = 0; i < GrayScale; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//�����Ҷȼ�[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u, divergence, c,
		deltaTmp, deltaMin;
	c = 1.0 / (graymax - graymin);
	for (i = 0; i < GrayScale; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = divergence = 0;
		for (j = 0; j < GrayScale; j++)
		{
			if (j <= i)   //��������
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //ǰ������
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		//cout << "u0: " << u0 << endl << "u1: " << u1 << endl;
		//return 255;
		for (int m = 0; m < width; m++)
		{
			for (int n = 0; n < height; n++)
			{
				if ((int)data[m*height + n] > i)
				{
					pixel[m*height + n] = exp(-c*abs(data[m * height + n] - u1));
				}
				else
				{
					pixel[m*height + n] = exp(-c*abs(data[m * height + n] - u0));
				}
				divergence += (2 - (1 - pixel[m*height + n] + 1)*exp(pixel[m*height + n] - 1) - 
							  (1 - 1 + pixel[m*height + n])*exp(1 - pixel[m*height + n]));
			}
		}
		//cout << "li: " << li << endl;
		//cout << pu << endl;
		deltaTmp = divergence;// li * 2 / (width * height);
		if (i == 0) deltaMin = deltaTmp;
		//cout << deltaTmp << endl;
		if (deltaTmp < deltaMin)
		{
			deltaMin = deltaTmp;
			threshold = i;
			//cout << "threshold: "<<threshold << endl;
		}
	}
	//cout << c << endl;
	return threshold;
}

Mat equalizeHist(IplImage *src, IplImage *dst, int max_scale)
{
	Mat image = src;
	Mat after = image * 1;
	double *p = new double[max_scale];
	for (int i = 0; i < max_scale; i++)
		p[i] = 0;
	int max, min;
	max = min = image.at<uchar>(0, 0);
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			int tmp = image.at<uchar>(i, j);
			if (max < tmp)
				max = tmp;
			if (min > tmp)
				min = tmp;
			p[tmp]++;
		}
	}

	for (int i = 0; i < max_scale; i++){
		p[i] = p[i] / (image.rows*image.cols);
	}
	for (int i = 1; i < max_scale; i++){
		p[i] = p[i - 1] + p[i];
	}
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			after.at<uchar>(i, j) = p[image.at<uchar>(i, j)] * (max - min) + min;
		}
	}
	*dst = after;

	//delete[] p;
	return after;
}
/*
 * is_color: ѡ�����ģʽ����RGB����HSV
 */
void histgramEqualization(IplImage *src, IplImage *dst, int max_scale, int is_color)
{
	if (is_color==1)
	{
		IplImage* rImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
		IplImage* gImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
		IplImage* bImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
		cvSplit(src, bImg, gImg, rImg, 0);
		Mat a = equalizeHist(bImg, bImg, max_scale);
		Mat b = equalizeHist(gImg, gImg, max_scale);
		Mat c = equalizeHist(rImg, rImg, max_scale);
		cvMerge(bImg, gImg, rImg, 0, dst);
	}
	else if (is_color == 0)
	{
		IplImage* hImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
		IplImage* sImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
		IplImage* vImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
		IplImage* hsv = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);
		cvCvtColor(src, hsv, CV_RGB2HSV);
		cvSplit(hsv, hImg, sImg, vImg, 0);
		Mat a = equalizeHist(vImg, vImg, max_scale);
		cvMerge(hImg, sImg, vImg, 0, dst);
		cvCvtColor(dst, dst, CV_HSV2RGB);
	}
	cvNamedWindow("origin");
	cvShowImage("origin", src);
	cvNamedWindow("processed");
	cvShowImage("processed", dst);
	waitKey(0);
}

void traceEdge(int y, int x, int nThrLow, unsigned char *pResult, Mat pMag, int cx)
{
	//��8�������ؽ��в�ѯ
	int xNum[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int yNum[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
	int yy, xx, k;
	for (k = 0; k<8; k++)
	{
		yy = y + yNum[k];
		xx = x + xNum[k];
		if (pResult[yy*cx + xx] == 128 && pMag.at<uchar>(yy, xx) >= nThrLow)
		{
			//�õ���Ϊ�߽��
			pResult[yy*cx + xx] = 255;
			//�Ըõ�Ϊ�����ٽ��и���
			traceEdge(yy, xx, nThrLow, pResult, pMag, cx);
		}
	}
}

void canny(IplImage *src, IplImage *dst)
{
	int nWidth, nHeight, nWidthStep, i = 0, j = 0;
	nWidth = src->width;
	nHeight = src->height;

	//תΪ�Ҷ�ͼ��
	IplImage *grayImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	cvCvtColor(src, grayImg, CV_RGB2GRAY);
	//return;
	nWidthStep = grayImg->widthStep;
	cvNamedWindow("image");
	cvShowImage("image", grayImg);
	waitKey(0);

	//��˹ģ��
	Mat srcImg = grayImg;
	GaussianBlur(srcImg, srcImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//�����ݶ����ݶȷ���
	Mat grad_x, grad_y, grad, abs_grad, orient;
	Mat gray = srcImg;
	int ddepth = CV_32F;
	Sobel(gray, grad_x, ddepth, 1, 0, 3);
	Sobel(gray, grad_y, ddepth, 0, 1, 3);
	cartToPolar(grad_x, grad_y, grad, orient, true);
	convertScaleAbs(grad, abs_grad);
	IplImage gradImg = abs_grad;
	cvShowImage("image", &gradImg);
	waitKey(0);

	//�Ǽ���ֵ����
	unsigned char* N = new unsigned char[nWidth*nHeight];  //�Ǽ���ֵ���ƽ��  
	double dTmp1 = 0.0, dTmp2 = 0.0;                       //�������������ص�ĻҶ�����  
	double g1 = 0.0, g2 = 0.0, g3 = 0.0, g4 = 0.0;
	double dWeight = 0.0;
	//��Եֵ����Ϊ0
	for (i = 0; i < nWidth; i++)
	{
		N[i] = 0;
		N[(nHeight - 1)*nWidth + i] = 0;
	}
	for (j = 0; j < nHeight; j++)
	{
		N[j*nWidth] = 0;
		N[j*nWidth + (nWidth - 1)] = 0;
	}
	//�Ǳ�Ե�Ĵ���

	for (i = 1; i<(nWidth - 1); i++)
	{
		for (j = 1; j<(nHeight - 1); j++)
		{
			int nPointIdx = i + j*nWidth;       //��ǰ����ͼ�������е�����ֵ
			double theta = orient.at<float>(j, i);
			if (abs_grad.at<uchar>(j,i) == 0)
				N[nPointIdx] = 0;         //�����ǰ�ݶȷ�ֵΪ0�����Ǿֲ����Ըõ㸳Ϊ0
			else
			{
				////////�����ж��������������Ȼ����������ֵ/////// 
				////////////////////��һ�����///////////////////////
				/////////		g1	g2					/////////////
				/////////			C					/////////////
				/////////			g3  g4				/////////////
				/////////////////////////////////////////////////////
				if (((theta >= 90) && (theta<135)) ||
					((theta >= 270) && (theta<315)))
				{
					//////����б�ʺ��ĸ��м�ֵ���в�ֵ���
					g1 = abs_grad.at<uchar>(j-1, i-1);
					g2 = abs_grad.at<uchar>(j-1, i);
					g3 = abs_grad.at<uchar>(j+1, i);
					g4 = abs_grad.at<uchar>(j+1, i+1);
					dWeight = 1.0/tan(theta/180*pi);   //������
					dTmp1 = g1*dWeight + g2*(1 - dWeight);
					dTmp2 = g4*dWeight + g3*(1 - dWeight);
				}
				////////////////////�ڶ������///////////////////////
				/////////		g1						/////////////
				/////////		g2	C	g3				/////////////
				/////////			    g4				/////////////
				/////////////////////////////////////////////////////
				else if (((theta >= 135) && (theta<180)) ||
					((theta >= 315) && (theta<360)))
				{
					g1 = abs_grad.at<uchar>(j - 1, i - 1);
					g2 = abs_grad.at<uchar>(j, i - 1);
					g3 = abs_grad.at<uchar>(j, i + 1);
					g4 = abs_grad.at<uchar>(j + 1, i + 1);
					dWeight = tan(theta / 180 * pi);   //����
					dTmp1 = g2*dWeight + g1*(1 - dWeight);
					dTmp2 = g4*dWeight + g3*(1 - dWeight);
				}
				////////////////////���������///////////////////////
				/////////			g1	g2				/////////////
				/////////			C					/////////////
				/////////		g4	g3    				/////////////
				/////////////////////////////////////////////////////
				else if (((theta >= 45) && (theta<90)) ||
					((theta >= 225) && (theta<270)))
				{
					g1 = abs_grad.at<uchar>(j - 1, i);
					g2 = abs_grad.at<uchar>(j - 1, i + 1);
					g3 = abs_grad.at<uchar>(j + 1, i);
					g4 = abs_grad.at<uchar>(j + 1, i - 1);
					dWeight = 1.0/tan(theta / 180 * pi);   //������
					dTmp1 = g2*dWeight + g1*(1 - dWeight);
					dTmp2 = g3*dWeight + g4*(1 - dWeight);
				}
				////////////////////���������///////////////////////
				/////////				g1				/////////////
				/////////		g4	C	g2				/////////////
				/////////		g3	    				/////////////
				/////////////////////////////////////////////////////
				else if (((theta >= 0) && (theta<45)) ||
					((theta >= 180) && (theta<225)))
				{
					g1 = abs_grad.at<uchar>(j - 1, i + 1);
					g2 = abs_grad.at<uchar>(j, i + 1);
					g3 = abs_grad.at<uchar>(j + 1, i - 1);
					g4 = abs_grad.at<uchar>(j, i - 1);
					dWeight = tan(theta / 180 * pi);   //����
					dTmp1 = g1*dWeight + g2*(1 - dWeight);
					dTmp2 = g3*dWeight + g4*(1 - dWeight);
				}
			}
			//////////���оֲ����ֵ�жϣ���д������////////////////
			if ((abs_grad.at<uchar>(j, i) >= dTmp1) && (abs_grad.at<uchar>(j, i) >= dTmp2)){
				N[nPointIdx] = 128;
				dst->imageData[j*nWidthStep + i] = abs_grad.at<uchar>(j, i);
			}
			else{
				N[nPointIdx] = 0;
				dst->imageData[j*nWidthStep + i] = 0;
			}
		}
	}
	cvShowImage("image", dst);
	waitKey(0);

	//˫��ֵ���
	int nHist[1024];
	int nEdgeNum;             //���ܱ߽���
	int nMaxMag = 0;          //����ݶ���
	int nHighCount;

	for (i = 0; i<1024; i++)
		nHist[i] = 0;
	for (i = 0; i<nHeight; i++)
	{
		for (j = 0; j<nWidth; j++)
		{
			if (N[i*nWidth + j] == 128)
				nHist[(int)abs_grad.at<uchar>(i, j)]++;
		}
	}
	nEdgeNum = nHist[0];
	nMaxMag = 0;                    //��ȡ�����ݶ�ֵ		
	for (i = 1; i<1024; i++)           //ͳ�ƾ����������ֵ���ơ����ж�������
	{
		if (nHist[i] != 0)       //�ݶ�Ϊ0�ĵ��ǲ�����Ϊ�߽���
		{
			nMaxMag = i;
		}
		nEdgeNum += nHist[i];   //����non-maximum suppression���ж�������
	}

	double	dRatHigh = 0.7;
	double	dThrHigh;
	double	dThrLow;
	double	dRatLow = 0.5;
	nHighCount = (int)(dRatHigh * nEdgeNum + 0.5);
	j = 1;
	nEdgeNum = nHist[1];
	while ((j<(nMaxMag - 1)) && (nEdgeNum < nHighCount))
	{
		j++;
		nEdgeNum += nHist[j];
	}
	dThrHigh = j;                            //����ֵ
	dThrLow = (int)((dThrHigh)* dRatLow+0.5);    //����ֵ
	
	for (i = 0; i<nHeight; i++)
	{
		for (j = 0; j<nWidth; j++)
		{
			if ((N[i*nWidth + j] == 128) && (abs_grad.at<uchar>(i,j) >= dThrHigh))
			{
				N[i*nWidth + j] = 255;
				traceEdge(i, j, dThrLow, N, abs_grad, nWidth);
			}
		} 
	}

	//����û������Ϊ�߽�ĵ�����Ϊ�Ǳ߽��
	for (i = 0; i<nHeight; i++)
	{
		for (j = 0; j<nWidth; j++)
		{
			if (N[i*nWidth + j] != 255)
			{
				dst->imageData[i*nWidthStep + j] = 0;   // ����Ϊ�Ǳ߽��
			}
			else
			{
				dst->imageData[i*nWidthStep + j] = 255;
			}
		}
	}
	cvShowImage("image", dst);
	waitKey(0);
}

void testSIFT(IplImage *src,IplImage *cmp)
{
	Mat srcmat = src;
	Mat cmpmat = cmp;

	//construct the SIFT detector
	int minHessian = 400;
	SiftFeatureDetector siftDetector(minHessian);

	//use SIFT to detect interest points
	vector<KeyPoint> srcKeyPoints;
	vector<KeyPoint> cmpKeyPoints;
	siftDetector.detect(srcmat, srcKeyPoints);
	siftDetector.detect(cmpmat, cmpKeyPoints);

	//draw the interest points on the image
	Mat descriptor1, descriptor2;
	SiftDescriptorExtractor extractor;
	extractor.compute(srcmat, srcKeyPoints, descriptor1);
	extractor.compute(cmpmat, cmpKeyPoints, descriptor2);

	//match the two descriptors
	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> cmatch;
	matcher.match(descriptor1, descriptor2, cmatch);

	//draw the results
	Mat img;
	drawMatches(srcmat, srcKeyPoints, cmpmat, cmpKeyPoints, cmatch, img);
	imshow("SIFT key points", img);

	waitKey(0);
	return;
}

void ahe()
{
	//��������
	//ͼƬ��
	IplImage *src, *mid, *dst;
	IplImage *histframesrc, *histframedst;
	CvSize     size, size1;
	//ֱ��ͼ��
	CvHistogram *histsrc[BLOCK][BLOCK], *histdst;
	int scale;
	float histmin, histmax;

	int bins = 256;
	float range[] = { 0, 255 };
	float *ranges[] = { range };
	//�Ӽ�
	int i, j, m, n;                                //ѭ�������������� 
	float s, t;                                        //�߲���
	int pixcolor;                                    //�м��������������ǰ�涨�壬��
	float sr[BLOCK][BLOCK][256];                                //S(r)ӳ�亯��
	//---------------------------------------------------------------
	//����ͼ��ת��Ϊ�Ҷ�ͼ
	src = cvLoadImage("test2.png");
	size = cvGetSize(src);
	mid = cvCreateImage(size, 8, 1);
	dst = cvCreateImage(size, 8, 1);
	cvCvtColor(src, mid, CV_BGR2GRAY);
	//�������С
	size1.width = size.width / BLOCK;
	size1.height = size.height / BLOCK;
	size.width = size1.width * BLOCK;
	size.height = size1.height * BLOCK;
	//�������������ֱ��ͼ,��һ��255
	for (m = 0; m <= BLOCK-1; m += 1)
	{
		for (n = 0; n <= BLOCK-1; n += 1)
		{
			cvSetImageROI(mid, Rect(m*size1.width, n*size1.height, size1.width, size1.height));
			histsrc[m][n] = cvCreateHist(1, &bins, CV_HIST_ARRAY, ranges, 1);
			cvCalcHist(&mid, histsrc[m][n], 0, 0);
			cvNormalizeHist(histsrc[m][n], 255);
		}
	}
	cvResetImageROI(mid);

	//�����ֱ��ͼ��s��r��ӳ�亯��
	for (m = 0; m <= BLOCK-1; m += 1)
	{
		for (n = 0; n <= BLOCK-1; n += 1)
		{
			sr[m][n][0] = cvQueryHistValue_1D(histsrc[m][n], 0);
			for (i = 1; i <= 255; i++)sr[m][n][i] = sr[m][n][i - 1] + cvQueryHistValue_1D(histsrc[m][n], i);
		}
	}

	//���ұ��߲�
	CvScalar pixel;
	for (j = 0; j<size.height; j += 1)
	{
		for (i = 0; i<size.width; i += 1)
		{
			pixel = cvGet2D(mid, j, i);                                            //��ȡ����
			pixcolor = pixel.val[0];
			m = cvRound(i / size1.width), n = cvRound(j / size1.height);                //M.N��������
			s = (i / float(size1.width) - (2 * m ) / 2.0), t = (j / float(size1.height) - (2 * n ) / 2.0);        //S.T.�����ڲ�ϵ��
			//printf("s:%f t:%f\n", s, t);
			//exit(-1);
			if ((m == 0 && n == 0) || (m == BLOCK && n == BLOCK))pixel.val[0] = sr[m][n][pixcolor];                        //����������ڲ�
			else
				if (m == 0)pixel.val[0] = (1-t)*sr[m][n - 1][pixcolor] + (t)*sr[m][n][pixcolor];
				else
					if (m == BLOCK && n != BLOCK)pixel.val[0] = (1-t)*sr[BLOCK-1][n - 1][pixcolor] + (t)*sr[BLOCK-1][n][pixcolor];
					else
						if (n == 0)pixel.val[0] = (1-s)*sr[m - 1][n][pixcolor] + (s)*sr[m][n][pixcolor];
						else
							if (n == BLOCK && m != BLOCK)pixel.val[0] = (1-s)*sr[m - 1][BLOCK-1][pixcolor] + (s)*sr[m][BLOCK-1][pixcolor];
							else
								pixel.val[0] = (1-s)*(1-t)*sr[m - 1][n - 1][pixcolor] + (1-s)*(t)*sr[m - 1][n][pixcolor] + (s)*(1-t)*sr[m][n - 1][pixcolor] + (s)*(t)*sr[m][n][pixcolor];
			cvSet2D(dst, j, i, pixel);
		}
	}
	cvShowImage("AHE", dst);
	cvShowImage("SRC", src);
	waitKey(0);
}

void clhe(float thresh)
{
	//��������
	//ͼƬ��
	IplImage *src = cvLoadImage("test1.png",0);
	cvShowImage("src", src);
	CvSize  size = cvGetSize(src);
	IplImage *dst = cvCreateImage(size, 8, 1);
	
	int bins = 256;
	float range[] = { 0, 255 };
	float *ranges[] = { range };
	//�Ӽ�
	int i, j, k, m, n;                                //ѭ�������������� 
	float s_r[256];                                //S(r)ӳ�亯��
	float  S;                                   //�������
	int scale;
	float histmin, histmax;
	
	//ֱ��ͼ��
	CvHistogram *histo_src = cvCreateHist(1, &bins, CV_HIST_ARRAY, ranges);
	CvHistogram *histo_dst = cvCreateHist(1, &bins, CV_HIST_ARRAY, ranges);
	
	cvCalcHist(&src, histo_src, 0, 0);                //����ֱ��ͼ
	cvNormalizeHist(histo_src, 255);                //��һ

	//���ƶԱȶ�
	{

		//��ȡ���ֵ
		cvGetMinMaxHistValue(histo_src, &histmin, &histmax);
		thresh *= histmax;

		//����get�������
		S = 0;
		for (i = 1; i <= 255; i++)
		{
			if (cvQueryHistValue_1D(histo_src, i) > thresh)
			{
				S += (cvQueryHistValue_1D(histo_src, i) - thresh);
				cvSetReal1D(histo_src->bins, i, thresh);
			}
		}
		S /= 255;

		//����+ƽ�������
		for (i = 1; i <= 255; i++)
		{
			cvSetReal1D(histo_src->bins,
				i,
				S + cvQueryHistValue_1D(histo_src, i)
				);
		}
	}

	////////////////////////////////

	//�ۼӣ���S(r)ӳ��
	s_r[0] = cvQueryHistValue_1D(histo_src, 0);
	for (i = 1; i <= 255; i++)
	{
		s_r[i] = s_r[i - 1] + cvQueryHistValue_1D(histo_src, i);
	}

	//����ͼ����sr��ϵ����ֱ��ͼ���⻯��
	CvScalar s;
	for (m = 0; m < size.height; m++)
	{
		for (n = 0; n < size.width; n++)
		{
			s = cvGet2D(src, m, n);
			i = s.val[0];//�õ�����ֵ
			i = s_r[i];//�õ�ӳ��ֵ
			s.val[0] = i;//��������ͨ��ֵ
			cvSet2D(dst, m, n, s);
		}
	}

	//SHOWOFFһ����
	cvSmooth(dst, dst, CV_GAUSSIAN);
	cvShowImage("dst", dst);

	//����dstֱ��ͼ
	cvCalcHist(&dst, histo_dst, 0, 0);
	cvNormalizeHist(histo_dst, 255);
	
	cvWaitKey(0);
}

int main()
{
	IplImage *src = cvLoadImage("1.bmp");
	IplImage *src1 = cvLoadImage("test2.png");
	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, 1);
	IplImage *dst1 = cvCreateImage(cvGetSize(src1), src1->depth, src1->nChannels);
	IplImage *cmp = cvLoadImage("mouses.png");
	/*CvSize size;
	size.width = cvGetSize(src).width / 4;
	size.height = cvGetSize(src).height / 4;
	IplImage *dst = cvCreateImage(size, src->depth, src->nChannels);
	IplImage *ppp = cvCreateImage(size, src->depth, src->nChannels);
	cvResize(src, dst);
	cvResize(cmp, ppp);*/
	/*if (src == NULL){
		cout << "Error in read image!!!" << endl;
		return -1;
	}

	//int threshold = otsuThreshold(src);
	//cout << "ostu: " << threshold << endl;
	//int threshold = fuzzyCompactnessThreshold(src);
	//cout << "compactness: " << threshold << endl;
	//int threshold = linearIndicesOfFuzzinessThreshold(src);
	//cout << "linear: " << threshold << endl;
	//int threshold = quadraticIndicesOfFuzzinessThreshold(src);
	//cout << "quadratic: " << threshold << endl;
	//int threshold = fuzzySimilarityThreshold(src);
	//cout << "similarity: " << threshold << endl;
	int threshold = fuzzyDivergenceThreshold(src);
	//cout << "divergence: " << threshold << endl;
	IplImage *gray = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	cvCvtColor(src, gray, CV_RGB2GRAY);
	IplImage *dst = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	cvThreshold(gray, dst, threshold, GrayScale, CV_THRESH_BINARY);
	cvNamedWindow("Origin Image");
	cvShowImage("Origin Image", src);
	cvNamedWindow("Binary Image");
	cvShowImage("Binary Image", dst);
	waitKey(0);
	return 0;*/
	//testSIFT(src,cmp);
	//testMouse();
	//histgramEqualization(src1, dst1, 255, 0);
	//canny(src, dst);
	clhe(0.9);
	//ahe();
	return 0;
}