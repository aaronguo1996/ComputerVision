#include <iostream>
#include <fstream>
#include <opencv.hpp>
//#include <opencv2/ml/ml.hpp>
//#include <opencv2\highgui\highgui.hpp>
using namespace std;
using namespace cv;

/***************************************************
* 【膨胀操作】
* 传入两个矩阵
* 操作后返回一个新的矩阵
* 【实现原理】
* 新建与原图像矩阵大小相同的返回矩阵；
* 遍历图像矩阵，比较图像矩阵与结构矩阵的重叠处是否存在一致
* 如果存在则把返回矩阵该位置置为1，否则置为0。
***************************************************/
Mat dilation(Mat image, Mat mask,int c)
{
	Mat result = Mat::zeros(image.rows, image.cols,CV_8U);
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			bool flag = false;
			for (int k = i - mask.rows / 2; k <= i + mask.rows / 2; k++){
				for (int m = j - mask.cols / 2; m <= j + mask.cols / 2; m++){
					if (k < 0 || m < 0 || k >= image.rows || m >= image.cols)
						continue;
					if (mask.at<uchar>(k + mask.rows / 2 - i,m + mask.cols / 2 - j) == c && c == image.at<uchar>(k,m)){
						flag = true;
						result.at<uchar>(i, j) = c;//1;
						break;
					}
				}
				if (flag) break;
			}
			if (!flag)	result.at<uchar>(i, j) = 0;
		}
	}
	return result;
}

/***************************************************
* 【腐蚀操作】
* 传入两个矩阵
* 操作后返回一个新的矩阵
* 【实现原理】
* 新建与原图像矩阵大小相同的返回矩阵；
* 遍历图像矩阵，比较图像矩阵与结构矩阵的重叠处是否全部一致
* 如果一致则把返回矩阵该位置置为1，否则置为0。
***************************************************/
Mat erosion(Mat image, Mat mask,int c)
{
	Mat result = image * 1;
	for (int i = 1; i < image.rows-1; i++)
	{
		for (int j = 1; j < image.cols-1; j++)
		{
			bool flag = false;
			/*if (i< mask.rows / 2 || j < mask.cols / 2 || image.rows - i < mask.rows / 2 || image.cols - j < mask.cols / 2)
				continue;*/
			for (int k = i - mask.rows / 2; k <= i + mask.rows / 2; k++)
			{
				for (int m = j - mask.cols / 2; m <= j + mask.cols / 2; m++)
				{
					if (k < 0 || m < 0 || k >= image.rows || m >= image.cols)
					{
						continue;
					}
					if (mask.at<uchar>(k + mask.rows / 2 - i, m + mask.cols / 2 - j) == c && c != image.at<uchar>(k, m))
					{
						flag = true;
						result.at<uchar>(i, j) = 0;
						break;
					}
				}
				if (flag)	break;
			}
			if (!flag)	result.at<uchar>(i, j) = c;//1;
		}
	}
	return result;
}

/***************************************************
* 【开操作】
* 传入两个矩阵
* 操作后返回一个新的矩阵
* 【实现原理】
* 先进行一次腐蚀操作；
* 再进行一次膨胀操作。
***************************************************/
Mat open(Mat image, Mat mask,int c)
{
	Mat result = erosion(image, mask,c);
	result = dilation(result, mask,c);
	return result;
}

/***************************************************
* 【闭操作】
* 传入两个矩阵
* 操作后返回一个新的矩阵
* 【实现原理】
* 先进行一次膨胀操作；
* 再进行一次腐蚀操作。
***************************************************/
Mat close(Mat image, Mat mask, int c)
{
	Mat result = dilation(image, mask,c);
	result = erosion(result, mask, c);
	return result;
}

Mat hitormiss(Mat src, Mat kernel)
{
	Mat k1 = (kernel == 1)/255;//[TODO]
	Mat k2 = (kernel == -1)/255;
	normalize(src, src, 0, 1, NORM_MINMAX);
	Mat dst = Mat::zeros(src.rows + 2, src.cols + 2, CV_8UC1);
	Mat ret = Mat::zeros(src.rows, src.cols, CV_8UC1);
	for (int i = 1; i <= src.rows; i++){
		for (int j = 1; j <= src.cols; j++){
			dst.at<uchar>(i, j) = src.at<uchar>(i - 1, j - 1);
		}
	}
	//cout << dst << endl;
	Mat e1, e2;
	e1 = erosion(dst, k1,1);
	e2 = erosion(1 - dst, k2,1);
	dst = e1&e2;
	for (int i = 1; i <= src.rows; i++){
		for (int j = 1; j <= src.cols; j++){
			ret.at<uchar>(i - 1, j - 1) = dst.at<uchar>(i, j);
		}
	}
	//cout << ret << endl;;
	return ret;
}

//矩阵逆时针旋转90度
void Rotation1(Mat &arr)
{
	Mat tmp = Mat::zeros(arr.cols, arr.rows, CV_8SC1);//局部变量，函数调用完后会自动释放
	int dst = 0;//arr.rows - 1;	  //这里我们从目标矩阵的最后一列开始存放数据

	for (int i = 0; i<arr.rows; i++, dst++)
		for (int j = 0; j<arr.cols; j++)
			tmp.at<char>(2-j, dst) = arr.at<char>(i, j);

	//将旋转后的矩阵保存回原来的矩阵
	for (int i = 0; i<arr.cols; i++)
		for (int j = 0; j<arr.rows; j++)
			arr.at<char>(i, j) = tmp.at<char>(i, j);
}

void Rotation2(Mat &arr)
{
	Mat tmp = Mat::zeros(arr.cols, arr.rows, CV_8SC1);//局部变量，函数调用完后会自动释放
	int dst = arr.rows - 1;	  //这里我们从目标矩阵的最后一列开始存放数据

	for (int i = 0; i<arr.rows; i++, dst--)
		for (int j = 0; j<arr.cols; j++)
			tmp.at<char>(j, dst) = arr.at<char>(i, j);

	//将旋转后的矩阵保存回原来的矩阵
	for (int i = 0; i<arr.cols; i++)
		for (int j = 0; j<arr.rows; j++)
			arr.at<char>(i, j) = tmp.at<char>(i, j);
}

bool compare(Mat mat1, Mat mat2)
{
	for (int i = 0; i < mat1.rows; i++)
		for (int j = 0; j < mat1.cols; j++)
			if (mat1.at<char>(i, j) != mat2.at<char>(i, j))
				return false;
			else
				continue;

	return true;
}

void display(Mat a)
{
	Mat dst = (a == 1);
	IplImage binaryImage = dst;
	cvNamedWindow("Image");
	cvShowImage("Image", &binaryImage);
	waitKey(0);
}

Mat thin(Mat src, Mat kernel)
{
	src = src - hitormiss(src, kernel);
	//src = src - hitormiss(src, kernel);
	return src;
}

Mat thicken(Mat src, Mat kernel)
{
	src = src | hitormiss(src, kernel);
	return src;
}

//测试代码
int test()
{
	IplImage *image = cvLoadImage("abcdef.jpg", 1);
	//cvNamedWindow("Src Image");
	//cvShowImage("Src Image", image);

	IplImage *grayImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	cvCvtColor(image, grayImage, CV_RGB2GRAY);
	//cvNamedWindow("Gray Image");
	//cvShowImage("Gray Image", grayImage);

	IplImage *binaryImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	cvThreshold(grayImage, binaryImage, 150, 255, CV_THRESH_BINARY);
	cvNamedWindow("Binary Image");
	cvShowImage("Binary Image", binaryImage);
	ofstream ofs("binary.txt");
	ofs << Mat(binaryImage) << endl;;
	Mat imageMat(binaryImage,0);

	//构造一个3*3的正方形进行膨胀和腐蚀处理
	Mat structureElement = Mat::zeros(3, 3, CV_8U);
	/*for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			structureElement.at <uchar>( i, j )= 255;//根据前景色具体情况定为白色或者黑色，此处因测试图片前景色为白色故设为白色
		}
	}*/
	structureElement.at<uchar>(0, 1) = structureElement.at <uchar>(2, 1)=
	structureElement.at <uchar>(1, 0) = structureElement.at <uchar>(1, 1) = structureElement.at <uchar>(1, 2) = 255;
	ofs.close();
	ofs.open("struct.txt");
	ofs << structureElement << endl;
	ofs.close();
	Mat ret = dilation(imageMat, structureElement,255);
	IplImage dilationImage = ret;
	cvNamedWindow("Dilated Image");
	cvShowImage("Dilated Image", &dilationImage);

	ret = erosion(imageMat, structureElement,255);
	IplImage erosionImage = ret;
	cvNamedWindow("Erosed Image");
	cvShowImage("Erosed Image", &erosionImage);
	waitKey(0);
	return 0;
}

void testThin()
{
	/*cv::Mat a = (cv::Mat_<uchar>(16, 16) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	cv::Mat b1 = (cv::Mat_<char>(3, 3) << -1, -1, -1, 0, 1, 0, 1, 1, 1);
	cv::Mat b2 = (cv::Mat_<char>(3, 3) << 0, -1, -1, 1, 1, -1, 0, 1, 0);*/
	cv::Mat a = (cv::Mat_<uchar>(5,11) << 
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
		1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0);
	cv::Mat b1 = (cv::Mat_<char>(3, 3) << -1, -1, -1, 0, 1, 0, 1, 1, 1);
	cv::Mat b2 = (cv::Mat_<char>(3, 3) << 0, -1, -1, 1, 1, -1, 1, 1, 0);
	cout << a << endl;
	a = thin(a, b1);
	cout << a << endl;
}

void testErosion()
{
	cv::Mat a = (cv::Mat_<uchar>(16, 16) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	cv::Mat b1 = (cv::Mat_<char>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
	//a = erosion(a, b1);
	a = dilation(a, b1,1);
	cout << a << endl;
}

void testHitormiss()
{
	cv::Mat a = (cv::Mat_<uchar>(16, 16) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
		0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
		0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	cv::Mat b1 = (cv::Mat_<char>(3, 3) << 0, 1, 0, -1, 1, 1, -1, -1, 0);
	//a = thin(a, b1, b2);
	Mat a1 = a * 1;
	Mat a2 = a * 1;
	Mat a3 = a * 1;
	Mat a4 = a * 1;
	a1 = hitormiss(a, b1);
	Rotation1(b1);
	a2 = hitormiss(a, b1);
	cout << a2 << endl;
	Rotation1(b1);
	a3 = hitormiss(a, b1);
	Rotation1(b1);
	a4 = hitormiss(a, b1);
	a = a1 | a2 | a3 | a4;
	cout << a << endl;
}

void testSkeleton()
{
	IplImage *image = cvLoadImage("potato1.png", 1);
	IplImage *grayImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	cvCvtColor(image, grayImage, CV_RGB2GRAY);
	IplImage *binaryImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	cvThreshold(grayImage, binaryImage, 146, 255, CV_THRESH_BINARY);
	cvNamedWindow("Binary Image");
	cvShowImage("Binary Image", binaryImage);
	cv::Mat b1 = (cv::Mat_<char>(3, 3) << /*0, 1, 0, 0, 1, 0, 1, 0, 1);*/-1, -1, -1, 0, 1, 0, 1, 1, 1);
	cv::Mat b2 = (cv::Mat_<char>(3, 3) << -1, -1, 0, -1, 1, 1, 0, 1, 1);//0, -1, -1, 1, 1, -1, 0, 1, 0);//-1, 0, 0, -1, 1, -1, -1, -1, -1);//1, 0, 0, 0, 1, 0, 1, 0, 1);//-1, 0, 0, -1, 1, -1, -1, -1, -1);
	cv::Mat b3 = (cv::Mat_<char>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
	cv::Mat b4 = (cv::Mat_<char>(3, 3) << -1, -1, -1, -1, 1, -1, -1, 0, 0);
	cv::Mat b5 = (cv::Mat_<char>(3, 3) << -1, -1, -1, -1, 1, -1, 0, 0, -1);

	Mat a = Mat(binaryImage) * 1;
	normalize(a, a, 0, 1, NORM_MINMAX);
	cout << "dilation..." << endl;
	a = dilation(a, b3,1);
	a = dilation(a, b3,1);
	cout << "closing and opening..." << endl;
	for (int i = 0; i < 7; i++)
	{
		a = close(a, b3,1);
		a = open(a, b3,1);
	}
	cout << "thinning..." << endl;
	Mat tmp = Mat::zeros(cvGetSize(binaryImage), CV_8UC1);
	while (!compare(tmp, a)){
		tmp = a * 1;
		thin(a, b1);
		Rotation2(b1);
		thin(a, b2);
		Rotation2(b2);
	}
	cout << "spruning..." << endl;
	for (int i = 0; i<20; i++){
		tmp = a * 1;
		thin(a, b4);
		Rotation2(b4);
		thin(a, b5);
		Rotation2(b5);
	}
	for (int i = 0; i < image->height; i++){
		for (int j = 0; j < image->width; j++){
			if (a.at<char>(i, j) == 1){
				((uchar*)(image->imageData + image->widthStep*i))[j * 3 + 1] = 100;
				((uchar*)(image->imageData + image->widthStep*i))[j * 3 ] = 200;
			}
		}
	}
	cvNamedWindow("Final Image");
	cvShowImage("Final Image", image);
	waitKey(0);
}
int main()
{
	//testSkeleton();
	test();
	return 0;
}