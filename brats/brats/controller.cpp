#include "Controller.h"

Controller::Controller()
{
	int i = 0, j = 0;
	for (i = 0; i < MAX_IMAGE_NUM; i++){
		char buf[NAME_LEN];
		for (j = 0; j < IMAGE_TYPE_NUM; j++){
			string image_type = type2string(j);
			sprintf_s(buf, "hg0005/BRATS_HG0005_%s/BRATS_HG0005_%s_%d.png", image_type.c_str(), image_type.c_str(), i+67);
			src[i][j] = cvLoadImage(buf,0);
			if (src[i][j] == NULL){
				cout << "Cannot open the image " << buf << endl;
				exit(-1);
			}
		}
	}
	ofs.open("out.txt");
	colors = Mat::zeros(IMAGE_SIZE,IMAGE_TYPE_NUM,CV_32FC1);
	gradients = Mat::zeros(IMAGE_SIZE, IMAGE_TYPE_NUM, CV_32FC1);
	spectrums = Mat::zeros(PIXEL_COUNT, IMAGE_TYPE_NUM, CV_32FC1);
	textures = Mat::zeros(PIXEL_COUNT, IMAGE_TYPE_NUM, CV_32FC1);
	fractals = Mat::zeros(PIXEL_COUNT, IMAGE_TYPE_NUM, CV_32FC1);
	results = Mat::zeros(IMAGE_SIZE,1, CV_32FC1);
	fractalRet = Mat::zeros(PIXEL_COUNT, 1, CV_32FC1);
}

Controller::~Controller()
{
	ofs.close();
	int i = 0, j = 0;
	for (i = 0; i < MAX_IMAGE_NUM; i++){
		for (j = 0; j < IMAGE_TYPE_NUM; j++){
			if (src[i][j]){
				cvReleaseImage(&src[i][j]);
			}
		}
	}
}

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

	//统计灰度级中每个像素在整幅图像中的个数
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			pixelCount[(int)data[i * width + j]]++;
		}
	}

	//计算每个像素在整幅图像中的比例
	for (i = 0; i < GrayScale; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//遍历灰度级[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u,
		deltaTmp, deltaMax = 0;
	for (i = 0; i < GrayScale; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
		for (j = 0; j < GrayScale; j++)
		{
			if (j <= i)   //背景部分
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //前景部分
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

void Controller::extractColor(int n)
{
	int i = n - 67, j = 0;
	for (j = 0; j < IMAGE_TYPE_NUM; j++){
		int step = src[i][j]->widthStep/sizeof(uchar);
		int width = src[i][j]->width;
		int height = src[i][j]->height;
		Mat data = src[i][j];
		normalize(data, data, 0, 255, NORM_MINMAX);
		int k = 0, m = 0, a = 0, b = 0;
		for (k = 0; k < height; k++){
			for (m = 0; m < width; m++){
				colors.at<float>(k*width + m, j) = src[i][j]->imageData[k*step + m];
			}
		}
	}
	normmat(colors);
}

//this cannot help
void Controller::extractEdge(int n)
{
	int i = n - 67, j = 0;
	for (j = 0; j < IMAGE_TYPE_NUM; j++){
		int threshold = otsuThreshold(src[i][j]);
		IplImage *gray = src[i][j];//cvCreateImage(cvGetSize(src[i][j]), IPL_DEPTH_8U, 1);
		//cvCvtColor(src[i][j], gray, CV_RGB2GRAY);
		IplImage *dst = cvCreateImage(cvGetSize(src[i][j]), IPL_DEPTH_8U, 1);
		cvThreshold(gray, dst, threshold, GrayScale, CV_THRESH_BINARY);
		Mat in = dst;
		Mat out = Mat::zeros(cvGetSize(dst), CV_8UC1);
		Canny(in, out,100,200);
		imshow("canny", out);
		waitKey(0);
	}
}

void Controller::extractGradient(int n)
{
	int i = n - 67, j = 0;
	for (j = 0; j < IMAGE_TYPE_NUM; j++){
		IplImage *gray = src[i][j];
		Mat in ,out_x,out_y,out;
		in = gray;
		normalize(in, in, 0, 255, NORM_MINMAX);
		Sobel(in, out_x, IPL_DEPTH_16S, 1, 0, 3);
		convertScaleAbs(out_x, out_x);
		Sobel(in, out_y, IPL_DEPTH_16S, 0, 1, 3);
		convertScaleAbs(out_y, out_y);
		addWeighted(out_x, 0.5, out_y, 0.5, 0, out);
		//normalize(out, out, 0, 255, NORM_MINMAX);
		int width = src[i][j]->width;
		int height = src[i][j]->height;
		int k = 0, m = 0;
		for (k = 0; k < height; k++){
			for (m = 0; m < width; m++){
				gradients.at<float>((k)*width + m, j) = out.at<uchar>(k,m);
			}
		}
	}
	normmat(gradients);
}

void Controller::extractFractal(int n)
{
	int i = n - 67, j = 0;
	for (j = 0; j < IMAGE_TYPE_NUM; j++){
		int step = src[i][j]->widthStep / sizeof(uchar);
		int width = src[i][j]->width;
		int height = src[i][j]->height;
		Mat data = src[i][j];
		normalize(data, data, 0, 255, NORM_MINMAX);
		int k = 0, m = 0, a = 0, b = 0;
		int avg = 0;
		for (k = 0; k < height / FRACTAL_SIZE; k++){
			for (m = 0; m < width / FRACTAL_SIZE; m++){
				int p1 = data.at<uchar>(k*FRACTAL_SIZE, m*FRACTAL_SIZE);
				int p2 = data.at<uchar>(k*FRACTAL_SIZE, (m + 1)*FRACTAL_SIZE - 1);
				int p3 = data.at<uchar>((k + 1)*FRACTAL_SIZE - 1, (m + 1)*FRACTAL_SIZE - 1);
				int p4 = data.at<uchar>((k + 1)*FRACTAL_SIZE - 1, m*FRACTAL_SIZE);
				int pc = (p1 + p2 + p3 + p4) / 4;
				Mat ea = (Mat_<float>(1, 3) << -FRACTAL_SIZE / 2, -FRACTAL_SIZE / 2, p1 - pc);
				Mat eb = (Mat_<float>(1, 3) << -FRACTAL_SIZE / 2, FRACTAL_SIZE / 2, p2 - pc);
				Mat ec = (Mat_<float>(1, 3) << FRACTAL_SIZE / 2, FRACTAL_SIZE / 2, p3 - pc);
				Mat ed = (Mat_<float>(1, 3) << FRACTAL_SIZE / 2, -FRACTAL_SIZE, p4 - pc);
				Mat Sead = ea.cross(ed); 
				int ead = sqrt(pow(Sead.at<float>(0, 0), 2.0) + pow(Sead.at<float>(0, 1), 2.0) + pow(Sead.at<float>(0, 2), 2.0));
				Mat Seab = ea.cross(eb);
				int eab = sqrt(pow(Seab.at<float>(0, 0), 2.0) + pow(Seab.at<float>(0, 1), 2.0) + pow(Seab.at<float>(0, 2), 2.0));
				Mat Sebc = eb.cross(ec);
				int ebc = sqrt(pow(Sebc.at<float>(0, 0), 2.0) + pow(Sebc.at<float>(0, 1), 2.0) + pow(Sebc.at<float>(0, 2), 2.0));
				Mat Secd = ec.cross(ed);
				int ecd = sqrt(pow(Secd.at<float>(0, 0), 2.0) + pow(Secd.at<float>(0, 1), 2.0) + pow(Secd.at<float>(0, 2), 2.0));
				fractals.at<float>(k*width / FRACTAL_SIZE + m, j) = log(ead+eab+ebc+ecd)/log(FRACTAL_SIZE);
			}
		}
	}
	normmat(fractals);
}

void Controller::extractSpectrum(int n)
{
	int i = n - 67, j = 0;
	for (j = 0; j < IMAGE_TYPE_NUM; j++){
		Mat input = src[i][j];
		Mat planes[] = { Mat_<float>(input), Mat::zeros(input.size(), CV_32F) };
		Mat complete;
		merge(planes, 2, complete);

		dft(complete, complete);
		split(complete, planes);
		magnitude(planes[0], planes[1], planes[0]);
		int width = src[i][j]->width;
		int height = src[i][j]->height;
		int k = 0, v = 0;
		for (k = 0; k < height; k++){
			for (v = 0; v < width; v++){
				spectrums.at<float>((k)*width + v, j) = complete.at<float>(k, v);
			}
		}
	}
	normmat(spectrums);
}

void Controller::extractTexture(int n)
{
	int i = n - 67, j = 0;
    Mat dst;
    int iSize=101;
	int step = src[i][j]->widthStep / sizeof(uchar);
	int width = src[i][j]->width;
	int height = src[i][j]->height;
	int k = 0, m = 0, a = 0, b = 0;
	//int avg = 0;
	//double sig = 2 * CV_PI, th = 4*CV_PI / 8 + CV_PI / 2, lm = 1.0, gm = 1;
	//Mat kernel = getGaborKernel(Size(iSize, iSize), sig, th, lm, gm);
	for (j = 0; j < IMAGE_TYPE_NUM; j++){
		//Mat I = src[i][j];
		//normalize(I, I, 1, 0, CV_MINMAX, CV_32F);
		//filter2D(I, dst, CV_32F, kernel);
		//normalize(dst, dst, 0, 255, CV_MINMAX, CV_32F);
		//imshow("dst", dst);
		//waitKey(0);
		for (k = 0; k < height / FRACTAL_SIZE; k++){
			for (m = 0; m < width / FRACTAL_SIZE; m++){
				//avg = 0;
				/*for (a = 0; a < FRACTAL_SIZE; a++){
					for (b = 0; b < FRACTAL_SIZE; b++){
						avg += dst.at<float>((k*FRACTAL_SIZE + a), (m*FRACTAL_SIZE + b));
					}
				}
				avg = avg / (FRACTAL_SIZE*FRACTAL_SIZE);*/
				textures.at<float>(k*width / FRACTAL_SIZE + m, j) = calGLCM(n,j,2,k*FRACTAL_SIZE,m*FRACTAL_SIZE);
			}
		}
	}
	ofs << textures;
	normmat(textures);
}

Mat Controller::generateTestcase(int type,int n)
{
	try{
		CvSVM svm;
		SVMParams params;

		params.kernel_type = CvSVM::RBF;
		params.svm_type = CvSVM::C_SVC;
		switch (type){
		case 0:
			extractColor(n);
			generateResults(n);
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);//110-600 120-250 90-1000
			svm.train(colors, results, Mat(), Mat(), params);
			break;
		case 1:
			extractGradient(n);
			generateResults(n);
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 150, 1e-6);//120-100
			svm.train(gradients, results, Mat(), Mat(), params);
			break;
		case 2:
			extractTexture(n);
			generateFractalResults(n);
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 45, 1e-6);
			svm.train(textures, fractalRet, Mat(), Mat(), params);
			break;
		case 3:
			extractFractal(n);
			generateFractalResults(n);
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 45, 1e-6);
			svm.train(fractals, fractalRet, Mat(), Mat(), params);
			break;
		default:
			break;
		}
		
		int i = n-67, j = 0;
		int step = src[i][0]->widthStep / sizeof(uchar);
		int width = src[i][0]->width;
		int height = src[i][0]->height;
		int k = 0, m = 0, a = 0, b = 0;

		Mat merge = Mat::zeros(cvGetSize(src[i][0]), CV_8UC1);
		for (int q = 0; q < 4; q++){
			switch(type){
			case 0:
				extractColor(n+q);
				svm.predict(colors, results);
				break;
			case 1:
				extractGradient(n+q);
				svm.predict(gradients, results);
				break;
			case 2:
				extractTexture(n + q);
				svm.predict(textures, results);
				break;
			case 3:
				extractFractal(n + q);
				svm.predict(fractals, results);
				break;
			default:
				break;
			}
			Mat out = Mat::zeros(cvGetSize(src[i][0]), CV_8UC1);
			switch (type){
			case 0: case 1:
				for (k = 0; k < height; k++){
					for (m = 0; m < width; m++){
						int index = k*width + m;
						if (results.at<float>(index) == 0)
							out.at<uchar>(k, m) = 0;
						else
							out.at<uchar>(k, m) = 255;
					}
				}
				break;
			case 2: case 3:
				for (k = 0; k < height / FRACTAL_SIZE; k++){
					for (m = 0; m < width / FRACTAL_SIZE; m++){
						for (a = 0; a < FRACTAL_SIZE; a++){
							for (b = 0; b < FRACTAL_SIZE; b++){
								int index = k*width / FRACTAL_SIZE + m;
								if (results.at<float>(index) == 0)
									out.at<uchar>((k*FRACTAL_SIZE + a), m*FRACTAL_SIZE + b) = 0;
								else
									out.at<uchar>((k*FRACTAL_SIZE + a), m*FRACTAL_SIZE + b) = 255;
							}
						}
					}
				}
				break;
			default:
				break;
			}
			for (k = 0; k < height; k++){
				for (m = 0; m < width; m++){
					if (out.at<uchar>(k, m) != 0 || merge.at<uchar>(k, m) != 0)
						merge.at<uchar>(k, m) = 255;
					else
						merge.at<uchar>(k, m) = 0;
				}
			}
			imshow("output", merge);
			waitKey(0);
		}
		return merge;
	}
	catch (Exception &e){
		qDebug() << "OOPS!_" << e.msg.c_str();
	}
}

void Controller::test()
{
	int n = 80;
	int step = src[0][0]->widthStep / sizeof(uchar);
	int width = src[0][0]->width;
	int height = src[0][0]->height;
	int k = 0, m = 0;
	Mat merge = Mat::ones(cvGetSize(src[0][0]), CV_8UC1);
	Mat out[4];
	for (int i = 0; i < 4; i++){
		out[i] = generateTestcase(i,n);
	}
	for (k = 0; k < height; k++){
		for (m = 0; m < width; m++){
			int vote = 0;
			for (int i = 0; i < 4; i++)
				if (out[i].at<uchar>(k, m) != 0)
					vote++;
			if (vote>=3)
				merge.at<uchar>(k, m) = 255;
			else
				merge.at<uchar>(k, m) = 0;
		}
	}
	imshow("output", merge);
	waitKey(0);
}

void Controller::generateFractalResults(int n)
{
	int i = 0, j = 0, k = 0, m = 0;
	char buf[NAME_LEN];
	sprintf_s(buf, "hg0005/BRATS_HG0005_truth/BRATS_HG0005_truth_%d.png", n);
	IplImage *p = cvLoadImage(buf,0);
	if (p == NULL){
		cout << "Cannot open the image " << buf << endl;
		exit(-1);
	}
	int step = p->widthStep;
	int width = p->width;
	int height = p->height;
	int flag = 0;
	for (i = 0; i < height/FRACTAL_SIZE; i++){
		for (j = 0; j < width/FRACTAL_SIZE; j++){
			flag = 0;
			for (m = 0; m < FRACTAL_SIZE; m++){
				for (k = 0; k < FRACTAL_SIZE; k++){
					int index = step*(i*FRACTAL_SIZE+m) + (j*FRACTAL_SIZE+k);
					if ((uchar)p->imageData[index] == 0){
						//do nothing
					}
					else{
						flag = 1;
						break;
					}
				}
				if (flag == 1) break;
			}
			fractalRet.at<float>(i*width/FRACTAL_SIZE + j) = flag;
		}
	}
	//ofs << results;
}

void Controller::generateResults(int n)
{
	int i = 0, j = 0, k = 0, m = 0;
	char buf[NAME_LEN];
	sprintf_s(buf, "hg0005/BRATS_HG0005_truth/BRATS_HG0005_truth_%d.png", n);
	IplImage *p = cvLoadImage(buf, 0);
	if (p == NULL){
		cout << "Cannot open the image " << buf << endl;
		exit(-1);
	}
	int step = p->widthStep;
	int width = p->width;
	int height = p->height;
	int flag = 0;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			int index = i*step + j;
			if ((uchar)p->imageData[index] == 0){
				results.at<float>(i*width + j) = 0;
			}
			else{
				results.at<float>(i*width + j) = 1;
			}
		}
	}
	//ofs << results;
}

string Controller::type2string(int t)
{
	switch (t){
	case FLAIR:
		return "FLAIR";
	case T1:
		return "T1";
	case T1C:
		return "T1C";
	case T2:
		return "T2";
	default:
		return "";
	}
	return "";
}

void Controller::normmat(Mat src)
{
	int i = 0, j = 0;
	float m = 0, a = 0;
	for (i = 0; i < src.cols; i++){
		m = a = src.at<float>(0, i);
		for (j = 0; j < src.rows; j++){
			m = min(m, src.at<float>(j, i));
			a = max(a, src.at<float>(j, i));
		}
		for (j = 0; j < src.rows; j++){
			if (a - m == 0)
				src.at<float>(j, i) = 0;
			else
				src.at<float>(j, i) = (src.at<float>(j, i) - m) / (a - m);
		}
	}
}

double Controller::calGLCM(int n,int k,int angleDirection,int startx,int starty)
{
	int m = n - 67, i = 0, j = 0;
	int width, height;
	IplImage *bWavelet = src[m][k];
	if (NULL == bWavelet)
		return 0;

	width = FRACTAL_SIZE;
	height = FRACTAL_SIZE;

	int * glcm = new int[GLCM_CLASS * GLCM_CLASS];
	int * histImage = new int[width * height];

	if (NULL == glcm || NULL == histImage)
		return 0;

	//灰度等级化---分GLCM_CLASS个等级
	uchar *data = (uchar*)bWavelet->imageData;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			histImage[i * width + j] = (int)(data[bWavelet->widthStep * (i+startx) + (j+starty)] * GLCM_CLASS / 256);
		}
	}

	//初始化共生矩阵
	for (i = 0; i < GLCM_CLASS; i++)
		for (j = 0; j < GLCM_CLASS; j++)
			glcm[i * GLCM_CLASS + j] = 0;

	//计算灰度共生矩阵
	int w, l;
	//水平方向
	if (angleDirection == GLCM_ANGLE_HORIZATION)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				if (j + GLCM_DIS >= 0 && j + GLCM_DIS < width)
				{
					k = histImage[i * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (j - GLCM_DIS >= 0 && j - GLCM_DIS < width)
				{
					k = histImage[i * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//垂直方向
	else if (angleDirection == GLCM_ANGLE_VERTICAL)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				if (i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//对角方向
	else if (angleDirection == GLCM_ANGLE_DIGONAL)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];

				if (j + GLCM_DIS >= 0 && j + GLCM_DIS < width && i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (j - GLCM_DIS >= 0 && j - GLCM_DIS < width && i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}

	//计算特征值
	double entropy = 0, energy = 0, contrast = 0, homogenity = 0;
	for (i = 0; i < GLCM_CLASS; i++)
	{
		for (j = 0; j < GLCM_CLASS; j++)
		{
			//熵
			if (glcm[i * GLCM_CLASS + j] > 0)
				entropy -= glcm[i * GLCM_CLASS + j] * log10(double(glcm[i * GLCM_CLASS + j]));
			//能量
			energy += glcm[i * GLCM_CLASS + j] * glcm[i * GLCM_CLASS + j];
			//对比度
			contrast += (i - j) * (i - j) * glcm[i * GLCM_CLASS + j];
			//一致性
			homogenity += 1.0 / (1 + (i - j) * (i - j)) * glcm[i * GLCM_CLASS + j];
		}
	}
	//返回特征值
	//i = 0;
	//featureVector[i++] = entropy;
	//featureVector[i++] = energy;
	//featureVector[i++] = contrast;
	//featureVector[i++] = homogenity;

	delete[] glcm;
	delete[] histImage;

	return entropy;
}