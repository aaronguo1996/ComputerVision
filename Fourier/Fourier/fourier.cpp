#include "fourier.h"
#include <fstream>
using namespace std;

MyPainterWidget::MyPainterWidget(QWidget *parent)
	:QWidget(parent)
{
}

MyPainterWidget::~MyPainterWidget()
{

}

void MyPainterWidget::paintEvent(QPaintEvent *p)
{ 
	QPainter painter(this);
	QPen pen;
	pen.setColor(qRgba(100,100,100,0));
	pen.setWidth(1);
	painter.setPen(pen);
	for (int i = 0; i <= 400 / 25; i++){
		painter.drawLine(QPoint(5, i * 25 + 5), QPoint(825, i * 25 + 5));
	}
	for (int i = 0; i <= 800 / 25; i++){
		painter.drawLine(QPoint(i * 25 + 5, 5), QPoint(i * 25 + 5, 610));
	}
	pen.setColor(qRgba(0, 0, 255, 0));
	pen.setWidth(4);
	painter.setPen(pen);
	painter.drawLine(QPoint(5,305), QPoint(825, 305));
	painter.drawLine(QPoint(5, 5), QPoint(5,610));
	pen.setColor(Qt::darkCyan);
	painter.setPen(pen);
	for (int i = 0; i < lines.size(); i++){
		//qDebug() << lines[i].first << lines[i].second << endl;
		painter.drawLine(lines[i].first, lines[i].second);
	}
	//painter.drawLines(lines);
}

void MyPainterWidget::mousePressEvent(QMouseEvent *e)
{
	setCursor(Qt::PointingHandCursor);
	start = e->pos();
	end = e->pos();
	//this->lines.push_back(start);
	if (start.x() >= 10 && start.y()>=5&&start.x()<=810&&start.y()<=305){
		emit sendPos(start.x(), start.y());
		this->isPressed = true;
	}
}

void MyPainterWidget::mouseMoveEvent(QMouseEvent *e)
{
	if (this->isPressed){
		end = e->pos();
		if (end.x() > start.x()&&end.x()>=8 && end.x()<=808 && end.y()<=305 && end.y()>=5){
			emit sendPos(end.x(), end.y());
			this->lines.push_back(qMakePair(start, end));
			update();
			start = end;
		}
	}
}

void MyPainterWidget::mouseReleaseEvent(QMouseEvent *e)
{
	setCursor(Qt::ArrowCursor);
	this->isPressed = false;
}

Fourier::Fourier(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	init(); 
	src_x = Mat::zeros(1, 1024, CV_32F);
	src_y = Mat::zeros(1, 1024, CV_32F);
	dst = Mat::zeros(1, 1024, CV_32F);
	factors = Mat::zeros(1, 1024, CV_32F);
	copyFactors = Mat::zeros(1, 1024, CV_32F);
	freq = Mat::zeros(1, 1024, CV_32F);
	index = 0;
	connect(input, SIGNAL(sendPos(int, int)), this, SLOT(collect(int, int)));
	connect(calc, SIGNAL(clicked()), this, SLOT(calcDft()));
	connect(reverse, SIGNAL(clicked()), this, SLOT(reDft()));
	connect(removesin, SIGNAL(clicked()), this, SLOT(changeSinFactor()));
	connect(removecos, SIGNAL(clicked()), this, SLOT(changeCosFactor()));
	connect(frequency, SIGNAL(sliderMoved(int)), this, SLOT(updateFactors()));
}

Fourier::~Fourier()
{

}

void Fourier::init()
{
	panel = new QWidget(this);
	QHBoxLayout *mainLayout = new QHBoxLayout(panel);
	QVBoxLayout *leftLayout = new QVBoxLayout(panel);

	CvSize size;
	size.height = 615;
	size.width = 1230;
	IplImage *image = cvCreateImage(size, IPL_DEPTH_8U, 3);
	drawAxis(830, 600, 25, image);
	Mat src = image;
	cv::cvtColor(src, src, CV_BGR2RGB);
	QImage img = QImage((const unsigned char*)(src.data), src.cols, src.rows, QImage::Format_RGB888);
	
	input = new MyPainterWidget(panel);
	//QLabel *il = new QLabel(input);
	//il->setPixmap(QPixmap::fromImage(img));
	input->setFixedSize(QSize(830, 350));
	inputLabel = new QLabel("Input Image", input);
	fourier = new QWidget(panel);
	fl = new QLabel(fourier);
	fl->setPixmap(QPixmap::fromImage(img));
	fourier->setFixedSize(QSize(830, 350));
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.5, 0.5, 0, 1);
	cvPutText(image, "0", cvPoint(5, 300), &font, CV_RGB(0, 0, 0));
	cvPutText(image, "T/4", cvPoint(205, 300), &font, CV_RGB(0, 0, 0));
	cvPutText(image, "T/2", cvPoint(405, 300), &font, CV_RGB(0, 0, 0));
	cvPutText(image, "3T/4", cvPoint(605, 300), &font, CV_RGB(0, 0, 0));
	cvPutText(image, "T", cvPoint(805, 300), &font, CV_RGB(0, 0, 0));
	output = new QWidget(panel);
	ol = new QLabel(output);
	ol->setPixmap(QPixmap::fromImage(img));
	output->setFixedSize(QSize(830, 350));
	outputLabel = new QLabel("Simulated Image", output);
	
	fourLabel = new QLabel("Fourier Spectrum", fourier);
	leftLayout->addWidget(inputLabel);
	leftLayout->addWidget(input);
	leftLayout->addWidget(outputLabel);
	leftLayout->addWidget(output);
	leftLayout->addWidget(fourLabel);
	leftLayout->addWidget(fourier);

	controller = new QWidget(panel);
	mainLayout->addLayout(leftLayout);
	mainLayout->addWidget(controller);

	QVBoxLayout *rightLayout = new QVBoxLayout(controller);
	number = new QLabel("Total sampling points: ", controller);
	rightLayout->addWidget(number);
	factor = new QLabel("Please choose the frequency you want to query", controller);
	rightLayout->addWidget(factor);
	frequency = new QSlider(controller);
	frequency->setDisabled(true);
	frequency->setOrientation(Qt::Horizontal);
	rightLayout->addWidget(frequency);
	QHBoxLayout *cosLayout = new QHBoxLayout(controller);
	cosfactor = new QLabel(QString("cos(i*%1wt)").arg(frequency->value()),controller);
	removecos = new QPushButton("Remove", controller);
	cosLayout->addWidget(cosfactor);
	cosLayout->addWidget(removecos);
	cosLayout->addStretch();
	rightLayout->addLayout(cosLayout);
	QHBoxLayout *sinLayout = new QHBoxLayout(controller);
	sinfactor = new QLabel(QString("sin(i*%1wt)").arg(frequency->value()), controller);
	removesin = new QPushButton("Remove", controller);
	sinLayout->addWidget(sinfactor);
	sinLayout->addWidget(removesin);
	sinLayout->addStretch();
	rightLayout->addLayout(sinLayout);
	rightLayout->addStretch();
	calc = new QPushButton("Calc Fourier", controller);
	rightLayout->addWidget(calc);
	reverse = new QPushButton("Reverse Fourier", controller);
	rightLayout->addWidget(reverse);
	//controller->setLayout(rightLayout);
	//panel->setLayout(mainLayout);
	this->setCentralWidget(panel);
}

void Fourier::drawAxis(int x, int y, int scale, IplImage* image)
{
	for (int i = 0; i <= y / scale; i++){
		cvLine(image, cvPoint(5, i*scale + 5), cvPoint(825, i*scale + 5), CV_RGB(100, 100, 100));
	}
	for (int i = 0; i < x / scale; i++){
		cvLine(image, cvPoint(i*scale + 5, 5), cvPoint(i*scale + 5, 610), CV_RGB(100, 100, 100));
	}
	cvLine(image, cvPoint(5, y / 2 + 5), cvPoint(825, y / 2 + 5), CV_RGB(0, 0, 255), 2.5);
	cvLine(image, cvPoint(5, 5), cvPoint(5, 610), CV_RGB(0, 0, 255), 2.5);
}

void Fourier::calcDft()
{
	dft(src_y, factors,0,index);
	frequency->setDisabled(false);
	frequency->setMaximum(index / 2);
	frequency->setMinimum(0);
	copyFactors = factors * 1;
	displayFourier();
}

void Fourier::collect(int x, int y)
{
	src_x.at<float>(index) = x;
	src_y.at<float>(index) = y-150;
	index++;
}

void Fourier::reDft()
{
	dft(factors, dst, DFT_INVERSE);
	normalize(dst, dst, 5, 295, NORM_MINMAX);
	CvSize size;
	size.height = 615;
	size.width = 1230;
	IplImage *image = cvCreateImage(size, IPL_DEPTH_8U, 3);
	drawAxis(830, 600, 25, image);
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.5, 0.5, 0, 1);
	cvPutText(image, "0", cvPoint(5, 300), &font, CV_RGB(0, 0, 0));
	cvPutText(image, "T/4", cvPoint(205, 300), &font, CV_RGB(0, 0, 0));
	cvPutText(image, "T/2", cvPoint(405, 300), &font, CV_RGB(0, 0, 0));
	cvPutText(image, "3T/4", cvPoint(605, 300), &font, CV_RGB(0, 0, 0));
	cvPutText(image, "T", cvPoint(805, 300), &font, CV_RGB(0, 0, 0));
	for (int i = 1; i <index; i += 1){
		cvLine(image, cvPoint((i-1)*800/index+5, dst.at<float>(i-1)), 
			cvPoint(i*800/index+5, dst.at<float>(i)), CV_RGB(255, 0, 0));
	}
	Mat src = image;
	cv::cvtColor(src, src, CV_BGR2RGB);
	QImage img = QImage((const unsigned char*)(src.data), src.cols, src.rows, QImage::Format_RGB888);
	ol->setPixmap(QPixmap::fromImage(img)); 
}

void Fourier::displayFourier()
{
	CvSize size;
	size.height = 615;
	size.width = 1230;
	IplImage *image = cvCreateImage(size, IPL_DEPTH_8U, 3);
	drawAxis(830, 600, 25, image);
	for (int i = 0; i <= index / 2; i++){
		if (i == 0){
			freq.at<float>(i) = (factors.at<float>(i * 2)/index);
		}
		else if (i == index / 2){
			freq.at<float>(i) = (factors.at<float>(i * 2 - 1) / index);
		}
		else{
			freq.at<float>(i) = (sqrt(factors.at<float>(i * 2 - 1) / index * 2 * factors.at<float>(i * 2 - 1) / index * 2 + factors.at<float>(i * 2) / index * 2 * factors.at<float>(i * 2) / index * 2));
		}
	}
	normalize(freq, freq, 5, 295, NORM_MINMAX);
	for (int i = 1; i <= index/2; i++){
		cvLine(image, cvPoint((i - 1) * 1600 / index + 5, 300 - freq.at<float>(i - 1)), cvPoint(i * 1600 / index + 5, 300 - freq.at<float>(i)), CV_RGB(255, 0, 0));
	}  
	Mat src = image;
	cv::cvtColor(src, src, CV_BGR2RGB);
	QImage img = QImage((const unsigned char*)(src.data), src.cols, src.rows, QImage::Format_RGB888);
	fl->setPixmap(QPixmap::fromImage(img));
}

void Fourier::changeSinFactor()
{
	QPushButton *button = static_cast<QPushButton*>(sender());
	int num = frequency->value();
	if (button->text() == "Remove"){
		factors.at<float>(num * 2) = 0;
		reDft();
		updateFactors();
		button->setText("Restore");
	}
	else{
		factors.at<float>(num * 2) = copyFactors.at<float>(num * 2);
		reDft();
		updateFactors();
		button->setText("Remove");
	}
	
}

void Fourier::changeCosFactor()
{
	QPushButton *button = static_cast<QPushButton*>(sender());
	int num = frequency->value();
	if (button->text() == "Remove"){
		button->setText("Restore");
		factors.at<float>(num * 2 - 1) = 0;
		reDft();
		updateFactors();
	}
	else{
		button->setText("Remove");
		factors.at<float>(num * 2 - 1) = copyFactors.at<float>(num * 2 - 1);
		reDft();
		updateFactors();
	}
	
}

void Fourier::updateFactors()
{
	int num = frequency->value();
	if (num == 0){
		sinfactor->setText(QString("sin(i*%1wt) %2").arg(num).arg(0));
		cosfactor->setText(QString("cos(i*%1wt) %2").arg(num).arg(factors.at<float>(num * 2)));
		removesin->hide();
		if (factors.at<float>(num * 2) == 0)
			removecos->setText("Restore");
		else
			removecos->setText("Remove");
	}
	else if (num == index / 2){
		sinfactor->setText(QString("sin(i*%1wt) %2").arg(num).arg(0));
		cosfactor->setText(QString("cos(i*%1wt) %2").arg(num).arg(factors.at<float>(num * 2 - 1)));
		removesin->hide();
		if (factors.at<float>(num * 2 - 1) == 0)
			removecos->setText("Restore");
		else
			removecos->setText("Remove");
	}
	else{
		sinfactor->setText(QString("sin(i*%1wt) %2").arg(num).arg(factors.at<float>(num * 2)));
		cosfactor->setText(QString("cos(i*%1wt) %2").arg(num).arg(factors.at<float>(num * 2 - 1)));
		removesin->show();
		if (factors.at<float>(num * 2) == 0)
			removesin->setText("Restore");
		else
			removesin->setText("Remove");
		if (factors.at<float>(num * 2 - 1) == 0)
			removecos->setText("Restore");
		else
			removecos->setText("Remove");
	}
	update();
}