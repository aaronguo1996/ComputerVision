#ifndef FOURIER_H
#define FOURIER_H

#include <QtWidgets/QMainWindow>
#include <qwidget.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qpoint.h>
#include <qpainter.h>
#include <qpen.h>
#include <qvector.h>
#include <qevent.h>
#include <qpair.h>
#include <qslider.h>
#include <qdebug.h>
#include <opencv.hpp>
#include "ui_fourier.h"
using namespace cv;

class MyPainterWidget : public QWidget
{
	Q_OBJECT
public:
	MyPainterWidget(QWidget *parent = 0);
	~MyPainterWidget();

protected:
	void paintEvent(QPaintEvent *p);
	void mousePressEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);

private:
	QPoint start;
	QPoint end;
	bool isPressed;
	QVector<QPair<QPoint, QPoint>>lines;
	void drawAxis(int x, int y, int scale, IplImage* image);

signals:
	void sendPos(int, int);
};

class Fourier : public QMainWindow
{
	Q_OBJECT

public:
	Fourier(QWidget *parent = 0);
	~Fourier();

private:
	Ui::FourierClass ui;
	QWidget *panel;
	MyPainterWidget *input;
	QWidget *output;
	QWidget *fourier;
	QWidget *controller;

	QPushButton *function;
	QPushButton *calc;
	QPushButton *reverse;
	QPushButton *removecos;
	QPushButton *removesin;
	
	QLabel *sinfactor;
	QLabel *cosfactor;
	QLabel *factor;
	QLabel *inputLabel;
	QLabel *outputLabel;
	QLabel *fourLabel;
	QLabel *number;
	QLabel *ol;
	QLabel *fl;

	QSlider *frequency;

	Mat src_x;
	Mat src_y;
	Mat src;
	Mat dst;
	Mat factors;
	Mat copyFactors;
	Mat freq;
	int index;

	void init();
	void drawAxis(int x, int y, int scale, IplImage* image);
	void repairPoints();

private slots:
	void collect(int, int);
	void calcDft();
	void reDft();
	void displayFourier();
	void changeCosFactor();
	void changeSinFactor();
	//void displayFactor();
	void updateFactors();
};

#endif // FOURIER_H
