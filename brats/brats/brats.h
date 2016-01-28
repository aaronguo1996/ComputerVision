#ifndef BRATS_H
#define BRATS_H

#include <QtWidgets/QMainWindow>
#include <opencv.hpp>
#include "ui_brats.h"
#include "controller.h"
using namespace cv;

class Brats : public QMainWindow
{
	Q_OBJECT

public:
	Brats(QWidget *parent = 0);
	~Brats();

private:
	Ui::BratsClass ui;
	Controller *controller;
};

#endif // BRATS_H
