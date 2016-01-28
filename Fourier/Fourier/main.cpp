#include "fourier.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	Fourier w;
	w.show();
	return a.exec();
}
