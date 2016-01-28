#include "brats.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	/*QApplication a(argc, argv);
	Brats w;
	w.show();
	return a.exec();*/
	Controller *controller =new Controller();
	controller->test();
}
