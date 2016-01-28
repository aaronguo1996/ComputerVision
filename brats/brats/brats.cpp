#include "brats.h"

Brats::Brats(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	controller = new Controller();
	controller->test();
}

Brats::~Brats()
{

}
