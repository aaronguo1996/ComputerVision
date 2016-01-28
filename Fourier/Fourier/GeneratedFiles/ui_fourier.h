/********************************************************************************
** Form generated from reading UI file 'fourier.ui'
**
** Created by: Qt User Interface Compiler version 5.5.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FOURIER_H
#define UI_FOURIER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_FourierClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *FourierClass)
    {
        if (FourierClass->objectName().isEmpty())
            FourierClass->setObjectName(QStringLiteral("FourierClass"));
        FourierClass->resize(600, 400);
        menuBar = new QMenuBar(FourierClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        FourierClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(FourierClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        FourierClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(FourierClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        FourierClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(FourierClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        FourierClass->setStatusBar(statusBar);

        retranslateUi(FourierClass);

        QMetaObject::connectSlotsByName(FourierClass);
    } // setupUi

    void retranslateUi(QMainWindow *FourierClass)
    {
        FourierClass->setWindowTitle(QApplication::translate("FourierClass", "Fourier", 0));
    } // retranslateUi

};

namespace Ui {
    class FourierClass: public Ui_FourierClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FOURIER_H
