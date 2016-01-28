/********************************************************************************
** Form generated from reading UI file 'brats.ui'
**
** Created by: Qt User Interface Compiler version 5.5.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BRATS_H
#define UI_BRATS_H

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

class Ui_BratsClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *BratsClass)
    {
        if (BratsClass->objectName().isEmpty())
            BratsClass->setObjectName(QStringLiteral("BratsClass"));
        BratsClass->resize(600, 400);
        menuBar = new QMenuBar(BratsClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        BratsClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(BratsClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        BratsClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(BratsClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        BratsClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(BratsClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        BratsClass->setStatusBar(statusBar);

        retranslateUi(BratsClass);

        QMetaObject::connectSlotsByName(BratsClass);
    } // setupUi

    void retranslateUi(QMainWindow *BratsClass)
    {
        BratsClass->setWindowTitle(QApplication::translate("BratsClass", "Brats", 0));
    } // retranslateUi

};

namespace Ui {
    class BratsClass: public Ui_BratsClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BRATS_H
