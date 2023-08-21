#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>
#include "RSA.h"
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_3_clicked();

    void on_pushButton_en_clicked();

    void on_pushButton_de_clicked();


    void on_comboBox_currentIndexChanged(int index);

private:
    Ui::MainWindow *ui;
    RSA* rsa[5] = {NULL, NULL, NULL, NULL, NULL};
    unsigned int index;
};
#endif // MAINWINDOW_H
