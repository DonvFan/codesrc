#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <string>
#include <iostream>

std::string hexToStr(std::string hexStr)
{
    static char table[]={0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f};
    if(hexStr.size()%2)
        hexStr.push_back('0');
    std::string res;
    for(int i = 0; i<hexStr.size(); i+=2)
    {
        char c1 = hexStr[i], c2 = hexStr[i+1];
        if(isdigit(c1))
            c1 = table[c1 - '0'];
        else if(islower(c1))
            c1 = table[c1 - 'a'];
        else if(isupper(c1))
            c1 = table[c1 - 'A'];

        if(isdigit(c2))
            c2 = table[c2 - '0'];
        else if(islower(c2))
            c2 = table[c2 - 'a'];
        else if(isupper(c2))
            c2 = table[c2 - 'A'];

        unsigned char c = ((c1 << 4) | c2);
        res.push_back(c);
    }
    return res;
}


std::string stringToHex(std::string str)
{
    static char hex[]={'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};
    std::string res;

    for(int i = 0; i<str.size(); i++)
    {
        unsigned char temp = str[i];
        unsigned char index1 = temp >> 4, index2 = temp & 0x0F;
        res.push_back(hex[index1]);
        res.push_back(hex[index2]);
    }
    return res;
}
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QIcon icon("Icon.png");
    this->setWindowIcon(icon);
    this->setWindowTitle("RSA");
    index = 0;

}
MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_3_clicked()
{
    unsigned int ci = ui->comboBox->currentIndex();
    ui->pushButton_3->setDisabled(true);
    QString orgin = ui->pushButton_3->text();
    ui->pushButton_3->setText("初始化中");
    if(rsa[ci] == NULL)
    {
        std::string bit_num = ui->comboBox->currentText().toStdString().substr(4);
        unsigned int num = stoi(bit_num)/2;
        rsa[ci] = new RSA(num);
    }

    ui->textBrowser_d->setText(rsa[ci]->get_d().c_str());
    ui->textBrowser_n->setText(rsa[ci]->get_n().c_str());
    ui->textBrowser_e->setText(rsa[ci]->get_e().c_str());
    QMessageBox::information(NULL, "Info", "初始化完成！");
    ui->pushButton_3->setDisabled(false);
    ui->pushButton_3->setText(orgin);
}


void MainWindow::on_pushButton_en_clicked()
{
    QString message = ui->plainTextEdit_input->toPlainText();
    std::string m = stringToHex(message.toStdString());
    const BigInt bi(m);
    std::string en = rsa[index]->encrypt(bi).toHexStr();
    QString out(en.c_str());
    ui->textBrowser_output->setText(out);
}




void MainWindow::on_pushButton_de_clicked()
{
    QString message = ui->plainTextEdit_input->toPlainText();
    const BigInt bi(message.toStdString());
    std::string de = rsa[index]->decrypt(bi).toHexStr();
    std::string s = hexToStr(de);
    QString out(s.c_str());
    ui->textBrowser_output->setText(out);
}





void MainWindow::on_comboBox_currentIndexChanged(int i)
{
    index = i;
}

