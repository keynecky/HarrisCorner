#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const double ALPHA = 0.04;
const double THRESHOLD = 10000000000;


void HarrisCorner(Mat img);
Mat FilterR(Mat& R, Mat& img, int wsize);
int VideoPlay();
Mat computeImage(Mat& ix, Mat& iy, int wsize, int para);


int main()
{
    VideoPlay();
    return 0;
}


void HarrisCorner(Mat img)
{
    Mat img_gray, result, filter_R;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);  //将彩色图像转化成灰度图像
    //imshow("gray", img_gray);
    Mat img_x = Mat::zeros(img_gray.size(), CV_64FC1);
    Mat img_y = Mat::zeros(img_gray.size(), CV_64FC1);
    Mat img_xx = Mat::zeros(img_gray.size(), CV_64FC1);
    Mat img_yy = Mat::zeros(img_gray.size(), CV_64FC1);
    Mat img_xy = Mat::zeros(img_gray.size(), CV_64FC1);
    Mat R = Mat::zeros(img_gray.size(), CV_64FC1);
    Mat img_det = Mat::zeros(img_gray.size(), CV_64FC1);
    Mat img_trace = Mat::zeros(img_gray.size(), CV_64FC1);

    Sobel(img_gray, img_x, CV_64FC1, 1, 0, 3);  //x方向梯度
    Sobel(img_gray, img_y, CV_64FC1, 0, 1, 3);  //y方向梯度

    img_xx = computeImage(img_x, img_y, 3, 1);
    img_yy = computeImage(img_x, img_y, 3, 2);
    img_xy = computeImage(img_x, img_y, 3, 4);
    R = computeImage(img_x, img_y, 3, 3);
    filter_R = FilterR(R, img, 10);
    
    imshow("R", filter_R);
    imwrite("./R.jpg", filter_R);
    imshow("processed image", img);
    imwrite("./processed image.jpg", img);
    waitKey(0);
}


Mat computeImage(Mat& ix, Mat& iy, int wsize, int para) {

    Mat I_xx, I_yy, I_xy, r;
    I_xx = Mat::zeros(ix.size(), CV_64FC1);
    I_yy = Mat::zeros(ix.size(), CV_64FC1);
    r = Mat::zeros(ix.size(), CV_64FC1);
    I_xy = Mat::zeros(ix.size(), CV_64FC1);


    for (int i = wsize / 2; i < (ix.rows - wsize / 2); i++)
        for (int j = wsize / 2; j < (ix.cols - wsize / 2); j++) {
            //compute A B C, A = Ix * Ix, B = Iy * Iy, C = Ix * Iy
            double A = 0;
            double B = 0;
            double C = 0;
            for (int ii = i - wsize / 2; ii <= (i + wsize / 2); ii++)
                for (int jj = j - wsize / 2; jj <= (j + wsize / 2); jj++) {
                    double xx = ix.at<double>(ii, jj);
                    double yy = iy.at<double>(ii, jj);
                    A += xx * xx;
                    B += yy * yy;
                    C += xx * yy;
                }
            double p = A + B;
            double q = A * B - C * C;
            //double delta = p * p - 4 * q;//A2+B2-AB+4C2

            I_xx.at<double>(i, j) = A;
            I_yy.at<double>(i, j) = B;
            I_xy.at<double>(i, j) = C;
            double rr = q - 0.06 * p * p;

            if (rr > THRESHOLD) {
                r.at<double>(i, j) = rr;
            }

        }
    switch (para) {
    case 1: return I_xx; break;
    case 2: return I_yy; break;
    case 3: return r; break;
    case 4:return I_xy; break;
    }
}


Mat FilterR(Mat& R, Mat& img, int wsize) {
    Mat result;
    result = Mat::zeros(R.size(), CV_64F);

    //find local maxima of R
    for (int i = wsize / 2; i < (R.rows - wsize / 2); i++)
        for (int j = wsize / 2; j < (R.cols - wsize / 2); j++) {
            double origin = R.at<double>(i, j);
            bool found = false;
            for (int ii = i - wsize / 2; ii <= (i + wsize / 2) && found == false; ii++)
                for (int jj = j - wsize / 2; jj <= (j + wsize / 2); jj++)
                    if (origin < R.at<double>(ii, jj)) {
                        origin = 0;
                        found = true;
                        break;
                    }
            if (origin == 0)
                result.at<double>(i, j) = 0;
            else
            {
                result.at<double>(i, j) = 255;
                circle(img, Point(j, i), 5, Scalar(0, 0, 255), 2, 8, 0);
            }  
        }

    return result;
}


int VideoPlay()
{
    Mat frame;
    cout << "Opening camera..." << endl;
    VideoCapture capture(0); // open the first camera
    if (!capture.isOpened())
    {
        cerr << "ERROR: Can't initialize camera capture" << endl;
        return 1;
    }

    size_t nFrames = 0;
    bool enableProcessing = false;
    for (;;)
    {
        capture >> frame; // read the next frame from camera
        if (frame.empty())
        {
            cerr << "ERROR: Can't grab camera frame." << endl;
            break;
        }
        nFrames++;
        if (!enableProcessing)
        {
            imshow("Frame", frame);
        }
        else
        {  // 处理当前帧
            imshow("image", frame);
            HarrisCorner(frame);
            break;
        }
        int key = waitKey(1);
        if (key == 27/*ESC*/)
            break;
        if (key == 32/*SPACE*/)
        {
            enableProcessing = !enableProcessing;
            cout << "Enable frame processing ('space' key): " << enableProcessing << endl;
        }
    }
    return 0;
}






