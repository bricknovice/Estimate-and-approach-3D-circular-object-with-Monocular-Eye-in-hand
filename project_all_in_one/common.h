/*
This code is intended for academic use only.
You are free to use and modify the code, at your own risk.

If you use this code, or find it useful, please refer to the paper:

Michele Fornaciari, Andrea Prati, Rita Cucchiara,
A fast and effective ellipse detector for embedded vision applications
Pattern Recognition, Volume 47, Issue 11, November 2014, Pages 3693-3708, ISSN 0031-3203,
http://dx.doi.org/10.1016/j.patcog.2014.05.012.
(http://www.sciencedirect.com/science/article/pii/S0031320314001976)


The comments in the code refer to the abovementioned paper.
If you need further details about the code or the algorithm, please contact me at:

michele.fornaciari@unimore.it

last update: 23/12/2014
*/

#pragma once
#include <opencv2\opencv.hpp>




typedef std::vector<cv::Point>	VP;
typedef std::vector< VP >	VVP;
typedef unsigned int uint;

#define _INFINITY 1024
#define M_PI 3.14159265358979323846

int inline sgn(float val) {
    return (0.f < val) - (val < 0.f);
};


bool inline isInf(float x)
{
	union
	{
		float f;
		int	  i;
	} u;

	u.f = x;
	u.i &= 0x7fffffff;
	return !(u.i ^ 0x7f800000);
};


float inline Slope(float x1, float y1, float x2, float y2)
{
	//reference slope
		float den = float(x2 - x1);
		float num = float(y2 - y1);
		if(den != 0)
		{
			return (num / den);
		}
		else
		{
			return ((num > 0) ? float(_INFINITY) : float(-_INFINITY));
		}
};

//void cvCanny2(	const void* srcarr, void* dstarr,
//				double low_thresh, double high_thresh,
//				void* dxarr, void* dyarr,
//                int aperture_size );
//
//void cvCanny3(	const void* srcarr, void* dstarr,
//				void* dxarr, void* dyarr,
//                int aperture_size );

void Canny2(	cv::InputArray image, cv::OutputArray _edges,
				cv::OutputArray _sobel_x, cv::OutputArray _sobel_y,
                double threshold1, double threshold2,
                int apertureSize, bool L2gradient );

void Canny3(	cv::InputArray image, cv::OutputArray _edges,
				cv::OutputArray _sobel_x, cv::OutputArray _sobel_y,
                int apertureSize, bool L2gradient );


float inline ed2(const cv::Point& A, const cv::Point& B)
{
	return float(((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y)));
}

float inline ed2f(const cv::Point2f& A, const cv::Point2f& B)
{
	return (B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y);
}


void Labeling(cv::Mat1b& image, std::vector<std::vector<cv::Point> >& segments, int iMinLength);
void LabelingRect(cv::Mat1b& image, VVP& segments, int iMinLength, std::vector<cv::Rect>& bboxes);
void Thinning(cv::Mat1b& imgMask, uchar byF=255, uchar byB=0);

bool SortBottomLeft2TopRight(const cv::Point& lhs, const cv::Point& rhs);
bool SortTopLeft2BottomRight(const cv::Point& lhs, const cv::Point& rhs);

bool SortBottomLeft2TopRight2f(const cv::Point2f& lhs, const cv::Point2f& rhs);

namespace YaedEllipse {
    struct Ellipse
    {
        float _xc;
        float _yc;
        float _a;
        float _b;
        float _rad;
        float _score;

        Ellipse() : _xc(0.f), _yc(0.f), _a(0.f), _b(0.f), _rad(0.f), _score(0.f) {};
        Ellipse(float xc, float yc, float a, float b, float rad, float score = 0.f) : _xc(xc), _yc(yc), _a(a), _b(b), _rad(rad), _score(score) {};
        Ellipse(const Ellipse& other) : _xc(other._xc), _yc(other._yc), _a(other._a), _b(other._b), _rad(other._rad), _score(other._score) {};

        void Draw(cv::Mat& img, const cv::Scalar& color, const int thickness)
        {
            ellipse(img, cv::Point(cvRound(_xc), cvRound(_yc)), cv::Size(cvRound(_a), cvRound(_b)), _rad * 180.0 / CV_PI, 0.0, 360.0, color, thickness);
        };

        void Draw(cv::Mat3b& img, const int thickness)
        {
            cv::Scalar color(0, cvFloor(255.f * _score), 0);
            ellipse(img, cv::Point(cvRound(_xc), cvRound(_yc)), cv::Size(cvRound(_a), cvRound(_b)), _rad * 180.0 / CV_PI, 0.0, 360.0, color, thickness);
        };

        bool operator<(const Ellipse& other) const
        {
            if (_score == other._score)
            {
                float lhs_e = _b / _a;
                float rhs_e = other._b / other._a;
                if (lhs_e == rhs_e)
                {
                    return false;
                }
                return lhs_e > rhs_e;
            }
            return _score > other._score;
        };
    };
}



float GetMinAnglePI(float alpha, float beta);

class Pos {
public:
    Pos(cv::Mat _euler) :eulerMat{ _euler }
    {

        cv::Mat angleVec = (cv::Mat_<double>(3, 1) << _euler.at<double>(3, 0), _euler.at<double>(4, 0), _euler.at<double>(5, 0));
        eulerAngle2rotationMatrix(angleVec, this->rotation);
        this->translation = (cv::Mat_<double>(3, 1) << _euler.at<double>(0, 0), _euler.at<double>(1, 0), _euler.at<double>(2, 0));
        affine = cv::Affine3d(this->rotation, this->translation);
        for (int i = 0; i < 6; i++) {
            eulerVec[i] = this->eulerMat.at<double>(i, 0);
        }
    }
    Pos(double _eulerVec[])
    {
        for (int i = 0; i < 6; i++) {
            eulerVec[i] = _eulerVec[i];
        }
        for (int i = 0; i < 6; i++) {
            this->eulerMat.at<double>(i, 0) = eulerVec[i];
        }
        cv::Mat angleVec = (cv::Mat_<double>(3, 1) << eulerMat.at<double>(3, 0), eulerMat.at<double>(4, 0), eulerMat.at<double>(5, 0));
        eulerAngle2rotationMatrix(angleVec, this->rotation);
        this->translation = (cv::Mat_<double>(3, 1) << eulerMat.at<double>(0, 0), eulerMat.at<double>(1, 0), eulerMat.at<double>(2, 0));
        affine = cv::Affine3d(this->rotation, this->translation);
    }
    Pos(cv::Mat _rotation, cv::Mat _translation) :rotation{ _rotation }, translation{ _translation }
    {
        affine = cv::Affine3d(this->rotation, this->translation);
        cv::Mat tmp;
        rotationMatrix2eulerAngle(this->rotation, tmp);
        eulerMat = (cv::Mat_<double>(6, 1) << this->translation.at<double>(0, 0), this->translation.at<double>(1, 0), this->translation.at<double>(2, 0), tmp.at<double>(0, 0), tmp.at<double>(1, 0), tmp.at<double>(2, 0));
        for (int i = 0; i < 6; i++) {
            eulerVec[i] = this->eulerMat.at<double>(i, 0);
        }
    }
    Pos(cv::Affine3d affine) :affine{ affine }
    {
        this->rotation = cv::Mat(affine.rotation());
        this->translation = cv::Mat(affine.translation()).reshape(0, 3);
        cv::Mat tmp;
        rotationMatrix2eulerAngle(this->rotation, tmp);
        eulerMat = (cv::Mat_<double>(6, 1) << this->translation.at<double>(0, 0), this->translation.at<double>(1, 0), this->translation.at<double>(2, 0), tmp.at<double>(0, 0), tmp.at<double>(1, 0), tmp.at<double>(2, 0));
        for (int i = 0; i < 6; i++) {
            eulerVec[i] = this->eulerMat.at<double>(i, 0);
        }
    }

    cv::Mat eulerMat;           //6*1
    cv::Mat rotation;           //3*3
    cv::Mat translation;        //3*1
    cv::Affine3d affine;        //4*4
    double eulerVec[6];     //1*6
private:
    void inline eulerAngle2rotationMatrix(cv::Mat angleVec, cv::Mat& rotMat) {

        angleVec.reshape(0, 1);
        for (int i = 0; i < 3; i++) {
            angleVec.at<double>(0, i) = (angleVec.at<double>(0, i) * M_PI) / 180;
        }

        std::vector<double> Xsc;
        std::vector<double> Ysc;
        std::vector<double> Zsc;

        Xsc.push_back(sinf(angleVec.at<double>(0, 0)));
        Xsc.push_back(cosf(angleVec.at<double>(0, 0)));
        Ysc.push_back(sinf(angleVec.at<double>(0, 1)));
        Ysc.push_back(cosf(angleVec.at<double>(0, 1)));
        Zsc.push_back(sinf(angleVec.at<double>(0, 2)));
        Zsc.push_back(cosf(angleVec.at<double>(0, 2)));

        double Rx[3][3] = { {      1,      0,      0},
                            {      0, Xsc[1], -Xsc[0]} ,
                            {      0, Xsc[0], Xsc[1]} };

        double Ry[3][3] = { { Ysc[1],      0, Ysc[0]},
                            {      0,      1,     0} ,
                            {-Ysc[0],      0, Ysc[1]} };

        double Rz[3][3] = { { Zsc[1],-Zsc[0],      0},
                            { Zsc[0], Zsc[1],      0} ,
                            {      0,      0,      1} };

        cv::Mat RxMat, RyMat, RzMat;

        RxMat = cv::Mat(3, 3, CV_64FC1, Rx);
        RyMat = cv::Mat(3, 3, CV_64FC1, Ry);
        RzMat = cv::Mat(3, 3, CV_64FC1, Rz);

        rotMat = RzMat * RyMat * RxMat;
    }
    void inline rotationMatrix2eulerAngle(cv::Mat R, cv::Mat& angleVec) {
        float sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

        bool singular = sy < 1e-6; // If

        double x, y, z;
        if (!singular)
        {
            x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
            y = std::atan2(-R.at<double>(2, 0), sy);
            z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
        }
        else
        {
            x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
            y = std::atan2(-R.at<double>(2, 0), sy);
            z = 0;
        }
        angleVec = (cv::Mat_<double>(3, 1) << (x * 180) / M_PI, (y * 180) / M_PI, (z * 180) / M_PI);
    }

};
