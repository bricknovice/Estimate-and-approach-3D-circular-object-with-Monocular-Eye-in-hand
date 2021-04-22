#define CERES_FOUND 1 

#pragma once
#include <windows.h>
#include "camera_calibration.h"
//#include "common.h"
#include "EllipseDetectorYaed.h"
#include <pylon/PylonIncludes.h>
#include <pylon/gige/PylonGigEIncludes.h>
#include <pylon/PylonGUI.h>
#include <stdafx.h>
#include <HRSDK.h>
#include <iostream>
#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <fstream>
#include <tinydir.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#define sind(x) (sin((x) * M_PI / 180))
#define cosd(x) (cos((x) * M_PI / 180))


using namespace Pylon;
using namespace Eigen;
using namespace cv;
using namespace std;
using namespace cv::sfm;
using namespace ceres;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

double lookat_point_model[3] = { 0,368,-370.489 }; //On [table].
double lookat_point_calib[3] = { 0,368,-170.545 }; //Above [table] below [base].
double chessboard_frame[6] = { -120.429,429.257,-370.489,0,0,0 };
int exitCode = 0;
const int grabFrames_amount_calib = 20;
const int grabFrames_amount_model = 50;
const bool force_calib = false;
const bool force_model = false;

struct ReprojectCost
{
    cv::Point2d observation;
    //member initialization list
    ReprojectCost(cv::Point2d& _observation)
        : observation(_observation)
    {
    }
    template <typename T>
    bool operator()(const T* const intrinsic, const T* const R_gripper2base, const T* const T_gripper2base, const T* const X, const T* const pos3d, T* residuals) const
    {   

        //intrinsicMat<T, 3, 3> = 
        //
        //fx 0 cx
        //0 fy cy
        //0 0 1
        //
        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> intrinsicMat;
        intrinsicMat(0, 0) = intrinsic[0];
        intrinsicMat(0, 1) = T(0);
        intrinsicMat(0, 2) = intrinsic[2];
        intrinsicMat(1, 0) = T(0);
        intrinsicMat(1, 1) = intrinsic[1];
        intrinsicMat(1, 2) = intrinsic[3];
        intrinsicMat(2, 0) = T(0);
        intrinsicMat(2, 1) = T(0);
        intrinsicMat(2, 2) = T(1);


        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R_gripper2baseMat =
            Eigen::Map <const Eigen::Matrix< T, 3, 3, Eigen::RowMajor> >(R_gripper2base);
        
        Eigen::Translation<T, 3>tr(T_gripper2base[0], T_gripper2base[1],T_gripper2base[2]);

        Eigen::Transform<T, 3, Eigen::Affine> gripper2base = tr * R_gripper2baseMat;

        Eigen::Matrix<T, 3, 1> uz;
        uz << T(0) , T(0) , T(1);
        Eigen::Matrix<T, 3, 1> uy;
        uy << T(0) , T(1) , T(0);
        Eigen::Matrix<T, 3, 1> ux;
        ux << T(1) , T(0) , T(0);
        const double kPi = 3.14159265358979323846;
        const T degrees_to_radians(kPi / 180.0);
        Eigen::AngleAxis<T> rollAngle(X[5]* degrees_to_radians, uz);
        Eigen::AngleAxis<T> yawAngle(X[4]* degrees_to_radians, uy);
        Eigen::AngleAxis<T> pitchAngle(X[3]* degrees_to_radians, ux);

 
        Eigen::Translation<T, 3>tr1(X[0], X[1], X[2]);

        Eigen::Transform<T, 3, Eigen::Affine> X_matrix = tr1 * rollAngle * yawAngle * pitchAngle;
        Eigen::Transform<T, 3, Eigen::Affine> base2cam = (gripper2base * X_matrix).inverse();

        Eigen::Matrix< T, 4, 4, Eigen::RowMajor> trans2mat = base2cam.matrix();
        Eigen::Matrix< T, 3, 4, Eigen::RowMajor> extrinsic = trans2mat.block<3, 4>(0, 0);

        Eigen::Matrix<T, 4, 1, Eigen::ColMajor> point3d;
        point3d(0, 0) = pos3d[0];
        point3d(1, 0) = pos3d[1];
        point3d(2, 0) = pos3d[2];
        point3d(3, 0) = T(1);

        //[3*1] = [3*3] * [3*4] * [4*1]
        Eigen::Matrix<T, 3, 1, Eigen::ColMajor> rst = intrinsicMat * extrinsic * point3d;

        const T u = rst(0, 0) / rst(2, 0);
        const T v = rst(1, 0) / rst(2, 0);

        residuals[0] = u - T(observation.x);
        residuals[1] = v - T(observation.y);

        return true;
    }
    
};


void __stdcall callBack(uint16_t cmd, uint16_t rlt, uint16_t* msg, int len) {

}

//Saving multiple homogenous coordinate matrix
void writeXml(FileStorage _fileStorage, string filename, vector<Mat> rvec, vector<Mat> tvec) {
    _fileStorage << "R_" + filename << "[";
    for (int i = 0; i < rvec.size(); i++) {
        _fileStorage << rvec[i];
    }
    _fileStorage << "]";
    // Storing T_gripper2base info as a array.
    _fileStorage << "T_" + filename << "[";
    for (int i = 0; i < tvec.size(); i++) {
        _fileStorage << tvec[i];
    }
    _fileStorage << "]";
}

//Saving single homogenous coordinate matrix
void writeXml(FileStorage _fileStorage, string filename, Mat rvec, Mat tvec) {
    _fileStorage << "R_" + filename << "[";

    _fileStorage << rvec;

    _fileStorage << "]";

    _fileStorage << "T_" + filename << "[";
    _fileStorage << tvec;
    _fileStorage << "]";
}

//Saving single matrix
void writeXml(FileStorage _fileStorage, string filename, Mat matrix) {
    _fileStorage << filename << "[";
    _fileStorage << matrix;
    _fileStorage << "]";
}

void writeXml(FileStorage _fileStorage, string filename, Pos pos) {
    _fileStorage << "R_" + filename << "[";

    _fileStorage << pos.rotation;

    _fileStorage << "]";

    _fileStorage << "T_" + filename << "[";
    _fileStorage << pos.translation;
    _fileStorage << "]";
}


void writeXml(FileStorage _fileStorage, string filename, vector<Pos> pos) {
    _fileStorage << "R_" + filename << "[";
    for (int i = 0; i < pos.size(); i++) {
        _fileStorage << pos[i].rotation;
    }
    _fileStorage << "]";
    // Storing T_gripper2base info as a array.
    _fileStorage << "T_" + filename << "[";
    for (int i = 0; i < pos.size(); i++) {
        _fileStorage << pos[i].translation;
    }
    _fileStorage << "]";
}


//Loading multiple homogenous coordinate matrix
void readXml(FileStorage _fileStorage, string filename, vector<Mat>& rvec, vector<Mat>& tvec) {
    FileNode fnode = _fileStorage["R_" + filename];
    for (FileNodeIterator it = fnode.begin(); it != fnode.end(); ) {
        Mat tmp;
        it >> tmp;
        rvec.push_back(tmp);
        tmp.release();
    }
    fnode = _fileStorage["T_" + filename];
    for (FileNodeIterator it = fnode.begin(); it != fnode.end(); ) {
        Mat tmp;
        it >> tmp;
        tvec.push_back(tmp);
        tmp.release();
    }
}

//Loading single homogenous coordinate matrix
void readXml(FileStorage _fileStorage, string filename, Mat& rvec, Mat& tvec) {
    FileNode fnode = _fileStorage["R_" + filename];
    for (FileNodeIterator it = fnode.begin(); it != fnode.end(); ) {
        Mat tmp;
        it >> tmp;
        rvec = tmp.clone();
    }
    fnode = _fileStorage["T_" + filename];
    for (FileNodeIterator it = fnode.begin(); it != fnode.end(); ) {
        Mat tmp;
        it >> tmp;
        tvec = tmp.clone();
    }
}


//Loading multiple homogenous coordinate matrix
void readXml(FileStorage _fileStorage, string filename, vector<Pos>& pos) {
    FileNode fnode = _fileStorage["R_" + filename];
    FileNode fnode2 = _fileStorage["T_" + filename];
    for (FileNodeIterator it = fnode.begin(), it2 = fnode2.begin(); it != fnode.end(), it2 != fnode2.end();) {
        Mat R_tmp;
        Mat T_tmp;
        it >> R_tmp;
        it2 >> T_tmp;
        pos.push_back(Pos(R_tmp, T_tmp));
        R_tmp.release();
        T_tmp.release();
        
    }
}

//Loading single homogenous coordinate matrix
void readXml(FileStorage _fileStorage, string filename, Pos pos) {
    FileNode fnode = _fileStorage["R_" + filename];
    FileNode fnode2 = _fileStorage["T_" + filename];
    for (FileNodeIterator it = fnode.begin(), it2 = fnode2.begin(); it != fnode.end();) {
        Mat R_tmp;
        Mat T_tmp;
        it >> R_tmp;
        it2 >> T_tmp;
        pos = Pos(R_tmp, T_tmp);
        R_tmp.release();
        T_tmp.release();
    }
}


//Loading single matrix
void readXml(FileStorage _fileStorage, string filename, Mat& matrix) {
    FileNode fnode = _fileStorage[filename];
    for (FileNodeIterator it = fnode.begin(); it != fnode.end(); ) {
        Mat tmp;
        it >> tmp;
        matrix = tmp.clone();
    }
}

void eulerAngle2rotationMatrix(Mat angleVec, Mat& rotMat) {

    angleVec.reshape(0, 1);
    for (int i = 0; i < 3; i++) {
        angleVec.at<double>(0, i) = (angleVec.at<double>(0, i) * M_PI) / 180;
    }

    vector<double> Xsc;
    vector<double> Ysc;
    vector<double> Zsc;

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

    Mat RxMat, RyMat, RzMat;

    RxMat = Mat(3, 3, CV_64FC1, Rx);
    RyMat = Mat(3, 3, CV_64FC1, Ry);
    RzMat = Mat(3, 3, CV_64FC1, Rz);

    rotMat = RzMat * RyMat * RxMat;
}


void rotationMatrix2eulerAngle(Mat R, Mat& angleVec) {
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
    angleVec = (Mat_<double>(3, 1) << (x * 180) / M_PI, (y * 180) / M_PI, (z * 180) / M_PI);
}



//Wait 'til manipulator stop
void wait(HROBOT device_id) {
    while (get_motion_state(device_id) != 1) {
    }
}

//look at "to" from "from"
void LookAt(cv::Vec3d from, cv::Vec3d to, cv::Vec3d up, Mat& m) {

    cv::Vec3d z = from - to;
    cv::Vec3d norm_z;
    cv::normalize(z, norm_z,1.0,0.0,NORM_L2);
    cv::Vec3d x = up.cross(z);
    cv::Vec3d norm_x;
    cv::normalize(x,norm_x , 1.0, 0.0, NORM_L2);
    Vec3d y = z.cross(x);
    cv::Vec3d norm_y;
    cv::normalize(y, norm_y, 1.0, 0.0, NORM_L2);
    m = (Mat_<double>(3, 3) << norm_x[0], norm_y[0], norm_z[0],
        norm_x[1], norm_y[1], norm_z[1],
        norm_x[2], norm_y[2], norm_z[2]
        );
   
}


void calib_circular_path_planning(HROBOT device_id,double *lookat_point,double** cart_coord, int grabFrames_amount, double radius, double added_z) {
    int num_point = grabFrames_amount / 2;
    double interval = 2 * radius / (num_point - 1);
    double init_pos[6];
    get_current_position(device_id, init_pos);
    //Planning x coordinate individually.
    double* x = new double[grabFrames_amount / 2];
    *x = *(init_pos)-radius;
    for (int i = 1; i < num_point; i++) {
        *(x + i) = *(x + i - 1) + interval;
    }

    double sol[2] = {};
    for (int i = 0; i < num_point; i++) {
        //Solving z coordinate[sol[0] and sol[1]]. 
        //double a = 1, b = -2 * (*(init_pos + 2)), c = pow(*(x + i) - *(init_pos), 2) - pow(radius, 2) + pow(*(init_pos + 2), 2);
        double a = 1, b = -2 * (*(init_pos + 1)), c = pow(*(x + i) - *(init_pos), 2) - pow(radius, 2) + pow(*(init_pos + 1), 2);
        sol[0] = (-1 * b + std::sqrt(pow(b, 2) - 4 * a * c)) / (2 * a);
        sol[1] = (-1 * b - std::sqrt(pow(b, 2) - 4 * a * c)) / (2 * a);
        //f(x) = y
        //Figuring out the rotation part [A,B,C] and save the result.
        for (int j = 0; j < 2; j++) {
            cv::Vec3d from(*(x + i), sol[j], *(init_pos + 2) + added_z);
            cv::Vec3d to(lookat_point[0], lookat_point[1], lookat_point[2]);
            cv::Vec3d up(0, 1, 0);
            Mat m(3,3,CV_64FC1);
            //Matrix3d m;
            LookAt(from, to, up, m);
            Mat angles;
            rotationMatrix2eulerAngle(m, angles);
            //Retriving result.
            if (j == 0) {
                *(*(cart_coord + i)) = *(x + i);
                *(*(cart_coord + i) + 1) = sol[j];
                *(*(cart_coord + i) + 2) = *(init_pos + 2) + added_z;
                *(*(cart_coord + i) + 3) = (angles.at<double>(0, 0));
                *(*(cart_coord + i) + 4) = (angles.at<double>(1, 0));
                *(*(cart_coord + i) + 5) = (angles.at<double>(2, 0));
            }
            else if (j == 1) {
                *(*(cart_coord + grabFrames_amount - 1 - i)) = *(x + i);
                *(*(cart_coord + grabFrames_amount - 1 - i) + 1) = sol[j];
                *(*(cart_coord + grabFrames_amount - 1 - i) + 2) = *(init_pos + 2) + added_z;
                *(*(cart_coord + grabFrames_amount - 1 - i) + 3) = (angles.at<double>(0, 0));
                *(*(cart_coord + grabFrames_amount - 1 - i) + 4) = (angles.at<double>(1, 0));
                *(*(cart_coord + grabFrames_amount - 1 - i) + 5) = (angles.at<double>(2, 0));
            }

        }
    }
}

void model_circular_path_planning(HROBOT device_id,double* lookat_point,double** cart_coord,int grabFrames_amount,double radius, double radius_lookat,double added_z) {
    int num_point = grabFrames_amount / 2;
    double interval = 2 * radius / (num_point - 1);
    double init_pos[6];
    get_current_position(device_id, init_pos);
    //Planning x coordinate individually.
    double* x = new double[grabFrames_amount / 2];
    *x = *(init_pos)-radius;
    for (int i = 1; i < num_point; i++) {
        *(x + i) = *(x + i - 1) + interval;
    }

    double* x_lookat = new double[grabFrames_amount / 2];
    double interval_lookat = 2 * radius_lookat / (num_point - 1);
    *x_lookat = *(lookat_point)-(radius_lookat);
    for (int i = 1; i < num_point; i++) {
        *(x_lookat + i) = *(x_lookat + i - 1) + interval_lookat;
    }

    double sol[2] = {};
    double sol_lookat[2] = {};
    for (int i = 0; i < num_point; i++) {
        double a = 1, b = -2 * (*(init_pos + 1)), c = pow(*(x + i) - *(init_pos), 2) - pow(radius, 2) + pow(*(init_pos + 1), 2);
        sol[0] = (-1 * b + std::sqrt(pow(b, 2) - 4 * a * c)) / (2 * a);
        sol[1] = (-1 * b - std::sqrt(pow(b, 2) - 4 * a * c)) / (2 * a);
        double a_lookat = 1, b_lookat = -2 * (*(lookat_point + 1)), c_lookat = pow(*(x_lookat + i) - *(lookat_point), 2) - pow(radius_lookat, 2) + pow(*(lookat_point + 1), 2);
        sol_lookat[0] = (-1 * b_lookat + std::sqrt(pow(b_lookat, 2) - 4 * a_lookat * c_lookat)) / (2 * a_lookat);
        sol_lookat[1] = (-1 * b_lookat - std::sqrt(pow(b_lookat, 2) - 4 * a_lookat * c_lookat)) / (2 * a_lookat);
        //f(x) = y
        //Figuring out the rotation part [A,B,C] and save the result.
        for (int j = 0; j < 2; j++) {
            cv::Vec3d from(*(x + i), sol[j], *(init_pos + 2) + added_z);
            //Vector3d from(*(x + i), sol[j] , *(init_pos + 2) + added_z);
            cv::Vec3d to(*(x_lookat + i), sol_lookat[j], *(lookat_point + 2) );
            //Vector3d to(lookat_point[0], lookat_point[1], lookat_point[2]);
            cv::Vec3d up(0, 1, 0);
            //Vector3d up(0, 1, 0);
            Mat m(3, 3, CV_64FC1);
            //Matrix3d m;
            LookAt(from, to, up, m);
            Mat angles;
            rotationMatrix2eulerAngle(m, angles);

            //Retriving result.
            if (j == 0) {
                *(*(cart_coord + i)) = *(x + i);
                *(*(cart_coord + i) + 1) = sol[j];
                *(*(cart_coord + i) + 2) = *(init_pos + 2) + added_z;
                *(*(cart_coord + i) + 3) = (angles.at<double>(0, 0));
                *(*(cart_coord + i) + 4) = (angles.at<double>(1, 0));
                *(*(cart_coord + i) + 5) = (angles.at<double>(2, 0));
            }
            else if (j == 1) {
                *(*(cart_coord + grabFrames_amount - 1 - i)) = *(x + i);
                *(*(cart_coord + grabFrames_amount - 1 - i) + 1) = sol[j];
                *(*(cart_coord + grabFrames_amount - 1 - i) + 2) = *(init_pos + 2) + added_z;
                *(*(cart_coord + grabFrames_amount - 1 - i) + 3) = (angles.at<double>(0, 0));
                *(*(cart_coord + grabFrames_amount - 1 - i) + 4) = (angles.at<double>(1, 0));
                *(*(cart_coord + grabFrames_amount - 1 - i) + 5) = (angles.at<double>(2, 0));
            }

        }
    }
}
void camera_grabbing(int device_id, double** cart_coord, int grabFrames_amount, vector<Mat>& R_transMat, vector<Mat>& T_transMat, string _filename) {
    try
    {
        int cnt = 0;

        CInstantCamera camera(CTlFactory::GetInstance().CreateDevice(CBaslerGigEDeviceInfo().SetIpAddress("140.120.182.137")));
        std::cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

        camera.MaxNumBuffer = 1;

        ptp_pos(device_id, 0, *(cart_coord));
        wait(device_id);
        camera.StartGrabbing(grabFrames_amount);

        CGrabResultPtr ptrGrabResult;

        while (camera.IsGrabbing())
        {
            ptp_pos(device_id, 0, *(cart_coord + cnt));
            wait(device_id);
            camera.RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);
            if (ptrGrabResult->GrabSucceeded())
            {
                //Saving image into png format.
                EPixelType pixelType = ptrGrabResult->GetPixelType();
                uint32_t width = ptrGrabResult->GetWidth();
                uint32_t height = ptrGrabResult->GetHeight();
                size_t paddingX = ptrGrabResult->GetPaddingX();
                EImageOrientation orientation = ImageOrientation_TopDown;
                size_t bufferSize = ptrGrabResult->GetImageSize();
                void* buffer = ptrGrabResult->GetBuffer();
                std::ostringstream ss;
                ss << std::setw(2) << std::setfill('0') << cnt;
                String_t filename =  String_t( _filename.c_str() ) + String_t("\\") + String_t(  ss.str().append(".jpg").c_str());
                std::cout << filename << endl;
                CImagePersistence::Save(
                    ImageFileFormat_Png,
                    filename,
                    buffer,
                    bufferSize,
                    pixelType,
                    width,
                    height,
                    paddingX,
                    orientation
                );
                

                double cur_pos[6] = {};
                get_current_position(device_id, cur_pos);
                Mat tmp = (Mat_<double>(1, 3) << cur_pos[3], cur_pos[4], cur_pos[5]);
                Mat rm,rm2;
                eulerAngle2rotationMatrix(tmp, rm);
                R_transMat.push_back(rm);
                //Displaying image.
                Pylon::DisplayImage(1, ptrGrabResult);
                T_transMat.push_back(Mat_<double>{ cur_pos[0], cur_pos[1], cur_pos[2] }.reshape(0, 3));
                tmp.release();
                rm.release();     
            }
            else
            {
                std::cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
            }
            cnt++;
        }
    }
    catch (const GenericException& e)
    {
        cerr << "An exception occurred." << endl
            << e.GetDescription() << endl;
        exitCode = 1;
    }
}

void calib_grab_motion(HROBOT device_id,double * lookat_point,int grabFrames_amount, double radius, double added_z,vector<Mat>& R_transMat, vector<Mat>& T_transMat,string filename) {

    PylonInitialize();
    double AxishomePoint[6] = { 0,0,0,0,-90,0 };
    double** cart_coord = new double* [grabFrames_amount];
    for (int i = 0; i < grabFrames_amount; i++) cart_coord[i] = new double[6];

    set_override_ratio(device_id, 20);

    ptp_axis(device_id, 0, AxishomePoint);
    wait(device_id); 


    //Grabbing [grabFrames_amount] images with circular path.
    //Outputting images and endeffector's transformation matrix.
    calib_circular_path_planning(device_id, lookat_point, cart_coord, grabFrames_amount, radius, added_z);
    camera_grabbing(device_id, cart_coord, grabFrames_amount, R_transMat, T_transMat, filename);
    
    
    //deallocate pointer
    for (int i = 0; i < 6; ++i) {
        delete[] cart_coord[i];
    }
    delete[] cart_coord;
    Pylon::PylonTerminate();
}

void model_grab_motion(HROBOT device_id, double* lookat_point, int grabFrames_amount, double radius, double added_z, vector<Mat>& R_transMat, vector<Mat>& T_transMat, string filename) {

    PylonInitialize();
    double AxishomePoint[6] = { 0,0,0,0,-90,0 };
    double** cart_coord = new double* [grabFrames_amount];
    for (int i = 0; i < grabFrames_amount; i++) cart_coord[i] = new double[6];
    //double radius_lookat = 135;
    double radius_lookat = 0;
    set_override_ratio(device_id, 20);

    ptp_axis(device_id, 0, AxishomePoint);
    wait(device_id);


    //Grabbing [grabFrames_amount] images with circular path.
    //Outputting images and endeffector's transformation matrix.
    model_circular_path_planning(device_id, lookat_point, cart_coord, grabFrames_amount, radius, radius_lookat, added_z);
    camera_grabbing(device_id, cart_coord, grabFrames_amount, R_transMat, T_transMat, filename);

    //deallocate pointer
    for (int i = 0; i < 6; ++i) {
        delete[] cart_coord[i];
    }
    delete[] cart_coord;
    Pylon::PylonTerminate();
}

bool cmp(string a, string b) {
    a.erase(a.begin(), a.begin()+7);
    a.erase(a.end()-4, a.end());
    b.erase(b.begin(), b.begin() + 7);
    b.erase(b.end() - 4, b.end());
    int _a = stoi(a), _b = stoi(b);

    return _a < _b;
    
}
void get_filenames(string dir_name, vector<string>& names)
{
    names.clear();
    tinydir_dir dir;
    tinydir_open(&dir, dir_name.c_str());

    while (dir.has_next)
    {
        tinydir_file file;
        tinydir_readfile(&dir, &file);
        if (!file.is_dir)
        {
            names.push_back(file.path);
        }
        tinydir_next(&dir);
    }
    //sort(names.begin(), names.end(), cmp);
    tinydir_close(&dir);
    
}
void atexit_handler() {
    Pylon::PylonTerminate();
    std::cout << "exiting..." << endl;
}

void extract_features_SIFT(
    vector<string>& image_names,
    vector<vector<KeyPoint>>& keyPoints,
    vector<Mat>& descriptors,
    vector<vector<Vec3b>>& kpts_color)
{
    keyPoints.clear();
    descriptors.clear();
    Mat image;

    //讀取影像，獲取影像特徵點，並保存
    Ptr<Feature2D> extractorAndDetector = cv::SIFT::create(0, 3, 0.04, 10, 1.3);

    int i = 0;
    for (auto it = image_names.begin(); it != image_names.end(); ++it)
    {
        image = imread(*it);
        std::cout << *it << endl;
        if (image.empty()) continue;

        std::cout << "Extracing features: " << *it << endl;

        vector<KeyPoint> key_points;
        Mat descriptor;

        //偶爾出現記憶體分配失敗的錯誤
        extractorAndDetector->detectAndCompute(image, noArray(), key_points, descriptor);

        //特徵點過少，則排除該影像
        if (key_points.size() <= 10) {
            std::cout << "Image[" << i << "]: " << key_points.size() << "  feature number is too less" << endl;
            i++;
            continue;
        }
        keyPoints.push_back(key_points);
        descriptors.push_back(descriptor);

        vector<Vec3b> colors(key_points.size());
        for (int i = 0; i < key_points.size(); ++i)
        {
            Point2f& p = key_points[i].pt;
            colors[i] = image.at<Vec3b>(p.y, p.x);
        }
        kpts_color.push_back(colors);


    }
}

//see more info in :https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
//https://docs.opencv.org/3.0-beta/doc/tutorials/features2d/akaze_matching/akaze_matching.html
//https://blog.csdn.net/weixin_44072651/article/details/89262277
void match_features_KNN(Mat& query, Mat& train, vector<DMatch>& matches)
{
    /*
    DMatch.distance - Distance between descriptors.The lower, the better it is.
    DMatch.trainIdx - Index of the descriptor in train descriptors
    DMatch.queryIdx - Index of the descriptor in query descriptors
    DMatch.imgIdx - Index of the train image.
    */

    vector<vector<DMatch>> knn_matches;
    BFMatcher matcher(NORM_L2);
    //FlannBasedMatcher mathcer(NORM_L2);
    //取得query與train之間相匹配的點於knn_matches，並且有兩個候選人
    matcher.knnMatch(query, train, knn_matches, 2); //k=2, Which make it a 2-nn matches.

    //若兩個候選人的跟keypoint的距離差太多，我們就捨棄，這就是ratio test。
    //獲取滿足Ratio Test的最小匹配的距離
    float min_dist = FLT_MAX;
    for (int r = 0; r < knn_matches.size(); ++r)
    {
        //Ratio Test
        if (knn_matches[r][0].distance > 0.4 * knn_matches[r][1].distance)
            continue;

        float dist = knn_matches[r][0].distance;
        //std::cout << "dist= " << dist << endl;
        if (dist < min_dist) min_dist = dist;
    }

    //std::cout << "|| min_dist= " << min_dist << endl;

    matches.clear();
    for (size_t r = 0; r < knn_matches.size(); ++r)
    {
        //排除不滿足Ratio Test的點和匹配距離過大的點
        if(
            knn_matches[r][0].distance > 0.4 * knn_matches[r][1].distance ||
            knn_matches[r][0].distance > 3.5 * max(min_dist, 10.0f)
        )
            continue;

        //保存匹配點
        matches.push_back(knn_matches[r][0]);
    }
}

void match_features_KNN(vector<Mat>& descriptors, vector<vector<DMatch>>& matched_kpts)
{
    matched_kpts.clear();
    // n個影像，兩兩順序有 n-1 對匹配
    // 1與2匹配，2與3匹配，3與4匹配，以此類推
    for (int i = 0; i < descriptors.size() - 1; ++i)
    {
        std::cout << "Matching images " << i << " - " << i + 1 << endl;
        vector<DMatch> matches;
        match_features_KNN(descriptors[i], descriptors[i + 1], matches);
        matched_kpts.push_back(matches);
    }
}

void get_matched_points(
    vector<KeyPoint>& p1,
    vector<KeyPoint>& p2,
    vector<DMatch> matches,
    vector<Point2f>& out_p1,
    vector<Point2f>& out_p2
)
{
    out_p1.clear();
    out_p2.clear();
    for (int i = 0; i < matches.size(); ++i)
    {
        out_p1.push_back(p1[matches[i].queryIdx].pt);
        out_p2.push_back(p2[matches[i].trainIdx].pt);
    }
}

void get_matched_colors(
    vector<Vec3b>& c1,
    vector<Vec3b>& c2,
    vector<DMatch> matches,
    vector<Vec3b>& out_c1,
    vector<Vec3b>& out_c2
)
{
    out_c1.clear();
    out_c2.clear();
    for (int i = 0; i < matches.size(); ++i)
    {
        out_c1.push_back(c1[matches[i].queryIdx]);
        out_c2.push_back(c2[matches[i].trainIdx]);
    }
}

void fusion_structure(
    vector<DMatch>& matches,
    vector<int>& struct_indices,
    vector<int>& next_struct_indices,
    vector<Point3d>& structure,
    vector<Point3d>& next_structure,
    vector<Vec3b>& colors,
    vector<Vec3b>& next_colors
)
{
    for (int i = 0; i < matches.size(); ++i)
    {
        int query_idx = matches[i].queryIdx;
        int train_idx = matches[i].trainIdx;

        int struct_idx = struct_indices[query_idx];
        if (struct_idx >= 0) //若該點在空間中已經存在，則這對匹配點對應的空間點應該是同一個，索引要相同
        {
            next_struct_indices[train_idx] = struct_idx;
            continue;
        }

        //若該點在空間中尚不存在，將該點加入到結構中，且這對匹配點的空間點索引都為新加入的點的索引
        structure.push_back(next_structure[i]);
        colors.push_back(next_colors[i]);
        struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
    }
}

void cust_reconstruct(Mat& intrinsicParamMat, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3d>& structure)
{
    //兩個相機的投影矩陣[R T]，triangulatePoints只支持float型
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);

    R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
    T1.convertTo(proj1.col(3), CV_32FC1);

    R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T2.convertTo(proj2.col(3), CV_32FC1);


    Mat fK;
    intrinsicParamMat.convertTo(fK, CV_32FC1);
    proj1 = fK * proj1;
    proj2 = fK * proj2;

    //三角測量
    Mat s;
    triangulatePoints(proj1, proj2, p1, p2, s);	//輸出為 S: 齊次座標(X,Y,Z,W)，其中 W 表示座標軸的遠近參數


    structure.clear();
    structure.reserve(s.cols);
    for (int i = 0; i < s.cols; ++i)
    {
        Mat_<float> col = s.col(i);
        col /= col(3);	//齊次坐標，(x,y,z)需要除以最後一個元素 w 才是真正的坐標值 = (x/w, y/w, z/w)
        structure.push_back(Point3f(col(0), col(1), col(2)));
    }

    //std::cout << "structure: " << structure << endl;
    //system("pause");
}


void maskout_points(vector<Point2f>& p1, Mat& mask)
{
    vector<Point2f> p1_copy = p1;
    p1.clear();

    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i) > 0)
            p1.push_back(p1_copy[i]);
    }
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
    vector<Vec3b> p1_copy = p1;
    p1.clear();

    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i) > 0)
            p1.push_back(p1_copy[i]);
    }
}

void leaveBetterFeature(vector<vector<DMatch>>& matches_for_all, int Remaining_Feature_Point_Number) {

    for (int j = 0; j < matches_for_all.size(); j++) {

        /* 根據 distance 大小，由小到大排序過濾完的特徵點 */
        sort(matches_for_all[j].begin(), matches_for_all[j].end());

        /*
        std::cout << "排序完" << endl;
        for (int i = 0; i < matches_for_all[j].size(); i++)
        {
            std::cout << matches_for_all[j][i].distance << endl;
        }
        */

        int length = matches_for_all[j].size();
        if (Remaining_Feature_Point_Number < length) {
            for (int i = Remaining_Feature_Point_Number; i < length; i++)
                matches_for_all[j].pop_back();
        }


        /*
        std::cout << "前二十名" << endl;
        for (int i = 0; i < matches_for_all[j].size(); i++)
        {
            std::cout << matches_for_all[j][i].distance << endl;
        }
        */
    }

}

void showPhotoAlbumMatchingCondition(vector<string> image_names, vector<vector<KeyPoint>> key_points_for_all, vector<vector<DMatch>> matches_for_all, bool whetherShowPhoto) {

    Mat image, resultImg;
    vector<Mat> image_album;

    for (auto it = image_names.begin(); it != image_names.end(); ++it)
    {
        image = imread(*it);
        if (image.empty()) continue;

        image_album.push_back(image.clone());
    }


    for (int i = 0; i < image_album.size() - 1; i++) {

        drawMatches(image_album.at(i), key_points_for_all.at(i), image_album.at(i + 1), key_points_for_all.at(i + 1), matches_for_all.at(i), resultImg, Scalar::all(-1),
            Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
        );
 
        if (whetherShowPhoto) {
            namedWindow("Matching Photos", WINDOW_AUTOSIZE);
            imshow("Matching Photos", resultImg);
            waitKey(0);
        }
    }
}

void getbase2cam(Mat R_gripper2base, Mat T_gripper2base, Mat R_cam2gripper, Mat T_cam2gripper, Mat& R_base2cam, Mat& T_base2cam, cv::Affine3d& path) {
    cv::Affine3d bTg(R_gripper2base, T_gripper2base);
    cv::Affine3d gTc(R_cam2gripper, T_cam2gripper);
    cv::Affine3d cTb = (bTg * gTc).inv();
    R_base2cam = Mat(cTb.rotation());
    T_base2cam = Mat(cTb.translation().reshape<3,1>());
    cv::Affine3d bTc = (bTg * gTc);
    path = bTc;
}
void getbase2cam(Mat R_gripper2base, Mat T_gripper2base, Mat R_cam2gripper, Mat T_cam2gripper, Mat& R_base2cam, Mat& T_base2cam) {
    cv::Affine3d bTg(R_gripper2base, T_gripper2base);
    cv::Affine3d gTc(R_cam2gripper, T_cam2gripper);
    cv::Affine3d cTb = (bTg * gTc).inv();
    R_base2cam = Mat(cTb.rotation());
    T_base2cam = Mat(cTb.translation().reshape<3, 1>());
}


void bundle_adjustment(
    Mat& cMatrix,                                   //[3*3]
    vector<Mat>& R_gripper2base,                    //[n*3*3]
    vector<Mat>& T_gripper2base,                    //[n*3*1]
    Mat& R_cam2gripper,                               //[3*3]
    Mat& T_cam2gripper,                               //[3*1]
    vector<vector<int>>& correspond_struct_idx,
    vector<vector<KeyPoint>>& key_points_for_all,
    vector<Point3d>& structure
)
{
    vector<Mat> gripper2base;
    ceres::Problem problem;
   
    

    //Set intrinsic parameter
    //rotation matrix (3*3) to [fx, fy, cx, cy](4*1).
    Mat cMatrixparameter = (Mat_<double>(4, 1) << cMatrix.at<double>(0, 0), cMatrix.at<double>(1, 1), cMatrix.at<double>(0, 2), cMatrix.at<double>(1, 2));
    problem.AddParameterBlock(cMatrixparameter.ptr<double>(), 4);
    problem.SetParameterBlockConstant(cMatrixparameter.ptr<double>());

    //Set gripper2base
    //R_gripper2base(3*3) and T_gripper2base(3*1).
    for (size_t i = 0; i < R_gripper2base.size(); ++i) {
        problem.AddParameterBlock(R_gripper2base[i].ptr<double>(), 9);
        problem.SetParameterBlockConstant(R_gripper2base[i].ptr<double>());

        problem.AddParameterBlock(T_gripper2base[i].ptr<double>(), 3);
        problem.SetParameterBlockConstant(T_gripper2base[i].ptr<double>());
    }

    //Set cam2gripper
    //R_cam2gripper(3*3) and T_cam2gripper(3*1) to angle representation(x,y,z,a,b,c)(6*1).
    Mat r;
    rotationMatrix2eulerAngle(R_cam2gripper,r);
    Mat cam2gripper = (Mat_<double>(1, 6) << T_cam2gripper.at<double>(0, 0), T_cam2gripper.at<double>(1, 0), T_cam2gripper.at<double>(2, 0), r.at<double>(0, 0), r.at<double>(1, 0), r.at<double>(2, 0));
    problem.AddParameterBlock(cam2gripper.ptr<double>(), 6);
    // load points
    ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
    for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
    {
        vector<int>& point3d_ids = correspond_struct_idx[img_idx];
        vector<KeyPoint>& key_points = key_points_for_all[img_idx];
        for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
        {
            int point3d_id = point3d_ids[point_idx];
            if (point3d_id < 0)
                continue;
            Point2d observed = key_points[point_idx].pt;
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 9, 3, 6, 3>(new ReprojectCost(observed));
            problem.AddResidualBlock(
                cost_function,
                loss_function,
                cMatrixparameter.ptr<double>(),			// 4*1 = 4, constant
                R_gripper2base[img_idx].ptr<double>(),	// 3*3 = 9, constant
                T_gripper2base[img_idx].ptr<double>(),	// 3*1 = 3, constant
                cam2gripper.ptr<double>(),              // 1*6 = 6, variable
                &(structure[point3d_id].x)				// Point in 3D space, variable
            );
        }
    }
    // Solve BA
    ceres::Solver::Options ceres_config_options;
    //最大迭代次數，及最小精確容忍度。
    ceres_config_options.max_num_iterations = 10;
    ceres_config_options.function_tolerance = 1e-6;
    //預設情況下，Minimizer（優化器）進度會記錄到stderr，具體取決於vlog級別。 如果此標誌設定為true，並且Solver::Option:: logging_type不是SILENT，則記錄輸出將傳送到stdout（在cmd打印出資訊）。
    ceres_config_options.minimizer_progress_to_stdout = true;
    //每次迭代都列印資訊，另一個可選的為SILENT。
    ceres_config_options.logging_type = PER_MINIMIZER_ITERATION;
    //Ceres用於評估Jacobian的執行緒數，越多優化速度越快。
    ceres_config_options.num_threads = 500;
    //如果為true，則在每個迭代結束時更新引數，否則在優化終止時才更新引數。
    ceres_config_options.update_state_every_iteration = true;
    //ceres_config_options.max_num_consecutive_invalid_steps=100;
    ceres_config_options.dense_linear_algebra_library_type = LAPACK;
    ceres_config_options.preconditioner_type = ceres::JACOBI;
    ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
    //ceres_config_options.linear_solver_type = ceres::DENSE_QR;
    ceres_config_options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_config_options, &problem, &summary);
    if (!summary.IsSolutionUsable())
    {
        std::cout << "Bundle Adjustment failed." << std::endl;
    }
    else
    {
        //Need to update cam2gripper(by return it to R_cam2gripper and T_cam2gripper)
        Mat R_cam2gripper_vec = (Mat_<double>(1,3)<< cam2gripper.at<double>(0,3), cam2gripper.at<double>(0, 4), cam2gripper.at<double>(0, 5));
        eulerAngle2rotationMatrix(R_cam2gripper_vec, R_cam2gripper);
        T_cam2gripper.at<double>(0, 0) = cam2gripper.at<double>(0, 0);
        T_cam2gripper.at<double>(1, 0) = cam2gripper.at<double>(0, 1);
        T_cam2gripper.at<double>(2, 0) = cam2gripper.at<double>(0, 2);
        // Display statistics about the minimization
        std::cout << std::endl
            << "Bundle Adjustment statistics (approximated RMSE):\n"
            //<< " #views: " << extrinsics_rot.size() << "\n"
            << " #residuals: " << summary.num_residuals << "\n"
            << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
            << " Time (s): " << summary.total_time_in_seconds << "\n"
            //<< " Intrinsic: " << intrinsic << "\n"
            << std::endl;
    }
}

void getimage2world(Mat _cMatrix, Mat R_cam2gripper,Mat T_cam2gripper,Pos pos ,Point3d pos2d, Point3d& pos3d) {
    Mat inv_cMatrix;
    Mat ext = (Mat_<double>(1, 3) << 0, 0, 1);
    vconcat(cMatrix.inv(), ext, inv_cMatrix);
    Mat matrix = Mat((cv::Affine3d(R_cam2gripper, T_cam2gripper) * pos.affine).matrix).rowRange(0,3) * inv_cMatrix * Mat(pos2d).reshape(1, 3); //[3*4] * [4*3] * [3*1]
    pos3d = Point3d(matrix);
}

double cosine_similarity(Mat A,Mat B)
{
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (unsigned int i = 0u; i < 3; ++i) {
        dot += A.at<double>(i,0) * B.at<double>(i, 0);
        denom_a += A.at<double>(i, 0) * A.at<double>(i, 0);
        denom_b += B.at<double>(i, 0) * B.at<double>(i, 0);
    }
    return dot / (std::sqrt(denom_a) * std::sqrt(denom_b));
}

void get_RMSE(Mat R_cam2gripper, Mat T_cam2gripper) {
    Mat R_pos1 = (Mat_<double>(1, 3) << -0.583, -16.423, 0);
    Mat T_pos1 = (Mat_<double>(1, 3) << -109.950, 431.824, 55.793);
    Mat R_pos2 = (Mat_<double>(1, 3) << 8.350, 16.877, 2.874);
    Mat T_pos2 = (Mat_<double>(1, 3) << 121.463, 340.049, 55.793);
    Mat R_base2camtest1, R_base2camtest2;
    Mat T_base2camtest1, T_base2camtest2;
    eulerAngle2rotationMatrix(R_pos1, R_pos1);
    eulerAngle2rotationMatrix(R_pos2, R_pos2);
    vector<Mat> R_g2b = { R_pos1,R_pos2 };
    vector<Mat> T_g2b = { T_pos1,T_pos2 };
    getbase2cam(R_pos1, T_pos1, R_cam2gripper, T_cam2gripper, R_base2camtest1, T_base2camtest1);
    getbase2cam(R_pos2, T_pos2, R_cam2gripper, T_cam2gripper, R_base2camtest2, T_base2camtest2);
    vector<Mat> R_ext1, ext2;
    vector<cv::Point2f> _p1, _p2;
    _p1.push_back({ 549,331 });
    _p2.push_back({ 381,93 });

    vector<cv::Point3d> structure_test;
    cv::Point3d gt = { -98.771,480.321,-257.421 };
    cust_reconstruct(cMatrix, R_base2camtest1, T_base2camtest1, R_base2camtest2, T_base2camtest2, _p1, _p2, structure_test);

    for (int j = 0; j < structure_test.size(); j++) {
        std::cout << structure_test << endl;
        cout << "RMSE: " << std::sqrt((pow(structure_test[j].x - gt.x, 2) + pow(structure_test[j].y - gt.y, 2) + pow(structure_test[j].z - gt.z, 2) / 3)) << endl;
    }

    vector<Point3d> structure;
    vector<Vec3b> colors;
    vector<cv::Affine3d> cam2base;

    Mat image_coord = (Mat_<double>(3, 1) << _p1[0].x, _p1[0].y, 1);    //3*1
    Mat cam_coord = cMatrix.inv() * image_coord;                        //3*1
    vconcat(cam_coord, cv::Mat::ones(1, 1, CV_64FC1), cam_coord);       //4*1
    Mat world_coord = Mat(cv::Affine3d(R_base2camtest1, T_base2camtest1).inv().matrix * cam_coord).rowRange(0,3) ; //3*1
    cam2base.push_back(cv::Affine3d(R_base2camtest1, T_base2camtest1).inv());
    structure.push_back(Point3d(world_coord));
    colors.push_back(cv::Vec3b(255, 255, 255));
}

void ellipsesDetection(Mat R_cam2gripper, Mat T_cam2gripper) {

    FileStorage cp("campos.xml", FileStorage::READ);
    vector<Pos> pos;
    

    if (!cp.isOpened()) {
        
        FileStorage cp("campos.xml", FileStorage::WRITE);
        
        
        
        double pos1[6] = { -476.650, 19.139, -197.223, 24.706, 0.001, -0.003 };
        double pos2[6] = { -552.570, 63.576, -199.586, 22.832, -14.481, -0.004 };
        double pos3[6] = { -432.302, 166.327, -199.586, -1.409, 21.017, 7.131 };
       /* double pos1[6] = { 153.227,471.727,31.38,-5.880,16.867,0 };
        double pos2[6] = { 2.109,317.890,9.58,17.725,-10.480,0 };
        double pos3[6] = { 104.984,535.990,-72.819,-13.927,0,0 };
        double pos4[6] = { -87.104,474.137,29.056,8.572,-21.100,-4.313 };*/
        pos.push_back(Pos(pos1).affine * cv::Affine3d(R_cam2gripper, T_cam2gripper));
        pos.push_back(Pos(pos2).affine * cv::Affine3d(R_cam2gripper, T_cam2gripper));
        pos.push_back(Pos(pos3).affine * cv::Affine3d(R_cam2gripper, T_cam2gripper));
        //pos.push_back(Pos(pos4).affine * cv::Affine3d(R_cam2gripper, T_cam2gripper));
        writeXml(cp,"campos",pos);
        //HROBOT device_id = Connect("140.120.182.134", 1, callBack);
        //vector<Mat>R, T;
        //camera_grabbing(device_id, pos1, 2,R,T,);
    }
    else {
        readXml(cp, "campos", pos);
    }
    


    string sWorkingDir = "C:\\Users\\eug19\\workspace\\\project_all_in_one\\\project_all_in_one\\dataset\\"; 
    string out_folder = "C:\\Users\\eug19\\workspace\\project_all_in_one\\";

    vector<string> names;
    vector<float> prs;
    vector<float> res;
    vector<float> fms;
    vector<double> tms;

    glob(sWorkingDir + "images\\" + "*.*", names);
    int cnt = 0;
    vector<cv::Affine3d> cam2basetest;
    vector<cv::Affine3d> gripper2basetest;
    vector<vector<cv::Affine3d>> circle2basetest;
    circle2basetest.resize(names.size());
    vector<vector<cv::Point3d>> circlecents;
    circlecents.resize(names.size());


    for (const auto& image_name : names)
    {
        string name_ext = image_name.substr(image_name.find_last_of("\\") + 1);
        string name = name_ext.substr(0, name_ext.find_last_of("."));

        Mat3b image = imread(image_name);

        Size sz = image.size();

        // Convert to grayscale
        Mat1b gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);

        // Parameters Settings (Sect. 4.2)
        int		iThLength = 2;
        //float	fThObb = 1.0f;          //Default value is 3.0f.
        float	fThObb = 3.0f;
        float	fThPos = 1.0f;
        float	fTaoCenters = 0.05f;
        int 	iNs = 16;
        float	fMaxCenterDistance = sqrt(float(sz.width * sz.width + sz.height * sz.height)) * fTaoCenters;
        //float	fThScoreScore = 0.60f;  //Default value is 0.72f;.
        float	fThScoreScore = 0.72f;  //Default value is 0.72f;.

        // Other constant parameters settings. 

        // Gaussian filter parameters, in pre-processing
        Size	szPreProcessingGaussKernelSize = Size(5, 5);
        double	dPreProcessingGaussSigma = 1.0;

        float	fDistanceToEllipseContour = 0.1f;	// (Sect. 3.3.1 - Validation)
        float	fMinReliability = 0.5;	// Const parameters to discard bad ellipses

        CEllipseDetectorYaed yaed;
        yaed.SetParameters(szPreProcessingGaussKernelSize,
            dPreProcessingGaussSigma,
            fThPos,
            fMaxCenterDistance,
            iThLength,
            fThObb,
            fDistanceToEllipseContour,
            fThScoreScore,
            fMinReliability,
            iNs
        );

        // Detect
        vector<YaedEllipse::Ellipse> ellsYaed;
        Mat1b gray_clone = gray.clone();
        yaed.Detect(gray_clone, ellsYaed);
        ellsYaed.resize(1);


        Mat3b resultImage = image.clone();

        yaed.DrawDetectedEllipses(resultImage, ellsYaed);
        cv::namedWindow("Yaed", WINDOW_NORMAL);
        cv:imshow("Yaed", resultImage);
        waitKey();


        float u = ellsYaed[0]._xc;
        float v = ellsYaed[0]._yc;
        float a = ellsYaed[0]._a;
        float b = ellsYaed[0]._b;
        float rad = ellsYaed[0]._rad;


        double A = pow(a, 2) * pow(sin(rad), 2) + pow(b, 2) * pow(cos(rad), 2);
        double B = 2 * (pow(b, 2) - pow(a, 2)) * sin(rad) * cos(rad);
        double C = pow(a, 2) * pow(cos(rad), 2) + pow(b, 2) * pow(sin(rad),2);
        double D = -(2 * A * u + B * v);
        double E = -(B * u + 2 * C * v);
        double F = A * pow(u, 2) + B * u * v + C * pow(v, 2) - pow(a, 2) * pow(b, 2);

        Mat EllipseC = (Mat_<double>(3, 3) << A, B / 2, D / 2, B / 2, C, E / 2, D / 2, E / 2, F);

        Mat EllipseQ = cMatrix.t() * EllipseC * cMatrix;
        cout << "EllipseC:" << endl;
        cout << EllipseC << endl;
        cout << "EllipseQ:" << endl;
        cout << EllipseQ << endl;



        //v1,v2,v3分別為x',y',z'
        Mat eigenval, eigenvec;
        cv::eigen(EllipseQ, eigenval, eigenvec);
        Mat unoblique2cam = eigenvec.clone();
        eigenvec.row(0).copyTo(unoblique2cam.row(1));
        eigenvec.row(1).copyTo(unoblique2cam.row(0));
        unoblique2cam = unoblique2cam.t();


        cout << "Eigenval:" << endl;
        cout << eigenval << endl;
        cout << "Eigenvec:" << endl;
        cout << eigenvec << endl;
       

        double l1 = eigenval.at<double>(0, 0);
        double l2 = eigenval.at<double>(1, 0);
        double l3 = eigenval.at<double>(2, 0);

        //_a為長軸,_b為短軸
        double _a = std::sqrt(-l3 / l2);
        double _b = std::sqrt(-l3 / l1);
        double _c = 1;
        cout << "Semi-major axis length:" << endl;
        cout << _a << endl;
        cout << "Semi-minor axis length:" << endl;
        cout << _b << endl;
        double _degrees = std::acos(std::sqrt( (std::pow(_b,2) * (std::pow(_a, 2) + std::pow(_c, 2))) / (std::pow(_a, 2) * (std::pow(_b, 2) + std::pow(_c, 2))) ));
        cout <<"degree: +-" <<_degrees << endl;
  
     

        vector<cv::Affine3d> circles2unoblique;
        circles2unoblique.push_back(cv::Affine3d(cv::Mat::eye(3, 3, CV_64FC1), 0).rotate(Vec3d(_degrees, 0, 0)));
        circles2unoblique.push_back(cv::Affine3d(cv::Mat::eye(3, 3, CV_64FC1), 0).rotate(Vec3d(-_degrees, 0, 0)));
        //cam2base * unoblique2cam * circles2unoblique = cirlces2base
        cv::Affine3d tmp1 = pos[cnt].affine * cv::Affine3d(unoblique2cam, 0) * circles2unoblique[0];
        cv::Affine3d tmp2 = pos[cnt].affine * cv::Affine3d(unoblique2cam, 0) * circles2unoblique[1];
        //If Z axis of cirlces2base classified as upward, then rotation it with opposite direction.
        double isup1 = Mat(tmp1.rotation()).at<double>(2, 2);
        double isup2 = Mat(tmp2.rotation()).at<double>(2, 2);
        cv::Affine3d _inv(-1 * cv::Mat::eye(3, 3, CV_64FC1), 0);
        tmp1 = (isup1 < 0) ? tmp1 * _inv : tmp1;
        tmp2 = (isup2 < 0) ? tmp2 * _inv : tmp2;

        cam2basetest.push_back(pos[cnt].affine);
        gripper2basetest.push_back(pos[cnt].affine* cv::Affine3d(R_cam2gripper, T_cam2gripper).inv());
        circle2basetest[cnt].push_back(tmp1);
        circle2basetest[cnt].push_back(tmp2);

        circlecents[cnt].push_back(cv::Point3d(0,
            (std::sin(_degrees)* pow(std::cos(_degrees), 2)* _c* (pow(_b, 2) + pow(_c, 2))) / (pow(_c, 2) * pow(std::cos(_degrees), 2) - pow(_b, 2) * pow(std::sin(_degrees), 2)),
            _c* std::cos(_degrees)));
        circlecents[cnt].push_back(cv::Point3d(0,
            (std::sin(-_degrees)* pow(std::cos(-_degrees), 2)* _c* (pow(_b, 2) + pow(_c, 2))) / (pow(_c, 2) * pow(std::cos(-_degrees), 2) - pow(_b, 2) * pow(std::sin(-_degrees), 2)),
            _c* std::cos(-_degrees)));


        
        //double axes_scale = 1;
        //viz::Viz3d myWindow("Viz Ellipse Cone");
        //
        //vector<Point3d> structure;
        //vector<Vec3b> colors;
        //vector<cv::Affine3d> cam2base;
        ////.rotate(Vec3d(CV_PI, 0, 0))
        //cam2base.push_back(pos[0].affine);
        //myWindow.showWidget("camera_frame_and_lines", viz::WTrajectory(cam2base, viz::WTrajectory::BOTH, 20));
        //myWindow.showWidget("camera_frustums", viz::WTrajectoryFrustums(cam2base, Matx33d(cMatrix), 20, cv::viz::Color::blue()));
        //myWindow.showWidget("Original coord", viz::WCoordinateSystem(20));
        //myWindow.showWidget("Floor Plane", viz::WPlane(Point3d(0, 0, -370.489), Vec3d(0, 0, 1), Vec3d(0, 1, 0), Size2d(1000, 1000), cv::viz::Color::black()));
        //myWindow.spin();
        cout << "===================================="<< endl;
        cnt++;
    }

    //Choosing the most possible z axis by comparing cosine similarity.
    Mat final_normal;
    double final_score = 0;
    int best_i=0, best_j=0,best_k=0, best_l=0;
    for (int i = 0; i < names.size(); i++) {
        for (int j = 0; j < 2; j++) {
            Mat vec1 = Mat(circle2basetest[i][j].rotation()).colRange(2, 3);            
            for (int k = i + 1; k < names.size(); k++) {
                for (int l = 0; l < 2; l++) {
                    Mat vec2 = Mat(circle2basetest[k][l].rotation()).colRange(2, 3);
                    double tmp_score = cosine_similarity(vec1, vec2);
                    /*cout << i << " " << j << " " << k << " " << l << endl;
                    cout << vec1 << endl;
                    cout << vec2 << endl;
                    cout << tmp_score << endl;
                    cout << final_score << endl;*/
                    if (tmp_score > final_score) {
                        final_score = tmp_score;
                        final_normal = vec1;
                        best_i = i;
                        best_j = j;
                        best_k = k;
                        best_l = l;
                    }
                }
            }
            
        }
    }
    //cout << final_normal << endl;
    //Projecting circle center(x,y,z,1) on image plane(u,v)
    Mat circlecentsMat1;
    vconcat(Mat(circlecents[best_i][best_j]), cv::Mat::ones(1, 1, CV_64FC1), circlecentsMat1);
    Mat out = Mat(circle2basetest[best_i][best_j].matrix) * circlecentsMat1;
    Mat spatialCircleCent1 = cMatrix * Mat(cam2basetest[best_i].inv().matrix).rowRange(0, 3) * out;
    spatialCircleCent1.at<double>(0, 0) = spatialCircleCent1.at<double>(0, 0) / spatialCircleCent1.at<double>(2, 0);
    spatialCircleCent1.at<double>(1, 0) = spatialCircleCent1.at<double>(1, 0) / spatialCircleCent1.at<double>(2, 0);
    spatialCircleCent1.at<double>(2, 0) = spatialCircleCent1.at<double>(2, 0) / spatialCircleCent1.at<double>(2, 0);
    vector<cv::Point2f>p1 = { cv::Point2f(float(spatialCircleCent1.at<double>(0, 0)), spatialCircleCent1.at<double>(1, 0))
    };

    Mat circlecentsMat2;
    vconcat(Mat(circlecents[best_k][best_l]), cv::Mat::ones(1, 1, CV_64FC1), circlecentsMat2);
    out = Mat(circle2basetest[best_k][best_l].matrix) * circlecentsMat2;
    Mat spatialCircleCent2 = cMatrix * Mat(cam2basetest[best_k].inv().matrix).rowRange(0, 3) * out;
    spatialCircleCent2.at<double>(0, 0) = spatialCircleCent2.at<double>(0, 0) / spatialCircleCent2.at<double>(2, 0);
    spatialCircleCent2.at<double>(1, 0) = spatialCircleCent2.at<double>(1, 0) / spatialCircleCent2.at<double>(2, 0);
    spatialCircleCent2.at<double>(2, 0) = spatialCircleCent2.at<double>(2, 0) / spatialCircleCent2.at<double>(2, 0);
    vector<cv::Point2f>p2 = { cv::Point2f(float(spatialCircleCent2.at<double>(0, 0)), float(spatialCircleCent2.at<double>(1, 0)))
    };

    //Triangulate point by given projective matrices and points
    Mat project1 = cMatrix * Mat(cam2basetest[best_i].inv().matrix).rowRange(0,3);
    Mat project1_float(3, 4, CV_32FC1);
    project1.convertTo(project1_float(Range(0, 3), Range(0, 4)), CV_32FC1);
    Mat project2 = cMatrix * Mat(cam2basetest[best_k].inv().matrix).rowRange(0, 3);
    Mat project2_float(3, 4, CV_32FC1);
    project2.convertTo(project2_float(Range(0, 3), Range(0, 4)), CV_32FC1);
    Mat s;
    triangulatePoints(project1_float, project2_float, p1, p2, s);	//輸出為 S: 齊次座標(X,Y,Z,W)，其中 W 表示座標軸的遠近參數

    //Stroing spatial circle translation info
    Mat T_final = (Mat_<double>(3,1)<<s.at<float>(0, 0) / s.at<float>(3, 0),
        s.at<float>(1, 0) / s.at<float>(3, 0),
        s.at<float>(2, 0) / s.at<float>(3, 0));

    //HROBOT device_id = Connect("140.120.182.134", 1, callBack);
    //double* _curpos = new double[6];
    //get_current_position(device_id, _curpos);
    double _curpos[6] = { -1.475, 368.0, 241.100, 0, 0, 0 };
    Pos curpos(_curpos);
    cout << curpos.rotation.col(2) << endl;
    cout << final_normal << endl;
    Mat k = curpos.rotation.col(2).cross(final_normal);
    cv::normalize(k, k, 1.0, 0.0, NORM_L2);
    k *= std::acos(cosine_similarity(curpos.rotation.col(2), final_normal));
    
    Mat current2target;
    cv::Rodrigues(k, current2target);

    vector<cv::Affine3d> final_target;
    final_target.push_back(cv::Affine3d(current2target * curpos.rotation, T_final));
    
    cout << "Target coordination:" << final_target[0].matrix << endl;
    viz::Viz3d myWindow("Viz point cloud");
    myWindow.showWidget("Original coord", viz::WCoordinateSystem(50));
    //myWindow.showWidget("Cam", viz::WTrajectory(cam2basetest, viz::WTrajectory::BOTH, 20));
    //myWindow.showWidget("Gripper", viz::WTrajectory(gripper2basetest, viz::WTrajectory::BOTH, 20));
    //myWindow.showWidget("Circle", viz::WTrajectory(circle2basetest, viz::WTrajectory::FRAMES, 20));
    myWindow.showWidget("target", viz::WTrajectory(final_target, viz::WTrajectory::FRAMES, 50));
    myWindow.showWidget("Floor Plane", viz::WPlane(Point3d(0, 0, -370.489), Vec3d(0, 0, 1), Vec3d(0, 1, 0), Size2d(1000, 1000), cv::viz::Color::black()));
    myWindow.spin();
}


int main(){

    FileStorage g2b("gripper2base.xml", FileStorage::READ);
    FileStorage w2c("world2cam.xml", FileStorage::READ);
    FileStorage cm("cMatrix.xml", FileStorage::READ);
    //[gripper frame] 轉換至 [base frame]
    vector<Pos> gripper2base;
    vector<Mat> R_gripper2base, T_gripper2base;

    //[camera frame] 轉換至 [world frame]
    vector<Pos> world2cam;
    vector<Mat> R_world2cam, T_world2cam;

    //[camera frame] 轉換至 [gripper frame] == 手眼校正的X
    vector<Pos> cam2gripper;
    Mat R_cam2gripper, T_cam2gripper;

    
    //於[image] folder 中照片之名稱利用 vector的方式儲存
    vector<string> calib_imgs;
    vector<string> model_imgs;

    //若[資料有遺漏] 或是 [強迫執行] ，則進行[Grab image sequence] 、 [camera calibration] 和 [儲存資料]，否則直接讀取資料。
    if ( (!g2b.isOpened() || !w2c.isOpened() || g2b["R_gripper2base"].size() != grabFrames_amount_calib || w2c["R_world2cam"].size() != grabFrames_amount_calib)  || force_calib) {

        FileStorage g2b("gripper2base.xml", FileStorage::WRITE);
        FileStorage w2c("world2cam.xml", FileStorage::WRITE);
        FileStorage cm("cMatrix.xml", FileStorage::WRITE);
        HROBOT device_id = Connect("140.120.182.134", 1, callBack);
        //HROBOT device_id = Connect("127.0.0.1", 1, callBack);
        

        std::atexit(atexit_handler);

        
        //Get [R_gripper2base], [T_gripper2base] and save [image sequences].
        if (R_gripper2base.empty() || T_gripper2base.empty() || R_gripper2base.size() != T_gripper2base.size()) {
            calib_grab_motion(device_id,lookat_point_calib, grabFrames_amount_calib, 30, 0, R_gripper2base, T_gripper2base,"calibimgs");
        }
        //Storing R_gripper2base info as a array.
        writeXml(g2b, "gripper2base", R_gripper2base, T_gripper2base);

        const int _argc = 9;
        char** _argv = new char* [_argc];
        for (int i = 0; i < _argc; i++) {
            _argv[i] = new char[24];
        }
        std::strcpy(_argv[0], "camera_calibration.exe");
        std::strcpy(_argv[1], "-w=6");
        std::strcpy(_argv[2], "-h=9");
        std::strcpy(_argv[3], "-o=camera.yml");
        std::strcpy(_argv[4], "-op");
        std::strcpy(_argv[5], "-oe");
        std::strcpy(_argv[6], "-oo");
        std::strcpy(_argv[7], "-s=19.5");
        std::strcpy(_argv[8], "image_list.xml");

        camera_calibration(_argc, _argv);

        writeXml(cm, "cMatrix", cMatrix);

        //Getting the transformation matrix of [world2cam]
        
        for (int i = 0; i < grabFrames_amount_calib;i++){
            Mat rvec = cv::Mat::zeros(3, 1, CV_32FC1), tvec = cv::Mat::zeros(3, 1, CV_32FC1);
            solvePnPRansac(oPoints, iPoints[i], cMatrix, dCoeffs, rvec, tvec, false, 100, 2);
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            R_world2cam.push_back(R);
            T_world2cam.push_back(tvec.reshape(0,3));
        }
        
        //Storing R_world2cam info as a array.
        writeXml(w2c, "world2cam", R_world2cam, T_world2cam);

        for (int i = 0; i < _argc; i++) {
            delete[] _argv[i];
        }
        delete[] _argv;

        std::cout << "camera calibration session over!" << endl;
    }
    else {
        
        //Reading  gripper2base and world2cam from xml.
        readXml(g2b, "gripper2base", R_gripper2base, T_gripper2base);
        readXml(w2c, "world2cam", R_world2cam, T_world2cam);
        readXml(cm, "cMatrix", cMatrix);
    }

    g2b.release();
    w2c.release();
    cm.release();
    calib_imgs.clear();

    get_filenames("calibimgs", calib_imgs);

    //用park的方法做手眼校正，將結果存於[R_cam2gripper]和[T_cam2gripper]。
    cv::calibrateHandEye(R_gripper2base, T_gripper2base, R_world2cam, T_world2cam, R_cam2gripper, T_cam2gripper, CALIB_HAND_EYE_PARK);
    FileStorage c2g("cam2gripper.xml", FileStorage::WRITE);
    writeXml(c2g, "cam2gripper", R_cam2gripper, T_cam2gripper);

    //印出結果
    std::cout << "=======================Output gripper2base matrix=======================" << endl;
    for (int i = 0; i < R_gripper2base.size(); i++) {
        std::cout << R_gripper2base[i] << endl;
        std::cout << T_gripper2base[i] << endl;
    }
    std::cout << endl;
    std::cout << "=====================================================================" << endl;

    //Get world2cam matrix
    std::cout << "=======================Output world2cam matrix=======================" << endl;
    for (int i = 0; i < R_world2cam.size(); i++) {
        std::cout << R_world2cam[i] << endl;
        std::cout << T_world2cam[i] << endl;
    }
    std::cout << endl;
    std::cout << "======================================================    ===============" << endl;

    std::cout << "=======================Output cam2gripper initial Calib=======================" << endl;
    std::cout << R_cam2gripper << endl;
    std::cout << T_cam2gripper << endl;
    std::cout << "=====================================================================" << endl;

    get_RMSE(R_cam2gripper, T_cam2gripper);
    
 
    ellipsesDetection(R_cam2gripper, T_cam2gripper);
    double postmp[6] = { -540.227,217.628,-450.959,15.871,-0.220 };
    cout << Pos(postmp).rotation << endl;
    return 0;


    /*double axes_scale = 20;
    viz::Viz3d myWindow("Viz point cloud");

    myWindow.showWidget("Original coord", viz::WCoordinateSystem(axes_scale));

    myWindow.showWidget("Floor Plane", viz::WPlane(Point3d(0, 0, -370.489), Vec3d(0, 0, 1), Vec3d(0, 1, 0), Size2d(1000, 1000), cv::viz::Color::black()));*/

    //    viz::WCloud structure_widget(structure, colors);
    //    myWindow.showWidget("Point Cloud", structure_widget);
    //myWindow.showWidget("camera_frustums", viz::WTrajectoryFrustums(base2cam4model, Matx33d(cMatrix), axes_scale, cv::viz::Color::blue()));
    // 
    //vector<cv::Affine3d> cam2basetest;
    //vector<cv::Affine3d> gripper2basetest;
    //for (int i = 0; i < R_gripper2base.size(); i++) {
    //    cam2basetest.push_back(cv::Affine3d(R_gripper2base[i], T_gripper2base[i])* cv::Affine3d(R_cam2gripper, T_cam2gripper));
    //    gripper2basetest.push_back(cv::Affine3d(R_gripper2base[i], T_gripper2base[i]));
    //}
    //myWindow.showWidget("1", viz::WTrajectory(gripper2basetest, viz::WTrajectory::BOTH, axes_scale - 10));
    //myWindow.showWidget("2", viz::WTrajectory(cam2basetest, viz::WTrajectory::BOTH, axes_scale - 10));
    //myWindow.spin();
    

    //FileStorage g2b4m("gripper2base4model.xml", FileStorage::READ);
    //FileStorage c2g("cam2gripper.xml", FileStorage::READ);
    //vector<Mat> R_gripper2base4model, T_gripper2base4model;
    //vector<Mat> R_base2cam4model, T_base2cam4model;
    //vector<cv::Affine3d> base2cam4model;

    

    

    //if (!c2g.isOpened()) {
    //    if (!g2b4m.isOpened() || force_model) {

    //        FileStorage g2b4m("gripper2base4model.xml", FileStorage::WRITE);
    //        HROBOT device_id = Connect("140.120.182.134", 1, callBack);
    //        //HROBOT device_id = Connect("127.0.0.1", 1, callBack);

    //        double radius = 140;
    //        double added_z = -100;

    //        //model_grab_motion(device_id, lookat_point_calib, grabFrames_amount_model, radius, added_z, R_gripper2base4model, T_gripper2base4model, "modelimgs");
    //        model_grab_motion(device_id, lookat_point_model, 70, radius, added_z, R_gripper2base4model, T_gripper2base4model, "modelimgs");
    //        //Storing R_gripper2base4model info as a array.
    //        writeXml(g2b4m, "gripper2base4model", R_gripper2base4model, T_gripper2base4model);

    //    }
    //    else {
    //        readXml(g2b4m, "gripper2base4model", R_gripper2base4model, T_gripper2base4model);
    //    }

    //    get_filenames("modelimgs", model_imgs);


    //    vector<Vec3b> colors;
    //    vector<vector<KeyPoint>> key_points_for_all;            //[image sequences[keypoints of image]]
    //    vector<Mat> descriptors;                                //[image sequences[descriptors of image]] shape = [50,keypoints,128]
    //    vector<vector<Vec3b>> colors_for_all;                   //[image sequences[keypoints color of image]]
    //    vector<vector<DMatch>> matches_for_all;                 //[image sequences pairs[matched keypoints from a pair of image]]
    //    vector<Point3d> structure;
    //    vector<vector<int>> correspond_struct_idx;
    //    vector<Point2f> p1, p2;
    //    vector<Vec3b> c2;
    //    vector<int> matchFeatureSizeVec;
    //    int avgOfQ1FeatureNum = 0;
    //    extract_features_SIFT(model_imgs, key_points_for_all, descriptors, colors_for_all);
    //    match_features_KNN(descriptors, matches_for_all);

    //    for (int i = 0; i < matches_for_all.size(); i++) {
    //        matchFeatureSizeVec.push_back(matches_for_all[i].size());
    //    }
    //    sort(matchFeatureSizeVec.begin(), matchFeatureSizeVec.end(), less<int>());
    //    vector<int>::iterator it_i;


    //    for (int i = 0; i < matchFeatureSizeVec.size() / 4; i++) {
    //        avgOfQ1FeatureNum += matchFeatureSizeVec[i];
    //    }


    //    avgOfQ1FeatureNum /= (matchFeatureSizeVec.size() / 4);
    //    std::cout << "前25%特徵匹配最少的特徵點數量之平均: " << avgOfQ1FeatureNum << endl;

    //    const int Remaining_Feature_Point_Number = 40;	// 設定數值為所有特徵點都留下特徵點數量最少那一組的特徵點點數
    //    leaveBetterFeature(matches_for_all, Remaining_Feature_Point_Number);	// 留下特徵相似度前幾名的特徵

    //    bool whetherShowPhotos = false;
    //    showPhotoAlbumMatchingCondition(model_imgs, key_points_for_all, matches_for_all, whetherShowPhotos);

    //    Mat R_pos1 = (Mat_<double>(1, 3) << -0.583, -16.423, 0);
    //    Mat T_pos1 = (Mat_<double>(1, 3) << -109.950, 431.824, 55.793);
    //    Mat R_pos2 = (Mat_<double>(1, 3) << 8.350, 16.877, 2.874);
    //    Mat T_pos2 = (Mat_<double>(1, 3) << 121.463, 340.049, 55.793);
    //    Mat R_base2camtest1, R_base2camtest2;
    //    Mat T_base2camtest1, T_base2camtest2;
    //    eulerAngle2rotationMatrix(R_pos1, R_pos1);
    //    eulerAngle2rotationMatrix(R_pos2, R_pos2);
    //    vector<Mat> R_g2b = { R_pos1,R_pos2 };
    //    vector<Mat> T_g2b = { T_pos1,T_pos2 };


    //    std::cout << "Feature extraction and Feature matching is over!" << endl;
    //    std::cout << "Start bundle adjustment!" << endl;
    //    std::cout << "Before BA:" << endl;
    //    

    //    getbase2cam(R_pos1, T_pos1, R_cam2gripper, T_cam2gripper, R_base2camtest1, T_base2camtest1);
    //    getbase2cam(R_pos2, T_pos2, R_cam2gripper, T_cam2gripper, R_base2camtest2, T_base2camtest2);
    //    vector<Mat> R_ext1, ext2;
    //    vector<cv::Point2f> _p1, _p2;
    //    _p1.push_back({ 549,331 });
    //    _p2.push_back({ 381,93 });

    //    vector<cv::Point3d> structure_test;
    //    cv::Point3d gt = { -98.771,480.321,-257.421 };
    //    cust_reconstruct(cMatrix, R_base2camtest1, T_base2camtest1, R_base2camtest2, T_base2camtest2, _p1, _p2, structure_test);
    //    for (int j = 0; j < structure_test.size(); j++) {
    //        std::cout << structure_test << endl;
    //        cout << "RMSE: " << std::sqrt((pow(structure_test[j].x - gt.x, 2) + pow(structure_test[j].y - gt.y, 2) + pow(structure_test[j].z - gt.z, 2) / 3)) << endl;
    //    }

    //    for (int z = 0; z < 10; z++) {

    //        if (z > 0) {
    //            structure.clear();
    //            colors.clear();
    //        }

    //        R_base2cam4model.clear();
    //        T_base2cam4model.clear();
    //        for (int i = 0; i < R_gripper2base4model.size(); i++) {
    //            Mat _R_base2cam4model, _T_base2cam4model;
    //            getbase2cam(R_gripper2base4model[i], T_gripper2base4model[i], R_cam2gripper, T_cam2gripper, _R_base2cam4model, _T_base2cam4model);
    //            R_base2cam4model.push_back(_R_base2cam4model);
    //            T_base2cam4model.push_back(_T_base2cam4model);
    //        }



    //        get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
    //        get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);

    //        cust_reconstruct(cMatrix, R_base2cam4model[0], T_base2cam4model[0], R_base2cam4model[1], T_base2cam4model[1], p1, p2, structure);
    //        correspond_struct_idx.clear();
    //        correspond_struct_idx.resize(key_points_for_all.size());

    //        for (int i = 0; i < key_points_for_all.size(); ++i)
    //        {
    //            correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
    //        }

    //        int idx = 0;
    //        vector<DMatch>& matches = matches_for_all[0];
    //        for (int i = 0; i < matches.size(); ++i)
    //        {
    //            //if (mask.at<uchar>(i) == 0)
    //                //continue;

    //            correspond_struct_idx[0][matches[i].queryIdx] = idx;
    //            correspond_struct_idx[1][matches[i].trainIdx] = idx;
    //            ++idx;
    //        }

    //        for (int i = 1; i < matches_for_all.size(); ++i)
    //        {
    //            vector<Point3f> object_points;
    //            vector<Point2f> image_points;


    //            vector<Point2f> p1, p2;
    //            vector<Vec3b> c1, c2;
    //            get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
    //            get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);


    //            //根據之前求得的R，T進行三維重建
    //            vector<Point3d> next_structure;
    //            cust_reconstruct(cMatrix, R_base2cam4model[i], T_base2cam4model[i], R_base2cam4model[i + 1], T_base2cam4model[i + 1], p1, p2, next_structure);
    //            //將新的重建結果與之前的融合
    //            fusion_structure(
    //                matches_for_all[i],
    //                correspond_struct_idx[i],
    //                correspond_struct_idx[i + 1],
    //                structure,
    //                next_structure,
    //                colors,
    //                c1
    //            );

    //        }

    //        bundle_adjustment(cMatrix, R_gripper2base4model, T_gripper2base4model, R_cam2gripper, T_cam2gripper, correspond_struct_idx, key_points_for_all, structure);

    //        std::cout << "iteration "<<z<<":"  << endl;

    //        getbase2cam(R_pos1, T_pos1, R_cam2gripper, T_cam2gripper, R_base2camtest1, T_base2camtest1);
    //        getbase2cam(R_pos2, T_pos2, R_cam2gripper, T_cam2gripper, R_base2camtest2, T_base2camtest2);
    //        vector<cv::Point2f> _p1, _p2;
    //        _p1.push_back({ 549,331 });
    //        _p2.push_back({ 381,93 });

    //        vector<cv::Point3d> structure_test;
    //        cv::Point3d gt = { -98.771,480.321,-257.421 };
    //        cust_reconstruct(cMatrix, R_base2camtest1, T_base2camtest1, R_base2camtest2, T_base2camtest2, _p1, _p2, structure_test);
    //        for (int j = 0; j < structure_test.size(); j++) {
    //            std::cout << structure_test << endl;
    //            cout << "RMSE: " << std::sqrt((pow(structure_test[j].x - gt.x, 2) + pow(structure_test[j].y - gt.y, 2) + pow(structure_test[j].z - gt.z, 2) / 3)) << endl;
    //        }

    //    }

    //    

    //    std::cout << "Saving handeye result..." << endl;

    //    FileStorage c2g("cam2gripper.xml", FileStorage::WRITE);

    //    writeXml(c2g, "cam2gripper", R_cam2gripper, T_cam2gripper);
    //    R_base2cam4model.clear();
    //    T_base2cam4model.clear();
    //    base2cam4model.clear();
    //    for (int i = 0; i < R_gripper2base4model.size(); i++) {
    //        Mat _R_base2cam4model, _T_base2cam4model;
    //        cv::Affine3d path;
    //        getbase2cam(R_gripper2base4model[i], T_gripper2base4model[i], R_cam2gripper, T_cam2gripper, _R_base2cam4model, _T_base2cam4model, path);
    //        R_base2cam4model.push_back(_R_base2cam4model);
    //        T_base2cam4model.push_back(_T_base2cam4model);
    //        base2cam4model.push_back(path);
    //    }

    //    /*for (int i = 0; i < structure.size(); i++) {
    //        if (structure[i].x < -500 || structure[i].x > 500 || structure[i].y < -500 || structure[i].y > 500 || structure[i].z < -500 || structure[i].z > 500) {
    //            structure[i] = Point3d(0, 0, 0);
    //        }
    //    }*/
    //    //視覺化結果。
    //double axes_scale = 20;
    //viz::Viz3d myWindow("Viz point cloud");

    //myWindow.showWidget("Original coord", viz::WCoordinateSystem(axes_scale));

    //myWindow.showWidget("Floor Plane", viz::WPlane(Point3d(0, 0, -370.489), Vec3d(0, 0, 1), Vec3d(0, 1, 0), Size2d(1000, 1000), cv::viz::Color::black()));

    ////    viz::WCloud structure_widget(structure, colors);
    ////    myWindow.showWidget("Point Cloud", structure_widget);

    //vector<cv::Affine3d> cam2basetest;
    //vector<cv::Affine3d> gripper2basetest;
    //for (int i = 0; i < R_gripper2base.size(); i++) {
    //    cam2basetest.push_back(cv::Affine3d(R_cam2gripper, T_cam2gripper) * cv::Affine3d(R_gripper2base[i], T_gripper2base[i]));
    //    gripper2basetest.push_back(cv::Affine3d(R_gripper2base[i], T_gripper2base[i]));
    //}
    //myWindow.showWidget("camera_frame_and_lines", viz::WTrajectory(gripper2basetest, viz::WTrajectory::BOTH, axes_scale - 10));
    //myWindow.showWidget("camera_frame_and_lines", viz::WTrajectory(cam2basetest, viz::WTrajectory::BOTH, axes_scale - 10));
    ////myWindow.showWidget("camera_frustums", viz::WTrajectoryFrustums(base2cam4model, Matx33d(cMatrix), axes_scale, cv::viz::Color::blue()));

    //myWindow.spin();

    
    

    
}