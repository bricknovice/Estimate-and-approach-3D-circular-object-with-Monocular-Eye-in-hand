#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <fstream>

extern std::vector<cv::Point3f> oPoints;				//objectPoints
extern std::vector<std::vector<cv::Point2f>> iPoints;	//imagePoints
extern cv::Mat cMatrix;									//cameraMatrix
extern cv::Mat dCoeffs;									//distCoeffs
int camera_calibration(int,char**);