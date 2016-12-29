#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include "CNN.h"

using namespace cv;
using namespace std;

IplImage *KeyFrame=NULL;
template<typename T>
string num_to_string(T x)
{
	stringstream ss;
	ss<<x; 
	string s1;
	ss>>s1;
	return s1;
}
void Mat2vector(Mat &src, vector<vector<vector<float> > > &dst)
{
	//assert(src.rows == dst[0].size());
	//MatIterator_<float> it, it2;
	//cout<<src.elemSize()<<endl;
	//it = src.begin<float>();
	//it2 = src.end<float>();
	//for (int i = 0; i<src.rows; i++)
	//{
	//	for (int j = 0; j<src.cols; j++)
	//	{
	//		dst[0][i][j] = *it++;
	//	}
	//}
	//for(int i=rect.x; i<rect.x+rect.width; ++i)
	//	for(int j=rect.y; j<rect.y+rect.height; ++j)
	//		{
	//			dst[0][i][j] = (float)(*(src.data + src.step[0] * i + src.step[1] * j));
	//		}

	for(int i=0; i<src.rows; ++i)
		for(int j=0; j<src.cols; ++j)
		{
			dst[0][i][j] = (float)(*(src.data + src.step[0] * i + src.step[1] * j));
		}
		// cv::Size size = src.size();

		// int total = size.width * size.height * src.channels();
		// int sign = 0;

		// std::vector<uchar> data(src.ptr(), src.ptr() + total);
		// std::string s(data.begin(), data.end()); 
		// string temp;
		// 	for (int i = 0; i<src.rows; i++)
		//{
		//	for (int j = 0; j<src.cols; j++)
		//	{
		//		temp.push_back(s[sign++]);
		//		dst[0][i][j] = atof(temp.c_str());
		//		temp.clear();
		//	}
		//}
		//return 0;
}

void Mat2vector(Mat &src, vector<vector<vector<float> > > &dst, IplImage *img_src, CvRect rect)
{
	for(int i=rect.x; i<rect.x+rect.width; ++i)
		for(int j=rect.y; j<rect.y+rect.height; ++j)
		{
			dst[0][i][j] = (float)(*(src.data + src.step[0] * i + src.step[1] * j));
		}
}
//vector<vector<double>> 转换为Mat  
void Vector2Mat(vector<vector<double>>src, Mat dst)
{
	assert(dst.rows == src.size());
	MatIterator_<double> it = dst.begin<double>();
	for (int i = 0; i<src.size(); i++)
	{
		for (int j = 0; j<src[0].size(); j++)
		{
			*it = src[i][j];
			it++;
		}
	}
}


string path = "D:\\opencv\\sources\\data\\haarcascades";
string dPath = path + "\\haarcascade_frontalface_alt2.xml";
const char *pcascadeName = (char *)dPath.data();
CvHaarClassifierCascade *pHaarClassCascade  = (CvHaarClassifierCascade*)cvLoad(pcascadeName);// 加载HAAR分类器   
//pHaarClassCascade = (CvHaarClassifierCascade*)cvLoad(pcascadeName);
//cvReleaseHaarClassifierCascade(&pHaarClassCascade);


bool DetectHaarFace(Mat img, CvHaarClassifierCascade *pHaarClassCascade, int ith_image)
{
	//Mat 转 IplImage
	IplImage *pSrcImage=&IplImage(img);

	bool hasFace = false;
	bool hasDalai = false;
	//加载测试图像
	IplImage *pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U, 1);
	if (pSrcImage == NULL || pGrayImage == NULL)
	{
		printf("can't load image!\n");
		return false;
	}
	cvCvtColor(pSrcImage, pGrayImage, CV_BGR2GRAY);

	if (pHaarClassCascade != NULL && pSrcImage != NULL && pGrayImage != NULL)
	{
		const static CvScalar colors[] =
		{
			CV_RGB(0,0,255),
			CV_RGB(0,128,255),
			CV_RGB(0,255,255),
			CV_RGB(0,255,0),
			CV_RGB(255,128,0),
			CV_RGB(255,255,0),
			CV_RGB(255,0,0),
			CV_RGB(255,0,255)
		};
		CvMemStorage *pcvMemStorage = cvCreateMemStorage(0);
		cvClearMemStorage(pcvMemStorage);

		//检测人脸  
		int TimeStart, TimeEnd;

		//TimeStart = GetTickCount();
		CvSeq *pcvSeqFaces = cvHaarDetectObjects(pGrayImage, pHaarClassCascade, pcvMemStorage);
		//if (pcvSeqFaces->total > 0)
		//{
		//	hasFace = true;
		//}

		//TimeEnd = GetTickCount();
		vector<vector<vector<float> > > input_data(1, vector<vector<float> >(IMAGE_SIZE, vector<float>(IMAGE_SIZE)));
		CNN cnn;
		//printf("the number of faces: %d\nSpending Time: %d ms\n", pcvSeqFaces->total, TimeEnd - TimeStart);    
		//提取人脸（矩形区域选择）  
		for (int i = 0; i < pcvSeqFaces->total; i++)
		{

			vector<Rect> eyeObjects;
			Point center;
			Scalar color = colors[i % 8];
			int radius;

			CvRect* r = (CvRect*)cvGetSeqElem(pcvSeqFaces, i);
			CvRect rect;
			rect.width = r->width;
			rect.height = r->height;
			rect.x = r->x;
			rect.y = r->y;
			//KeyFrame 检测出的目标人脸
			cvRectangle(pSrcImage, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width+5, rect.y + rect.height+5), cvScalar(0, 0, 255),6);
			KeyFrame = cvCreateImage(cvSize(rect.width, rect.height), pSrcImage->depth, pSrcImage->nChannels);
			//char *d = KeyFrame->imageData;
			//for(int i=0; i<10; ++i)
			//	for(int j=0; j<10; ++j)
			//	{
			//		cout<<(int)(*d++)<<endl;
			//	}

			cvSetImageROI(pSrcImage,rect);
			cvCopy(pSrcImage,KeyFrame);
			cvResetImageROI(pSrcImage);

			CvSize ImgSize;
			ImgSize.width=112;
			ImgSize.height=112;
			IplImage *ImgResize=cvCreateImage(ImgSize,KeyFrame->depth,KeyFrame->nChannels);;
			cvResize(KeyFrame,ImgResize,CV_INTER_NN);


			//Mat input_data_image = KeyFrame;
			Mat input_data_gray;
			//IplImage imgTemp = input_data_image;
			//IplImage *rgb_data = cvCloneImage(&imgTemp);
			//Mat resized_data;
			//resize(input_data_gray, resized_data, Size(IMAGE_SIZE, IMAGE_SIZE));
			IplImage *gray_data = cvCreateImage(cvGetSize(ImgResize), IPL_DEPTH_8U, 1);
			cvCvtColor(ImgResize, gray_data, CV_BGR2GRAY);
			input_data_gray=cv::Mat(gray_data);

			/*	cvNamedWindow("test1");
			cvShowImage("test1",KeyFrame);
			cvWaitKey();

			cvNamedWindow("test2");
			cvShowImage("test2",ImgResize);
			cvWaitKey();

			cvNamedWindow("test3");
			cvShowImage("test3",gray_data);
			cvWaitKey();*/


			/*Mat2vector(resized_data, input_data, pGrayImage, rect);
			vector<vector<float>> temp_data(input_data[0].size(), vector<float>(input_data[0][0].size()));
			for(int j=0; j<input_data[0].size(); ++j)
			for(int k=0; k<input_data[0][0].size(); ++k)
			temp_data[j][k] = input_data[0][j][k];*/

			//input_data.clear();
			//Mat temp(temp_data);
			//resize(temp, resized_data, Size(IMAGE_SIZE, IMAGE_SIZE));
			Mat2vector(input_data_gray, input_data);
			if (cnn.model(input_data) == 1)
			{
				cvRectangle(pSrcImage, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width-5, rect.y + rect.height-5), cvScalar(255, 0, 0),3);
				hasDalai = true;
			}
			else
			{
				cvRectangle(pSrcImage, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width-10, rect.y + rect.height-10), cvScalar(0, 255, 0));
			}
		}

		if(pcvSeqFaces->total > 0 && hasDalai)
		{
			string savapath = "E:\\project220\\CNN_C++\\Second\\pos\\" + num_to_string<int>(ith_image) + ".jpg";
			cvSaveImage(savapath.c_str(), pSrcImage);
			return true;
		}
		else
		{	
			string savapath = "E:\\project220\\CNN_C++\\Second\\neg" + num_to_string<int>(ith_image) + ".jpg";
			cvSaveImage(savapath.c_str(), pSrcImage);
			return false;
		}
	}
}
