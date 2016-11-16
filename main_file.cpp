// CNNModel.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "FaceDetect.h"
#include <io.h>


string image_path = "E:\\220project\\CNN_Model_C++\\VOC2007";
string pos_path = "md" + ' ' + image_path + "\\pos";
string neg_path = "md" + ' ' + image_path + "\\neg";

void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void para_to_vec()
{
	ifstream parafile("D:\\杨赟\\CNN\\Model.txt");
	if (!parafile.is_open())
		cout << "open parafile error" << endl;


	string temp;
	int num_line = 1;
	while (getline(parafile, temp))
	{
		vector<string> temp_para = split(temp, ',');
		int temp_para_loc = 0;

		if (num_line == 1)
		{
			for (int i = 0; i<CONV1_FILTER_NUM; ++i)
				for (int j = 0; j<NUM_CHANNLS; ++j)
					for (int p = 0; p<FILTER_SIZE; ++p)
						for (int q = 0; q<FILTER_SIZE; ++q)
							conv1_weight[i][j][p][q] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 2)
		{
			for (int i = 0; i<CONV1_FILTER_NUM; ++i)
				conv1_bias[i] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 3)
		{
			for (int i = 0; i<CONV2_FILTER_NUM; ++i)
				for (int j = 0; j<CONV1_FILTER_NUM; ++j)
					for (int p = 0; p<FILTER_SIZE; ++p)
						for (int q = 0; q<FILTER_SIZE; ++q)
							conv2_weight[i][j][p][q] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 4)
		{
			for (int i = 0; i<CONV2_FILTER_NUM; ++i)
				conv2_bias[i] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 5)
		{
			for (int i = 0; i<CONV3_FILTER_NUM; ++i)
				for (int j = 0; j<CONV2_FILTER_NUM; ++j)
					for (int p = 0; p<FILTER_SIZE; ++p)
						for (int q = 0; q<FILTER_SIZE; ++q)
							conv3_weight[i][j][p][q] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 6)
		{
			for (int i = 0; i<CONV3_FILTER_NUM; ++i)
				conv3_bias[i] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 7)
		{
			for (int i = 0; i<CONV4_FILTER_NUM; ++i)
				for (int j = 0; j<CONV3_FILTER_NUM; ++j)
					for (int p = 0; p<FILTER_SIZE; ++p)
						for (int q = 0; q<FILTER_SIZE; ++q)
							conv4_weight[i][j][p][q] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 8)
		{
			for (int i = 0; i<CONV4_FILTER_NUM; ++i)
				conv4_bias[i] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 9)
		{
			for (int i = 0; i<2048; ++i)
				for (int j = 0; j<FC1_NUM; ++j)
					fc1_weight[i][j] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 10)
		{
			for (int i = 0; i<FC1_NUM; ++i)
				fc1_bias[i] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 11)
		{
			for (int i = 0; i<FC1_NUM; ++i)
				for (int j = 0; j<NUM_LABELS; ++j)
					fc2_weight[i][j] = atof(temp_para[temp_para_loc++].c_str());
		}

		if (num_line == 12)
		{
			for (int i = 0; i<NUM_LABELS; ++i)
				fc2_bias[i] = atof(temp_para[temp_para_loc++].c_str());
		}

		num_line++;
		temp.clear();
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	para_to_vec();
	CNN MyCNN;
	vector<string> all_image;
	getFiles(image_path, all_image);

	system(pos_path.c_str());
	system(neg_path.c_str());

	for (int i = 0; i < all_image.size(); ++i)
	{
		Mat image = imread(all_image[i]);
		//string copystr;
		bool has_dalai = DetectHaarFace(image, pHaarClassCascade, all_image[i]);
		//if (has_dalai)
		//{
		//	string copystr = "xcopy " + image_path + "\\" + all_image[i] + ' ' + image_path + "\\pos";
		//	
		//}
		//else
		//{
		//	string copystr = "xcopy " + image_path + "\\" + all_image[i] + ' ' + image_path + "\\neg";
		//}
		//system(copystr.c_str());
	}


	return 0;
}
