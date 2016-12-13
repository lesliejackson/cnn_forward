#pragma once
#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<fstream>
#include<sstream>
#include<stdlib.h>

using namespace std;
#define NUM_CHANNLS 1
#define FILTER_SIZE 3
#define CONV1_FILTER_NUM 64
#define CONV2_FILTER_NUM 64
#define CONV3_FILTER_NUM 128
#define CONV4_FILTER_NUM 128
#define NUM_LABELS 2
#define FC1_NUM 224
#define PIXEL_DEPTH 255
#define IMAGE_SIZE 112

vector<vector<vector<vector<float> > > > conv1_weight(CONV1_FILTER_NUM, vector<vector<vector<float> > >(NUM_CHANNLS, vector<vector<float> >(FILTER_SIZE, vector<float>(FILTER_SIZE))));
vector<float> conv1_bias(CONV1_FILTER_NUM);

vector<vector<vector<vector<float> > > > conv2_weight(CONV2_FILTER_NUM, vector<vector<vector<float> > >(CONV1_FILTER_NUM, vector<vector<float> >(FILTER_SIZE, vector<float>(FILTER_SIZE))));
vector<float> conv2_bias(CONV2_FILTER_NUM);

vector<vector<vector<vector<float> > > > conv3_weight(CONV3_FILTER_NUM, vector<vector<vector<float> > >(CONV2_FILTER_NUM, vector<vector<float> >(FILTER_SIZE, vector<float>(FILTER_SIZE))));
vector<float> conv3_bias(CONV3_FILTER_NUM);

vector<vector<vector<vector<float> > > > conv4_weight(CONV4_FILTER_NUM, vector<vector<vector<float> > >(CONV3_FILTER_NUM, vector<vector<float> >(FILTER_SIZE, vector<float>(FILTER_SIZE))));
vector<float> conv4_bias(CONV4_FILTER_NUM);

vector<vector<float> > fc1_weight(2048, vector<float>(FC1_NUM));
vector<float> fc1_bias(FC1_NUM);

vector<vector<float> > fc2_weight(FC1_NUM, vector<float>(NUM_LABELS));
vector<float> fc2_bias(NUM_LABELS);

vector<string> split(string s, char sep)
{
	vector<string> res;
	int pre_sep = 0;

	for (int i = 0; i<s.length(); ++i)
	{
		if (s[i] == sep)
		{
			res.push_back(s.substr(pre_sep, i - pre_sep));
			pre_sep = i + 1;
		}

	}
	res.push_back(s.substr(pre_sep, s.length() - pre_sep));

	return res;
}

class CNN
{
public:
	int model(vector<vector<vector<float> > >& data);
	vector<vector<vector<float> > > relu(vector<vector<vector<float> > >& feature_map);
	vector<vector<float> > relu(vector<vector<float> >& feature_map);
	vector<vector<float> > softmax(vector<vector<float> >& fc1_output);
	vector<vector<float> > matmul(vector<vector<float> >& mat1, vector<vector<float> > mat2);
	vector<vector<vector<float> > > max_pool(vector<vector<vector<float> > >& feature_map, int pool_size); //default stride=2, padding=SAME
	vector<vector<vector<float> > > conv2d(vector<vector<vector<float> > >& feature_map, const vector<vector<vector<vector<float> > > >& conv_filter, int filter_size, int filter_num); //default stride=1, padding=SAME 
	vector<vector<vector<float> > > add_bias(vector<vector<vector<float> > >& feature_map, vector<float> bias);
	vector<vector<float> > add_bias(vector<vector<float> >& feature_map, vector<float> bias);
	vector<vector<float> > reshape(vector<vector<vector<float> > >& data);  //reshape data from 3 to 2

	float max_4(float a, float b, float c, float d)
	{
		return (a>b ? a : b) > (c>d ? c : d) ? (a>b ? a : b) : (c>d ? c : d);
	}

	vector<vector<vector<float> > > data_pretreat(vector<vector<vector<float> > >& data)
	{
		vector<vector<vector<float> > > res(data);
		int num_channel = data.size();
		int data_size = data[0].size();
		for (int k = 0; k<num_channel; ++k)
			for (int i = 0; i<data_size; ++i)
				for (int j = 0; j<data_size; ++j)
					res[k][i][j] = (data[k][i][j] - PIXEL_DEPTH / 2.0) / PIXEL_DEPTH;

		return res;
	}

};

vector<vector<vector<float> > > CNN::relu(vector<vector<vector<float> > >& feature_map)
{
	int feature_num = feature_map.size();
	int feature_height = feature_map[0].size();
	int feature_weight = feature_map[0][0].size();

	for (int i = 0; i<feature_num; ++i)
		for (int j = 0; j<feature_height; ++j)
			for (int k = 0; k<feature_weight; ++k)
			{
				if (feature_map[i][j][k] < 0)
					feature_map[i][j][k] = 0;
			}

	return feature_map;
}

vector<vector<float> > CNN::relu(vector<vector<float> >& feature_map)
{
	int feature_height = feature_map.size();
	int feature_weight = feature_map[0].size();

	for (int j = 0; j<feature_height; ++j)
		for (int k = 0; k<feature_weight; ++k)
		{
			if (feature_map[j][k] < 0)
				feature_map[j][k] = 0;
		}

	return feature_map;
}

vector<vector<float> > CNN::softmax(vector<vector<float> >& fc1_output)
{
	int rows = fc1_output.size();
	int columns = fc1_output[0].size();

	float sum = 0;

	for (int i = 0; i<columns; ++i)
		sum += (float)exp(fc1_output[0][i]);

	for (int i = 0; i<columns; ++i)
		fc1_output[0][i] = (float)exp(fc1_output[0][i]) / sum;

	return fc1_output;
}

vector<vector<float> > CNN::matmul(vector<vector<float> >& mat1, vector<vector<float> > mat2)
{
	vector<vector<float> > wrong;
	int mat1_rows = mat1.size();
	if (mat1_rows == 0) return wrong;
	int mat1_cols = mat1[0].size();
	int mat2_rows = mat2.size();
	if (mat2_rows == 0) return wrong;
	int mat2_cols = mat2[0].size();

	if (mat1_cols != mat2_rows)
		return wrong;

	vector<vector<float> > res(mat1_rows, vector<float>(mat2_cols, 0));

	for (int i = 0; i<mat1_rows; ++i)
		for (int j = 0; j<mat2_cols; ++j)
			for (int k = 0; k<mat1_cols; ++k)
				res[i][j] += mat1[i][k] * mat2[k][j];
	return res;
}

vector<vector<float> > CNN::reshape(vector<vector<vector<float> > >& data)
{
	int feature_num = data.size();
	int feature_height = data[0].size();
	int feature_weight = data[0][0].size();
	int res_loc = 0;
	vector<vector<float> > res(1, vector<float>(feature_num*feature_weight*feature_height));

	for (int i = 0; i<feature_height; ++i)
		for (int j = 0; j<feature_weight; ++j)
			for (int k = 0; k<feature_num; ++k)
			{
				res[0][res_loc++] = data[k][i][j];
			}

	return res;
}

vector<vector<vector<float> > > CNN::max_pool(vector<vector<vector<float> > >& feature_map, int pool_size)
{
	int feature_num = feature_map.size();
	int feature_height = feature_map[0].size();
	int feature_weight = feature_map[0][0].size();

	//当padding为SAME时需要填充0值以便矩阵的边缘也能处理，一般来说左边和上边填充的列数和行数为总的需要填充的列数和行数的一半
	//即padding_left = padding_cols / 2; padding_top = padding_rows/2; padding_right = padding_cols - padding_cols/2; padding_bottom = padding_rows-padding_rows/2;
	//一般来说特征图都是正方形，即padding_rows = padding_cols

	int top_left_pad = (feature_height % pool_size) / 2;
	int bottom_right_pad = feature_height % pool_size - top_left_pad;
	int new_feature_height = feature_height%pool_size == 0 ? feature_height / pool_size : feature_height / pool_size + 1;
	int new_feature_weight = feature_weight%pool_size == 0 ? feature_weight / pool_size : feature_weight / pool_size + 1;

	vector<vector<vector<float> > > res(feature_num,
		vector<vector<float> >(new_feature_height,
			vector<float>(new_feature_weight)));

	for (int i = 0; i<feature_num; ++i)
		for (int j = -top_left_pad, init_height = 0; j<feature_height; j = j + pool_size)
		{
			for (int k = -top_left_pad, init_weight = 0; k<feature_weight; k = k + pool_size)
			{
				float max_num;
				if (j >= 0 && k >= 0)
					max_num = feature_map[i][j][k];
				else
					max_num = 0;
				for (int m = 0; m<pool_size; ++m)
					for (int n = 0; n<pool_size; ++n)
					{
						if (j + m < feature_height && j + m >= 0 && k + n < feature_weight && k + n >= 0)
						{
							if (feature_map[i][j + m][k + n] > max_num)
								max_num = feature_map[i][j + m][k + n];
						}
						else
						{
							if (max_num < 0)
								max_num = 0;
						}

					}

				res[i][init_height][init_weight] = max_num;
				//if(j+1 < feature_height && k+1 < feature_weight)
				//	res[i][init_height][init_weight] = max_4(feature_map[i][j][k], feature_map[i][j][k+1], feature_map[i][j+1][k], feature_map[i][j+1][k+1]);
				//else if(j+1 < feature_height && k+1 >= feature_weight)
				//	res[i][init_height][init_weight] = max_4(feature_map[i][j][k], 0, feature_map[i][j+1][k], 0);
				//else if(j+1 >= feature_height && k+1 < feature_weight)
				//	res[i][init_height][init_weight] = max_4(feature_map[i][j][k], feature_map[i][j][k+1], 0, 0);
				//else
				//	res[i][init_height][init_weight] = max_4(feature_map[i][j][k], 0, 0, 0);

				init_weight++;
			}
			init_height++;
		}

	return res;
}

vector<vector<vector<float> > > CNN::add_bias(vector<vector<vector<float> > >& feature_map, vector<float> bias)
{
	int feature_num = feature_map.size();
	int feature_height = feature_map[0].size();
	int feature_weight = feature_map[0][0].size();

	for (int i = 0; i<feature_num; ++i)
		for (int j = 0; j<feature_height; ++j)
			for (int k = 0; k<feature_weight; ++k)
				feature_map[i][j][k] += bias[i];
	return feature_map;
}

vector<vector<float> > CNN::add_bias(vector<vector<float> >& feature_map, vector<float> bias)
{
	int feature_height = feature_map.size();
	int feature_weight = feature_map[0].size();

	for (int j = 0; j<feature_height; ++j)
		for (int k = 0; k<feature_weight; ++k)
			feature_map[j][k] += bias[k];
	return feature_map;
}

vector<vector<vector<float> > > CNN::conv2d(vector<vector<vector<float> > >& feature_map, const vector<vector<vector<vector<float> > > >& conv_filter, int filter_size, int filter_num)
{
		
//输入为三维(img_size, img_size, input_channels), 卷积核四维(filter_size,filter_size,input_channels,output_channels)
//对于每一个output_channels，其值为input_channels个filter和img卷积结果的和，filter在img上的卷积从img的左上角开始，以img上的每个像素点为中心进行卷积/	
//卷积的padding和max_pool一样	
	int feature_num = feature_map.size();
	int new_feature_num = conv_filter.size();
	int feature_height = feature_map[0].size();
	int feature_weight = feature_map[0][0].size();

	vector<vector<vector<float> > > res(new_feature_num, vector<vector<float> >(feature_height, vector<float>(feature_weight, 0)));

	for (int i = 0; i<new_feature_num; ++i)
		for (int p = 0; p<feature_height; ++p)
			for (int q = 0; q<feature_weight; ++q)
			{
				for (int m = 0; m<filter_size; ++m)
					for (int n = 0; n<filter_size; ++n)
					{
						for (int j = 0; j<feature_num; ++j)
							if (p + m - 1 < feature_height && p + m - 1 >= 0 && q + n - 1 < feature_weight && q + n - 1 >= 0)
								res[i][p][q] += conv_filter[i][j][m][n] * feature_map[j][p + m - 1][q + n - 1];
					}

			}

	return res;
}

int CNN::model(vector<vector<vector<float> > >& data)
{
	vector<vector<vector<float> > > temp_data;
	temp_data = data_pretreat(data);

	temp_data = conv2d(temp_data, conv1_weight, 3, 64);
	temp_data = relu(add_bias(temp_data, conv1_bias));
	temp_data = max_pool(temp_data, 2);

	temp_data = conv2d(temp_data, conv2_weight, 3, 64);
	temp_data = relu(add_bias(temp_data, conv2_bias));
	temp_data = max_pool(temp_data, 2);

	temp_data = conv2d(temp_data, conv3_weight, 3, 128);
	temp_data = relu(add_bias(temp_data, conv3_bias));
	temp_data = max_pool(temp_data, 2);

	temp_data = conv2d(temp_data, conv4_weight, 3, 128);
	temp_data = relu(add_bias(temp_data, conv4_bias));
	temp_data = max_pool(temp_data, 4);

	vector<vector<float> > feature = reshape(temp_data);

	feature = matmul(feature, fc1_weight);
	feature = add_bias(feature, fc1_bias);
	feature = relu(feature);
	feature = matmul(feature, fc2_weight);
	feature = add_bias(feature, fc2_bias);
	feature = softmax(feature);

	float max_num = feature[0][0];
	int res = 0;

	for (int i = 1; i<NUM_LABELS; ++i)
		if (feature[0][i] > max_num)
		{
			max_num = feature[0][i];
			res = i;
		}

	return res;
}

