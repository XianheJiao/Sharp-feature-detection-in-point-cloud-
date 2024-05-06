#include <iostream>
#include <vector>
#include <pcl/kdtree/kdtree_flann.h>  //kdtree近邻搜索
#include <pcl/io/pcd_io.h>  //文件输入输出
#include <pcl/point_types.h>  //点类型相关定义
#include <pcl/common/distances.h>
#include <boost/thread/thread.hpp>
#include <string>
#include <time.h>
using namespace std;

void ReadnormalFromTxt(const std::string& file_path, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,vector<vector<float>>&normal)
{
	std::ifstream file(file_path.c_str());//c_str()：生成一个const char*指针，指向以空字符终止的数组。
	std::string line;
	while (getline(file, line)) {
		std::stringstream ss(line);
		pcl::PointXYZ pointTemp;
		vector<float> normalTemp(3);
		ss >> pointTemp.x;
		ss >> pointTemp.y;
		ss >> pointTemp.z;
		ss >> normalTemp[0];
		ss >> normalTemp[1];
		ss >> normalTemp[2];
		normal.push_back(normalTemp);
		cloud->push_back(pointTemp);
	}
	file.close();
}
vector<int> unique_element_in_vector(vector<int> v) {
	vector<int>::iterator vector_iterator;
	sort(v.begin(), v.end());
	vector_iterator = unique(v.begin(), v.end());
	if (vector_iterator != v.end()) {
		v.erase(vector_iterator, v.end());
	}
	return v;
}

vector<int> vectors_intersection(vector<int> v1, vector<int> v2) {
	vector<int> v;
	sort(v1.begin(), v1.end());
	sort(v2.begin(), v2.end());
	set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(), inserter(v, v.begin()));//求差集 v1有  v2没有
	return v;
}
vector<int> delete_element_in_vector(vector<int> v, int element) {
	vector<int>::iterator it;
	it = find(v.begin(), v.end(), element);
	if (it != v.end()) {
		v.erase(it);
	}

	return v;
}
int main()
{
	_finddata64i32_t fileInfo;
	string path_base = "normal/";
	string out_path_base = "40/";

	//string path_base = "E:\\MSLNet\\code\\second\\normal\\";
	//string out_path_base = "E:\\MSLNet\\code\\second\\40\\";
	stringstream ss;
	ss << path_base;
	ss << "*";

	intptr_t hFile = _findfirst(ss.str().c_str(), &fileInfo);
	int skip_2 = 0;
	if (hFile == -1) {
		return -1;
	}

	do
	{
		if (skip_2 < 2) {
			skip_2 = skip_2 + 1;
			continue;
		}
		cout << fileInfo.name << endl;
		//代码开始

		cout << "start!" << endl;
		time_t start, end;
		start = clock();

		string path = path_base + fileInfo.name;
		string out_path = out_path_base + fileInfo.name;
		//读取点云数据
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		vector<vector<float>> normal;
		ReadnormalFromTxt(path, cloud, normal);
		//建立kd-tree
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //建立kdtree对象
		vector<vector<int>>neighbor_1(cloud->size());
		vector<int>search_neightbor;
		vector<int>now_neightbor;
		vector<vector<int>>difference(cloud->size());
		vector<vector<int>>adjacency_neightbor(cloud->size());
		vector<vector<int>>adjacency(cloud->size());
		pcl::PointXYZ searchPoint;
		kdtree.setInputCloud(cloud); //设置需要建立kdtree的点云指针
		int K = 7;
		vector<int> pointIdxNKNSearch(K);
		vector<float> pointNKNSquaredDistance(K);
		//读取法向

		//K近邻搜索
		for (int i = 0; i < cloud->size(); i++) {
			searchPoint = cloud->points[i]; //设置查找点
			kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
			auto it = pointIdxNKNSearch.begin();
			pointIdxNKNSearch.erase(it);
			neighbor_1[i] = pointIdxNKNSearch;
		}
		adjacency = neighbor_1;
		int ring = 5;
		int scale_num = 40;
		for (int now_ring = 2; now_ring <= ring; now_ring++) {
			for (int i = 0; i < cloud->size(); i++) {
				if (now_ring == 2) {
					search_neightbor = adjacency[i];
				}
				else {
					search_neightbor = difference[i];
				}
				for (int now_idx = 0; now_idx < search_neightbor.size(); now_idx++) {
					now_neightbor = neighbor_1[search_neightbor[now_idx]];
					adjacency_neightbor[i].insert(adjacency_neightbor[i].end(), now_neightbor.begin(), now_neightbor.end());
				}
				adjacency_neightbor[i] = unique_element_in_vector(adjacency_neightbor[i]);
				difference[i] = vectors_intersection(adjacency_neightbor[i], adjacency[i]);
				difference[i] = delete_element_in_vector(difference[i], i);
				adjacency[i].insert(adjacency[i].end(), difference[i].begin(), difference[i].end());
			}
		}

		vector<float> now_point_normal;
		vector<float> neightbor_point_normal;
		vector<float> distence;
		float cos;
		vector<vector<float>> cos_list(cloud->size());
		for (int i = 0; i < cloud->size(); i++) {
			adjacency[i].resize(scale_num);
			for (int n = 0; n < scale_num; n++) {
				float sqdis = pcl::euclideanDistance(cloud->points[i], cloud->points[adjacency[i][n]]);
				distence.push_back(sqdis);
			}
			//整体有序，插入排序
			for (int i1 = 0; i1 < scale_num; i1++) {
				for (int j1 = i1; j1 > 0 && distence[j1] < distence[j1 - 1]; j1--) {
					float temp = distence[j1 - 1];
					distence[j1 - 1] = distence[j1];
					distence[j1] = temp;

					int temp_idx = adjacency[i][j1 - 1];
					adjacency[i][j1 - 1] = adjacency[i][j1];
					adjacency[i][j1] = temp_idx;
				}
			}
			distence.clear();
			now_point_normal = normal[i];
			for (int m = 0; m < adjacency[i].size(); m++) {
				neightbor_point_normal = normal[adjacency[i][m]];
				cos = abs(now_point_normal[0] * neightbor_point_normal[0] + now_point_normal[1] * neightbor_point_normal[1] + now_point_normal[2] * neightbor_point_normal[2]);
				cos_list[i].push_back(cos);
			}
		}
		ofstream ofs;
		ofs.open(out_path, ios::out);
		for (int i = 0; i < cloud->size(); i++) {
			for (int m = 0; m < scale_num; m++) {
				ofs << cos_list[i][m];
				if (m < scale_num - 1) {
					ofs << " ";
				}
				else {
					ofs << endl;
				}
			}

		}
		ofs.close();
		end = clock();
		cout << "time: " << (end - start) << " ms" << std::endl;
		//代码结束
	} while (_findnext(hFile, &fileInfo) == 0);

	return 0;
}


