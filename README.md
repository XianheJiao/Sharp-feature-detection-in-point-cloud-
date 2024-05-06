三维点云尖锐特征提取

环境配置：
必要：
matlab2018b，
python==3.6，
tensorflow==2.4.0 ，
numpy，
pandas，
sklearn，
open3d
如果要使用内蕴邻域源代码：
visual studio 2019
pcl 1.12.0 

程序的输入：txt格式的点云，放在input文件夹中（使用WriteName.py可以快速将名字统一到test_all.txt，便于后续使用）
程序的输出：输出点云的尖锐特征点，包括尖锐特征点集、点云特征颜色图、点云特征热度图


使用方式（本程序所有的代码都实现了批量处理）：
第一步：将所有txt格式的点云放到input中，然后使用WriteName.py
第二步：运行MSLNet/code/first/RUN.m，生成点云的法向信息，保存至MSLNet/code/second/normal文件夹下
第三步：运行MSLNet/code/second/main.py 生成点云的尖锐特征点

注：此程序法向计算引用自https://github.com/hrzhou2/NH-Net-master
在此非常感谢hrzhou的工作，优点是法向计算准确，缺点是计算较慢，经过测试，5w点大约在5-10分钟内，50w点在2小时内，如果需要速度较快的法向计算方式，可以自行利用PCA实现法向计算，按同样的格式保存至normal文件夹下


 