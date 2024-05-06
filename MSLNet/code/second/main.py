import time
import tensorflow as tf
import re,os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import open3d as o3d
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if re.match(r'.*', f,re.I):
                fullname = os.path.join(root, f)
                yield fullname

colorBase=r'..\\..\\output\\color\\'
edgeBase=r'..\\..\\output\\edge\\'
heatBase=r'..\\..\\output\\heat\\'
xyzBase=r'.\\normal\\'
k40Base=r'.\\40\\'
#清空40缓存数据
for k40Path in findAllFile(k40Base):
    os.remove(k40Path)

#首先调用exe程序，计算内蕴形状描述符，并将输出结果暂时保存在40文件夹中
exePath=r'IntrisicNeighbor.exe'
os.system(exePath)

#接下来利用神经网络提取尖锐特征
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#设置阈值，值越高，越少的点被识别为尖锐特征
threshold=0.5

#定义模型
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128,input_dim=40,use_bias=True,kernel_initializer='uniform',
                            bias_initializer='zeros',activation='relu'))
model.add(tf.keras.layers.Dense(units=128,activation='relu'))
model.add(tf.keras.layers.Dense(units=256,activation='relu'))
model.add(tf.keras.layers.Dense(units=256,activation='relu'))
model.add(tf.keras.layers.Dense(units=1024,activation='relu'))
model.add(tf.keras.layers.Dense(units=512,activation='relu'))
model.add(tf.keras.layers.Dense(units=256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),loss='binary_crossentropy',metrics=['accuracy'])
model.load_weights(r'./checkpoint/normal0105.ckpt')
#执行边缘提取
for k40Path in findAllFile(k40Base):
    start_time = time.time()
    name = re.split(r'\\', k40Path)[-1]
    name = re.split(r'\.txt', name)[0]
    colorPath = colorBase + name + r'.txt'
    edgePath = edgeBase + name + r'.txt'
    heatPath = heatBase + name + r'.txt'
    xyz_path=xyzBase+name+r'.txt'
    k40_path=k40Base+name+r'.txt'
    xyz = np.array(pd.read_csv(xyz_path, sep=' ', header=None).iloc[:, 0:3])
    df_data=pd.read_csv(k40_path,sep=' ',header=None)
    # 分离特征值与标签
    ndarray_data = df_data.values
    features = ndarray_data
    # 数据标准化
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    norm_feature = minmax_scale.fit_transform(features)
    x_data = norm_feature

    label_predict = model.predict(x_data)
    label = np.where(label_predict > threshold, True, False).squeeze()
    label_noyellow = np.copy(label)
    # 删除大片黄色点
    edge = xyz[label]
    pcd = o3d.io.read_point_cloud(xyz_path, format='xyz')
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # 建立KD树索引
    for i in range(len(xyz)):
        count_yellow = 0
        count_red = 0
        count_green = 0
        if label[i] == False:
            continue
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 30)  # K近邻搜索
        neighbor_1 = idx[1:]
        heat_nei = label_predict[neighbor_1].squeeze()
        for heat in heat_nei:
            if heat >= 0.4 and heat < 0.6:
                count_yellow = count_yellow + 1
            if heat >= 0.8:
                count_red = count_red + 1
            if heat < 0.4:
                count_green = count_green + 1
        if count_yellow + count_red >= 30 * 0.6:
            if count_green < 30 * 0.1:
                label_noyellow[i] = False
    edge = xyz[label_noyellow]
    color = np.where(label_predict > threshold, [255, 128, 128], [128, 128, 255])
    data_color = np.column_stack([xyz, color])
    # 热度图
    color_label = np.array(model.predict(x_data)).squeeze()
    red = np.ones(len(color_label))
    green = np.ones(len(red))
    blue = np.ones(len(red))
    max_color = max(color_label)
    for i in range(len(color_label)):
        if color_label[i] >= 0.75:
            red[i] = 255
            green[i] = (max_color - color_label[i]) * (1 / (max_color - 0.75)) * 255
            blue[i] = 0
        elif color_label[i] >= 0.5 and color_label[i] < 0.75:
            red[i] = (color_label[i] - 0.5) * 4 * 255
            green[i] = 255
            blue[i] = 0
        elif color_label[i] >= 0.25 and color_label[i] < 0.5:
            red[i] = 0
            green[i] = 255
            blue[i] = (0.5 - color_label[i]) * 4 * 255
        elif color_label[i] < 0.25:
            red[i] = 0
            green[i] = (color_label[i]) * 4 * 255
            blue[i] = 255
    data_color = np.column_stack([xyz, red, green, blue])
    np.savetxt(colorPath, data_color, fmt='%.5f')
    np.savetxt(edgePath, edge, fmt='%.5f')
    np.savetxt(heatPath, data_color, fmt='%.5f')
    end_time = time.time()
    print(f'耗费的时间为{end_time - start_time}')