from utils import *
import time
import tensorflow as tf
import re,os
from sklearn import preprocessing
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



#64k
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128,input_dim=64,use_bias=True,kernel_initializer='uniform',
                            bias_initializer='zeros',activation='relu'))
model.add(tf.keras.layers.Dense(units=128,activation='relu'))
model.add(tf.keras.layers.Dense(units=256,activation='relu'))
model.add(tf.keras.layers.Dense(units=256,activation='relu'))
model.add(tf.keras.layers.Dense(units=1024,activation='relu'))
model.add(tf.keras.layers.Dense(units=512,activation='relu'))
model.add(tf.keras.layers.Dense(units=256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#定义模型优化器
model.compile(optimizer=tf.keras.optimizers.Adam(0.00001),loss='binary_crossentropy',metrics=['accuracy'])
model.load_weights(r'C:\Users\jxh\PycharmProjects\pythonProject1\checkpoint\NH64\normal0075.ckpt')


#96k
model96=tf.keras.models.Sequential()
model96.add(tf.keras.layers.Dense(units=128,input_dim=96,use_bias=True,kernel_initializer='uniform',
                            bias_initializer='zeros',activation='relu'))
model96.add(tf.keras.layers.Dense(units=128,activation='relu'))
model96.add(tf.keras.layers.Dense(units=256,activation='relu'))
model96.add(tf.keras.layers.Dense(units=256,activation='relu'))
model96.add(tf.keras.layers.Dense(units=1024,activation='relu'))
model96.add(tf.keras.layers.Dense(units=512,activation='relu'))
model96.add(tf.keras.layers.Dense(units=256,activation='relu'))
model96.add(tf.keras.layers.Dropout(0.5))
model96.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#定义模型优化器
model96.compile(optimizer=tf.keras.optimizers.Adam(0.00001),loss='binary_crossentropy',metrics=['accuracy'])
model96.load_weights(r'C:\Users\jxh\PycharmProjects\pythonProject1\checkpoint\NH96\normal0065.ckpt')

#128k
model128=tf.keras.models.Sequential()
model128.add(tf.keras.layers.Dense(units=128,input_dim=128,use_bias=True,kernel_initializer='uniform',
                            bias_initializer='zeros',activation='relu'))
model128.add(tf.keras.layers.Dense(units=128,activation='relu'))
model128.add(tf.keras.layers.Dense(units=256,activation='relu'))
model128.add(tf.keras.layers.Dense(units=256,activation='relu'))
model128.add(tf.keras.layers.Dense(units=1024,activation='relu'))
model128.add(tf.keras.layers.Dense(units=512,activation='relu'))
model128.add(tf.keras.layers.Dense(units=256,activation='relu'))
model128.add(tf.keras.layers.Dropout(0.5))
model128.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#定义模型优化器
model128.compile(optimizer=tf.keras.optimizers.Adam(0.00001),loss='binary_crossentropy',metrics=['accuracy'])
model128.load_weights(r'C:\Users\jxh\PycharmProjects\pythonProject1\checkpoint\NH128\normal0050.ckpt')


modelavg=tf.keras.models.Sequential()
modelavg.add(tf.keras.layers.Dense(units=1,input_dim=3,use_bias=True,kernel_initializer='uniform',
                            bias_initializer='zeros',activation='sigmoid'))

#定义模型优化器
modelavg.compile(optimizer=tf.keras.optimizers.Adam(0.0001),loss='binary_crossentropy',metrics=['accuracy'])
modelavg.load_weights(r'C:\Users\jxh\PycharmProjects\pythonProject1\checkpoint\avg\normal0085.ckpt')


radius=0.05
threshold=0.6
def findAllskinFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if re.match(r'.*', f,re.I):
                fullname = os.path.join(root, f)
                yield fullname

color_base=r'E:\ABC\多尺度测试\color\\'
edge_base=r'E:\ABC\多尺度测试\edge\\'
heat_base=r'E:\ABC\多尺度测试\heat\\'
xyz_base=r'E:\ABC\xyz\no_noise\\'


k64_base=r'E:\ABC\挑选\多尺度\64\\'
k96_base=r'E:\ABC\挑选\多尺度\96\\'
k128_base=r'E:\ABC\挑选\多尺度\128\\'

temp_base=r'E:\ABC\挑选\多尺度\avg\\'
for k64_path in findAllskinFile(k64_base):
    start_time = time.time()
    name = re.split(r'\\', k64_path)[-1]
    name = re.split(r'\.', name)[0]
    flag=re.findall(r'noise',name)
    if len(flag)>0:
        continue

    # color_path=color_base+name+r'color.txt'
    color_path = color_base + name + r'.txt'
    edge_path = edge_base + name + r'.txt'
    heat_path = heat_base + name + r'.txt'
    k96_path = k96_base + name + r'.txt'
    k128_path = k128_base + name + r'.txt'
    xyz_path=xyz_base+name+r'.txt'
    xyz = np.array(pd.read_csv(xyz_path, sep=' ', header=None).iloc[:, 0:3])


    df_data=pd.read_csv(k64_path,sep=' ',header=None)
    df_data96=pd.read_csv(k96_path,sep=' ',header=None)
    df_data128=pd.read_csv(k128_path,sep=' ',header=None)

    # 分离特征值与标签
    ndarray_data = df_data.values
    ndarray_data96 = df_data96.values
    ndarray_data128 = df_data128.values

    features = ndarray_data
    features96 = ndarray_data96
    features128 = ndarray_data128

    # 数据标准化
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    norm_feature = minmax_scale.fit_transform(features)
    norm_feature96 = minmax_scale.fit_transform(features96)
    norm_feature128 = minmax_scale.fit_transform(features128)
    x_data = norm_feature
    x_data96 = norm_feature96
    x_data128 = norm_feature128

    label_predict = model.predict(x_data)
    label_predict96 = model96.predict(x_data96)
    label_predict128 = model128.predict(x_data128)

    heat1=label_predict.squeeze()
    heat2=label_predict96.squeeze()
    heat3=label_predict128.squeeze()
    heat_all=np.column_stack([heat1,heat2,heat3])
    temp_path = temp_base + name + r'.txt'
    np.savetxt(temp_path,heat_all,fmt='%.5f')
    #mean_label_predict=(label_predict+label_predict96+label_predict128)/3
    #mean_label_predict=label_predict128
    mean_label_predict=modelavg.predict(heat_all)

###

###
    label = np.where(mean_label_predict > threshold, True, False).squeeze()
    edge = xyz[label]

    label_noyellow = np.copy(label)
    # # 删除大片黄色点
    # pcd = o3d.io.read_point_cloud(xyz_path, format='xyz')
    # pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # 建立KD树索引
    #
    # for i in range(len(xyz)):
    #     count_yellow = 0
    #     count_red = 0
    #     count_green = 0
    #     if label[i] == False:
    #         continue
    #     [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 30)  # K近邻搜索
    #     neighbor_1 = idx[1:]
    #     heat_nei = label_predict[neighbor_1].squeeze()
    #     for heat in heat_nei:
    #         if heat >= 0.4 and heat < 0.6:
    #             count_yellow = count_yellow + 1
    #         if heat >= 0.8:
    #             count_red = count_red + 1
    #         if heat < 0.4:
    #             count_green = count_green + 1
    #     if count_yellow + count_red >= 30 * 0.6:
    #         if count_green < 30 * 0.1:
    #             label_noyellow[i] = False
    edge = xyz[label_noyellow]

    color = np.where(mean_label_predict > threshold, [255, 128, 128], [128, 128, 255])
    data_color = np.column_stack([xyz, color])
    np.savetxt(color_path, data_color, fmt='%.5f')
    np.savetxt(edge_path, edge, fmt='%.5f')
    end_time = time.time()
    print(f'耗费的时间为{end_time - start_time}')

    # 热度图
    color_label = np.array(model.predict(x_data)).squeeze()
    red = np.ones(len(color_label))
    green = np.ones(len(red))
    blue = np.ones(len(red))
    # edge=color_label[mask]
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

    # （255,0,0） （0,0,255）

    data_color = np.column_stack([xyz, red, green, blue])
    np.savetxt(heat_path, data_color, fmt='%.5f')