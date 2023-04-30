#纯净版
#减小数据体积 更换为16位tif
#改用tifffile读取 tif
#V5改进版 提升模型速度
#将kri移植进样本中
#测试基准带有kri生成器网络 RMSE 欧式距离 以及视觉显示效果
from osgeo import gdal
import tensorflow as tf
from pykrige.ok import OrdinaryKriging
import numpy as np
import os
import time
import tifffile as tiff
import sys
from matplotlib import pyplot as plt
from IPython import display
import datetime

from PIL import ImageTk, Image, ImageDraw
import PIL
import collections
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100
CURRENT_EPOCH = 1;  # Set this if training stops in between.
RESTORE_CKP = True  # True For starting from checkpoint and For testing in interactive tool
DATBASE_PATH = 'D:\\03 1129\\0428'  # give database path here
OUTPUT_DIR = "./output"
#Open_Interactive_Tool = False # set True to open interactive tool

CKP_SAVE_INT = 10  # chkpt interval
EPOCHS = 200 # iterations


def MKDIR(Dir):
    if not os.path.isdir(Dir):
        os.mkdir(Dir)


MKDIR(OUTPUT_DIR)
MKDIR("./Test")
MKDIR(OUTPUT_DIR + "/frame")

class GRID:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
        del dataset
        return im_proj, im_geotrans, im_data

    # 写文件，以写成tif为例
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64
        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset


run = GRID()





def preprocess(img_path: str):
  f_name = bytes.decode(img_path.numpy())     # for this you need tf.py_function
  #img = tiff.imread(img_path)
  #img = tiff.imread(f_name)
  img = tiff.imread(f_name)
  #proj_D, geotrans_D, img = run.read_img(f_name)
  #矩阵进行轴转换 例如将（3，256，256）转换为（256，256，3）
  img=img.transpose((1,2,0))
  w = img.shape[1] // 3
  #real_image包含 krig插值结果和初始DEM
  real_image = img[:, w:, :]
  input_image = img[:, :w, :]
  #归一化
  input_image = (input_image / ((2335 - 304) / 2)) - 1
  #krig_image=krig_processor(input_image)
  real_image = (real_image / ((2335 - 304) / 2)) - 1
  #捕获第三通道数据
  #input_image=input_image[:,:,2]
  #real_image=real_image[:,:,2]

  #img = tf.image.resize(img, (128, 128))
  #img = tf.image.convert_image_dtype(img, tf.float32)
  input_image = tf.cast(input_image, tf.float32)
  #krig_image = tf.cast(krig_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)
  return input_image, real_image


#计算已知点数目
def cal_knownp_num(data):
    # 获取array
    data_array = data.numpy()
    # 降维 此时数据为（1，256，256，3）
    data_array = data_array.squeeze(axis=0)
    data_r = data_array[:, :, 2]
    # 获取特征点数目
    # 此处根据实际数据修改
    n_num = np.sum(data_r == -1)
    # u_num 已知点 已知点数目要不少于10
    u_num = data_r.size - n_num
    u_num = tf.cast(u_num, tf.float32)
    return u_num






#krig_intp生成器
#为了避免意外，krig_generator需要至少10个已知点
#@tf.function
def krig_generator(data):
    # 获取array
    data_array = data.numpy()

    # 降维 此时数据为（1，256，256，3）
    data_array=data_array.squeeze(axis=0)

    #data_array = data[0]
    data_r = data_array[:, :, 2]
    # 获取特征点数目
    # 此处根据实际数据修改
    n_num = np.sum(data_r == -1)
    #u_num 已知点 已知点数目要不少于10
    u_num = data_r.size - n_num
    #u_num = data_r.shape[0] * data_r.shape[1] - n_num
    # 构建u_num*3数组存放特征点高程
    f_e = np.ones([u_num, 3], dtype=float)
    k = 0
    col = data_r.shape[1]
    row = data_r.shape[0]
    for i in range(col - 1):
        for j in range(row - 1):
            if data_r[i, j] != -1:
                f_e[k, 0] = i
                f_e[k, 1] = j
                f_e[k, 2] = data_r[i, j]
                k = k + 1

    # b = tf.convert_to_tensor(f_e)
    x_range = col
    y_range = row
    range_step = 1
    gridx = np.arange(0.0, x_range, range_step)  # 三个参数的意思：范围0.0 - 0.6 ，每隔0.1划分一个网格
    gridy = np.arange(0.0, y_range, range_step)
    ok3d = OrdinaryKriging(f_e[:, 1], f_e[:, 0], f_e[:, 2], variogram_model="spherical")  # 模型
    # variogram_model是变差函数模型，pykrige提供 linear, power, gaussian, spherical, exponential, hole-effect几种variogram_model可供选择，默认的为linear模型。
    # 使用不同的variogram_model，预测效果是不一样的，应该针对自己的任务选择合适的variogram_model。
    k3d1, ss3d = ok3d.execute("grid", gridx, gridy)  # k3d1是结果，给出了每个网格点处对应的值
    # 从掩码数组中提取array
    k3d1 = k3d1.__array__()
    #扩展维度（256，256，3）
    k3d1 = np.expand_dims(k3d1, axis=2)
    k3d1 = np.concatenate((k3d1, k3d1, k3d1), axis=2)
    k3d1 = tf.cast(k3d1, tf.float32)
    return k3d1



AUTOTUNE = tf.data.experimental.AUTOTUNE

#train_dataset = tf.data.Dataset.list_files(DATBASE_PATH + '/train_k/*.tif')
#train_dataset = train_dataset.map(lambda x: tf.py_function(preprocess, inp=[x], Tout= [tf.float32,tf.float32]))
#train_dataset = train_dataset.shuffle(BUFFER_SIZE)
#train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(DATBASE_PATH + '/test/*.tif', shuffle= False)
test_dataset = test_dataset.map(lambda x: tf.py_function(preprocess, inp=[x], Tout= [tf.float32,tf.float32]))
test_dataset = test_dataset.batch(BATCH_SIZE)




def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


# down_model = downsample(3, 4)
# down_result = down_model(tf.expand_dims(inp, 0))
# print (down_result.shape)


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()
#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)


# gen_output = generator(inp[tf.newaxis,...], training=False)
# plt.imshow(gen_output[0,...])


def generator_loss(disc_generated_output, gen_output, target):
    #w=target.shape[2]//2
    #target=target[:,:, w:, :]
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    #RMSE loss
    l1_loss = tf.reduce_mean(tf.square(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
#tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

# disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
# plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-6, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)
# IGPN_FCFE IGPN_RMSE IGPN_L1
checkpoint_dir = 'D:\\03 1129\\0428\\ckpt'
#checkpoint_dir = 'D:\\03 1129\\0428\\ckpt\\IGPN_L1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

if RESTORE_CKP:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))





def generate_images(model, test_input, tar, epoch):
    prediction = model(test_input, training=True)
    w=tar.shape[2]//2
    krig_tar=tar[:,:,:w,:]
    tar=tar[:,:,w:,:]
    #if tf.py_function(cal_knownp_num, inp=[test_input], Tout=[tf.float32])>tf.constant([10],tf.float32):
    #gen_output_k = tf.py_function(func=krig_generator, inp=[test_input], Tout=tf.float32)
    prediction=(prediction+krig_tar)/2
    fig = plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(OUTPUT_DIR + '/frame/image_at_epoch_{:04d}.png'.format(epoch), bbox_inches='tight', pad_inches=0)
    fig.clear()
    plt.close(fig)



#可视化生成效果 显示山脊 山谷线 测试模型结果
def generate_result_images(model, test_input, tar, epoch):
    prediction = model(test_input, training=True)
    w=tar.shape[2]//2
    #krig_tar=tar[:,:,:w,:]
    tar=tar[:,:,w:,:]
    #if tf.py_function(cal_knownp_num, inp=[test_input], Tout=[tf.float32])>tf.constant([10],tf.float32):
    gen_output_k = tf.py_function(func=krig_generator, inp=[test_input], Tout=tf.float32)
    prediction=(prediction+gen_output_k)/2
    fig = plt.figure(figsize=(15, 15))
    #input = np.zeros((256,256,3))
    #test_input_tep=test_input[0]
    #input[:,:,0]=test_input_tep[:,:,0]*255
    #input[:, :, 1] = test_input_tep[:, :, 1] * 255
    #input[:, :, 2] = test_input_tep[:, :, 2]
    display_list = [input, tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(OUTPUT_DIR + '/frame/image_at_epoch_{:04d}.png'.format(epoch), bbox_inches='tight', pad_inches=0)
    fig.clear()
    plt.close(fig)
#########################开始测试####################

model = generator

def view_para(model, test_input, tar):
    prediction = model(test_input, training=True)
    pre_tep = tf.expand_dims(prediction[:, :, :, 2], 3)
    pre_3 = tf.concat([pre_tep, pre_tep, pre_tep], 3)
    w = tar.shape[2] // 2
    krig_tar = tar[:, :, :w, :]
    tar = tar[:, :, w:, :]

    # gen_output_k 克里金插值结果，默认不开 提升性能
    # gen_output_k = tf.py_function(func=krig_generator, inp=[test_input], Tout=tf.float32)
    # prediction=(pre_3+gen_output_k)/2

    prediction1 = (pre_3 + krig_tar) / 2

    RMSE_kri = tf.sqrt(tf.reduce_mean(tf.square(tar - krig_tar)))
    # RMSE_pre3 = tf.sqrt(tf.reduce_mean(tf.square(tar - pre_3)))
    RMSE_loss1 = tf.sqrt(tf.reduce_mean(tf.square(tar - prediction1)))
    # RMSE_loss = tf.sqrt(tf.reduce_mean(tf.square(tar-prediction)))
    # EU_dis= tf.reduce_mean(tf.abs(tar-prediction))
    EU_dis1 = tf.reduce_mean(tf.abs(tar - prediction1))
    # EU_pre3 = tf.reduce_mean(tf.abs(tar - pre_3))
    EU_kri = tf.reduce_mean(tf.abs(tar - krig_tar))
    return RMSE_kri, RMSE_loss1, EU_kri, EU_dis1
############模型测试###########
#最终计算出的RMSE 和ME 均需要乘以 （2335-304）/（2*1000）
for n, (test_input, tar) in test_dataset.enumerate():
    if n==3:
        prediction = model(test_input, training=True)
        pre_tep = tf.expand_dims(prediction[:,:,:,2],3)
        pre_3=tf.concat([pre_tep, pre_tep, pre_tep], 3)
        w = tar.shape[2] // 2
        krig_tar=tar[:,:,:w,:]
        tar=tar[:,:,w:,:]


        #gen_output_k 克里金插值结果，默认不开 提升性能
        gen_output_k = tf.py_function(func=krig_generator, inp=[test_input], Tout=tf.float32)
        #prediction=(pre_3+gen_output_k)/2


        prediction1=(pre_3+krig_tar)/2
        #通过tifffile写出
        prediction2= (prediction1*0.5+0.5)*(2335-304)
        prediction3=prediction2[0,:,:,0].numpy()
        tiff.imwrite('D:/03 1129/0428/exp_epo/IGPN_L1_R4.tif',prediction3)

        #Groundtruth
        #tar1=tar[0,:,:,0].numpy()
        #tar2 = (tar1 * 0.5 + 0.5) * (2335 - 304)
        #tiff.imwrite('D:/03 1129/0428/exp_epo/R4_tar.tif', tar2)

        #RMSE_kri = tf.sqrt(tf.reduce_mean(tf.square(tar-krig_tar)))*(2335-304)/2
        #RMSE_pre3 = tf.sqrt(tf.reduce_mean(tf.square(tar - pre_3)))
        RMSE_loss1 = tf.sqrt(tf.reduce_mean(tf.square(tar-prediction1)))*(2335-304)/2
        #RMSE_loss = tf.sqrt(tf.reduce_mean(tf.square(tar-prediction)))
        #EU_dis= tf.reduce_mean(tf.abs(tar-prediction))
        EU_dis1 = tf.reduce_mean(tf.abs(tar - prediction1))*(2335-304)/2
        #EU_pre3 = tf.reduce_mean(tf.abs(tar - pre_3))


        #EU_kri = tf.reduce_mean(tf.abs(tar - krig_tar))*(2335-304)/2
        #恢复高程
        #pre_o=(pre_tep*0.5+0.5)*(2335-304)

        fig = plt.figure(figsize=(15, 15))
        plt.imshow(prediction1[0] * 0.5 + 0.5)
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(pre_3[0] * 0.5 + 0.5)
        #plt.imshow(tar[0] * 0.5 + 0.5)
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(krig_tar[0] * 0.5 + 0.5)
        #fig = plt.figure(figsize=(15, 15))
        #plt.imshow(pre_3[0] * 0.5 + 0.5)

###################################################

########################kri method###############
for n, (test_input, tar) in test_dataset.enumerate():
    if n==2:
        #prediction = model(test_input, training=True)
        #pre_tep = tf.expand_dims(prediction[:,:,:,2],3)
        #pre_3=tf.concat([pre_tep, pre_tep, pre_tep], 3)
        w = tar.shape[2] // 2
        krig_tar=tar[:,:,:w,:]
        tar=tar[:,:,w:,:]


        #gen_output_k 克里金插值结果，默认不开 提升性能
        #gen_output_k = tf.py_function(func=krig_generator, inp=[test_input], Tout=tf.float32)
        #prediction=(pre_3+gen_output_k)/2


        #prediction1=(pre_3+krig_tar)/2
        #通过tifffile写出
        prediction2= (krig_tar*0.5+0.5)*(2335-304)
        prediction3=prediction2[0,:,:,0].numpy()
        tiff.imwrite('D:/03 1129/0428/exp_epo/KRI_R3.tif',prediction3)

        #tar1=tar[0,:,:,0].numpy()
        #tar2 = (tar1 * 0.5 + 0.5) * (2335 - 304)
        #tiff.imwrite('D:/03 1129/0428/exp_epo/R4_tar.tif', tar2)

        RMSE_kri = tf.sqrt(tf.reduce_mean(tf.square(tar-krig_tar)))
        #RMSE_pre3 = tf.sqrt(tf.reduce_mean(tf.square(tar - pre_3)))
        #RMSE_loss1 = tf.sqrt(tf.reduce_mean(tf.square(tar-prediction1)))
        #RMSE_loss = tf.sqrt(tf.reduce_mean(tf.square(tar-prediction)))
        #EU_dis= tf.reduce_mean(tf.abs(tar-prediction))
        #EU_dis1 = tf.reduce_mean(tf.abs(tar - prediction1))
        #EU_pre3 = tf.reduce_mean(tf.abs(tar - pre_3))
        EU_kri = tf.reduce_mean(tf.abs(tar - krig_tar))
        #恢复高程
        #pre_o=(pre_tep*0.5+0.5)*(2335-304)

        #fig = plt.figure(figsize=(15, 15))
        #plt.imshow(prediction1[0] * 0.5 + 0.5)
        #fig = plt.figure(figsize=(15, 15))
        #plt.imshow(pre_3[0] * 0.5 + 0.5)
        #plt.imshow(tar[0] * 0.5 + 0.5)
        #fig = plt.figure(figsize=(15, 15))
        #plt.imshow(krig_tar[0] * 0.5 + 0.5)
        #fig = plt.figure(figsize=(15, 15))
        #plt.imshow(pre_3[0] * 0.5 + 0.5)

###################################################





####################模型讨论1 骨架线影响###########
for n, (test_input, tar) in test_dataset.enumerate():
    if n==0:
        #创建-1张量
        zero_tensor = tf.zeros([1, 256, 256, 1])-1
        #w_rd w_vl w_f 分别是 去除山脊线 去除山谷线 全部去除
        w_rd = tf.concat([zero_tensor, test_input[:, :, :, 1:2], test_input[:, :, :, 2:3]], 3)
        w_vl = tf.concat([test_input[:, :, :, 0:1],zero_tensor, test_input[:, :, :, 2:3]], 3)
        w_f = tf.concat([zero_tensor, zero_tensor, test_input[:, :, :, 2:3]], 3)
        prediction_w_rd = model(w_rd, training=True)
        prediction_w_vl = model(w_vl, training=True)
        prediction_w_f = model(w_f, training=True)
        pre_tep_wrd = tf.expand_dims(prediction_w_rd[:,:,:,2],3)
        pre_tep_wvl = tf.expand_dims(prediction_w_vl[:,:,:,2],3)
        pre_tep_wf = tf.expand_dims(prediction_w_f[:, :, :, 2], 3)

        pre_3_wrd=tf.concat([pre_tep_wrd, pre_tep_wrd, pre_tep_wrd], 3)
        pre_3_wvl = tf.concat([pre_tep_wvl, pre_tep_wvl, pre_tep_wvl], 3)
        pre_3_wf = tf.concat([pre_tep_wf, pre_tep_wf, pre_tep_wf], 3)

        w = tar.shape[2] // 2
        krig_tar=tar[:,:,:w,:]
        tar=tar[:,:,w:,:]


        #gen_output_k 克里金插值结果，默认不开 提升性能
        #gen_output_k = tf.py_function(func=krig_generator, inp=[test_input], Tout=tf.float32)
        #prediction=(pre_3+gen_output_k)/2


        prediction1_wrd=(pre_3_wrd+krig_tar)/2
        prediction1_wvl = (pre_3_wvl + krig_tar) / 2
        prediction1_wf = (pre_3_wf + krig_tar) / 2
        #通过tifffile写出
        prediction2_wrd= (prediction1_wrd*0.5+0.5)*(2335-304)
        prediction2_wvl = (prediction1_wvl * 0.5 + 0.5) * (2335 - 304)
        prediction2_wf = (prediction1_wf * 0.5 + 0.5) * (2335 - 304)
        prediction3_wrd=prediction2_wrd[0,:,:,0].numpy()
        prediction3_wvl = prediction2_wvl[0, :, :, 0].numpy()
        prediction3_wf = prediction2_wf[0, :, :, 0].numpy()


        tiff.imwrite('D:/03 1129/0428/exp_epo/IGPN_R1_wrd.tif',prediction3_wrd)
        tiff.imwrite('D:/03 1129/0428/exp_epo/IGPN_R1_wvl.tif', prediction3_wvl)
        tiff.imwrite('D:/03 1129/0428/exp_epo/IGPN_R1_wf.tif', prediction3_wf)

        #tar1=tar[0,:,:,0].numpy()
        #tar2 = (tar1 * 0.5 + 0.5) * (2335 - 304)
        #tiff.imwrite('D:/03 1129/0428/exp_epo/R4_tar.tif', tar2)

        #RMSE_kri = tf.sqrt(tf.reduce_mean(tf.square(tar-krig_tar)))
        #RMSE_pre3 = tf.sqrt(tf.reduce_mean(tf.square(tar - pre_3)))
        RMSE_loss1_wrd = tf.sqrt(tf.reduce_mean(tf.square(tar-prediction1_wrd)))
        RMSE_loss1_wvl = tf.sqrt(tf.reduce_mean(tf.square(tar - prediction1_wvl)))
        RMSE_loss1_wf = tf.sqrt(tf.reduce_mean(tf.square(tar - prediction1_wf)))
        #RMSE_loss = tf.sqrt(tf.reduce_mean(tf.square(tar-prediction)))
        #EU_dis= tf.reduce_mean(tf.abs(tar-prediction))
        EU_dis1_wrd = tf.reduce_mean(tf.abs(tar - prediction1_wrd))
        EU_dis1_wvl = tf.reduce_mean(tf.abs(tar - prediction1_wvl))
        EU_dis1_wf = tf.reduce_mean(tf.abs(tar - prediction1_wf))

        #EU_pre3 = tf.reduce_mean(tf.abs(tar - pre_3))
        #EU_kri = tf.reduce_mean(tf.abs(tar - krig_tar))
        #恢复高程
        #pre_o=(pre_tep*0.5+0.5)*(2335-304)

        #fig = plt.figure(figsize=(15, 15))
        #plt.imshow(prediction1[0] * 0.5 + 0.5)
        #fig = plt.figure(figsize=(15, 15))
        #plt.imshow(pre_3[0] * 0.5 + 0.5)
        #plt.imshow(tar[0] * 0.5 + 0.5)
        #fig = plt.figure(figsize=(15, 15))
        #plt.imshow(krig_tar[0] * 0.5 + 0.5)
        #fig = plt.figure(figsize=(15, 15))
        #plt.imshow(pre_3[0] * 0.5 + 0.5)
###########################################























#泛化性测试########################################################
a1, b1, c1 = run.read_img(DATBASE_PATH + '/ASTGTM2_N26E105_dem.tif')
c2 = c1[:256, :256]
c3 = np.zeros([256, 256])
c32 = np.ones([10,1])*(-0.3)
c31=tf.cast(c3-1,tf.float32)
c4 = c2 - 304
for i in range(c2.shape[0]):
    for j in range(c2.shape[1]):
        if i % 11 == 0:
            if j % 11 == 0:
                c3[i, j] = c4[i, j]
c5 = (c3 / ((2335 - 304) / 2)) - 1
c51 = (c3 / ((2335 - 304) / 2)) - 1
for i in range(c32.size):
    c5[8*i,17*i]=c32[i]

c6 = tf.cast(c51, tf.float32)
c7 = tf.expand_dims(c6, 2)
c71= tf.expand_dims(c31, 2)
c8 = tf.concat([c71, c71, c7], 2)
c9 = tf.expand_dims(c8, 0)
prediction = model(c9, training=True)

pre_tep = tf.expand_dims(prediction[:, :, :, 2], 3)
pre_3 = tf.concat([pre_tep, pre_tep, pre_tep], 3)
#gen_output_k = tf.py_function(func=krig_generator, inp=[c9], Tout=tf.float32)
#prediction1=(pre_3+gen_output_k)/2
fig = plt.figure(figsize=(15, 15))
plt.imshow(pre_3[0] * 0.5 + 0.5)
#fig = plt.figure(figsize=(15, 15))
#plt.imshow(prediction1[0] * 0.5 + 0.5)