#纯净版
#减小数据体积 更换为16位tif
#改用tifffile读取 tif
#V5改进版 提升模型速度
#将kri移植进样本中
#
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

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100
CURRENT_EPOCH = 0;  # Set this if training stops in between.
RESTORE_CKP = False  # True For starting from checkpoint and For testing in interactive tool
DATBASE_PATH = r'/home/d211/data/Project/mxc'  # give database path here
OUTPUT_DIR = r'/home/d211/data/Project/mxc'
#Open_Interactive_Tool = False # set True to open interactive tool

CKP_SAVE_INT = 10  # chkpt interval
EPOCHS = 400 # iterations


def MKDIR(Dir):
    if not os.path.isdir(Dir):
        os.mkdir(Dir)


MKDIR(OUTPUT_DIR)
MKDIR(r'/home/d211/data/Project/mxc/Test')
MKDIR(OUTPUT_DIR + "/frame_0424")

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

train_dataset = tf.data.Dataset.list_files(DATBASE_PATH + '/data/clip_TIFF_B0424/train/*.tif')
train_dataset = train_dataset.map(lambda x: tf.py_function(preprocess, inp=[x], Tout= [tf.float32,tf.float32]))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(DATBASE_PATH + '/data/clip_TIFF_B0424/test/*.tif')
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
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)


# gen_output = generator(inp[tf.newaxis,...], training=False)
# plt.imshow(gen_output[0,...])


def generator_loss(disc_generated_output, gen_output, target):
    #w=target.shape[2]//2
    #target=target[:,:, w:, :]
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    #RMSE loss
    l1_loss = tf.sqrt(tf.reduce_mean(tf.square(target - gen_output)))

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
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

# disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
# plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-6, beta_1=0.5)

checkpoint_dir = r'/home/d211/data/Project/mxc/training_checkpoints_wk0427'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

if RESTORE_CKP:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))





def generate_images(model, test_input, tar, epoch):
    prediction = model(test_input, training=True)
    
    prediction_tep = tf.expand_dims(prediction[:,:,:,2],3)
    prediction1 = tf.concat([prediction_tep,prediction_tep,prediction_tep],3)
    
    w=tar.shape[2]//2
    #krig_tar=tar[:,:,:w,:]
    tar=tar[:,:,w:,:]
    #if tf.py_function(cal_knownp_num, inp=[test_input], Tout=[tf.float32])>tf.constant([10],tf.float32):
    #gen_output_k = tf.py_function(func=krig_generator, inp=[test_input], Tout=tf.float32)
    #prediction=(prediction1+krig_tar)/2
    prediction = prediction1
    fig = plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(OUTPUT_DIR + '/frame_0427_wk/image_at_epoch_{:04d}.png'.format(epoch), bbox_inches='tight', pad_inches=0)
    fig.clear()
    plt.close(fig)
#可视化生成效果 显示山脊 山谷线




for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target, 0)

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image,  target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        #替换成第三通道数值
        gen_output_tep = tf.expand_dims(gen_output[:,:,:,2],3)
        gen_output1 = tf.concat([gen_output_tep,gen_output_tep,gen_output_tep],3)
       
       
        w=target.shape[2]//2
        #krig_tar=target[:,:,:w,:]
        target=target[:,:,w:,:]
        #添加krig_generator
        #if cal_knownp_num(input_image)>10:
        #if tf.py_function(cal_knownp_num, inp=[input_image], Tout=[tf.float32])>tf.constant([10],tf.float32):
        #gen_output_k = tf.py_function(func=krig_generator, inp=[input_image], Tout=tf.float32)
        #gen_output_k=krig_generator(input_image)
        #gen_output=(gen_output1 + krig_tar)/2
        gen_output = gen_output1
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        #plt.plot(history.epoch, history.history.get('loss'))
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    #判别器能力太强  减少学习更新次数
    if epoch % 3 == 0:
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        # display.clear_output(wait=True)

        for example_input,  example_target in test_ds.take(1):
            generate_images(generator, example_input,  example_target, epoch + 1 + CURRENT_EPOCH)
        print("Epoch: ", epoch + 1 + CURRENT_EPOCH)

        # Train
        for n, (input_image,  target) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image,  target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % CKP_SAVE_INT == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1 + CURRENT_EPOCH,
                                                           time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)


with tf.device('/gpu:0'):
    fit(train_dataset, EPOCHS, test_dataset)
