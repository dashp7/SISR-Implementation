import tensorflow as tf

import os
import math
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory

from IPython.display import display


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL
import time as t

showplot = 0
res_fraction = 1

def define_datasets(train_dir, batch_size, crop_size, val_split):
	ds_train = image_dataset_from_directory(
	    str(train_dir),
	    batch_size=batch_size,
	    image_size=(crop_size, crop_size),
	    validation_split=val_split,
	    subset="training",
	    seed=1875,
	    label_mode=None,
	)

	ds_valid = image_dataset_from_directory(
	    str(train_dir),
	    batch_size=batch_size,
	    image_size=(crop_size, crop_size),
	    validation_split=val_split,
	    subset="validation",
	    seed=1875,
	    label_mode=None,
	)

	return ds_train, ds_valid

def imgscale(inp_img):
    inp_img = inp_img / 255.0
    return inp_img

def map_datasets(ds_train, ds_valid):
	ds_train = ds_train.map(imgscale)
	ds_valid = ds_valid.map(imgscale)
	return ds_train, ds_valid


def generate_test_paths(test_dir):
	test_path =os.path.join(test_dir,"test")
	testimg_locs = sorted(
	    [
	        os.path.join(test_path, fname)
	        for fname in os.listdir(test_path)
	        if fname.endswith(".jpg")
	    ]
	)
	return testimg_locs

def inp_transform(input, input_size, sr_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dim = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dim)
    return tf.image.resize(y, [input_size, input_size], method="area")

def op_transform(input):
    input = tf.image.rgb_to_yuv(input)
    last_dim = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dim)
    return y

def transform_ds(ds_train, ds_valid, input_size, sr_factor):
	ds_train = ds_train.map(lambda x: (inp_transform(x, input_size, sr_factor), op_transform(x)))
	ds_train = ds_train.prefetch(buffer_size=32)
	ds_valid = ds_valid.map(lambda x: (inp_transform(x, input_size, sr_factor), op_transform(x)))
	ds_valid = ds_valid.prefetch(buffer_size=32)

	return ds_train, ds_valid

plt.rcParams.update({'lines.linewidth':3})
plt.rcParams.update({'font.size': 20})
def plot_results(img, prefix, title):
	img_array = img_to_array(img)
	img_array = img_array.astype("float32") / 255.0

	fig, ax = plt.subplots(1,1, figsize=(25.6*res_fraction,14.4*res_fraction)) #Resizing to res_fraction*100% of 2K resolution(2560x1440)
	im = ax.imshow(img_array[::-1], origin="lower")
	plt.title(title)
	axins = zoomed_inset_axes(ax, 2, loc=2)
	axins.imshow(img_array[::-1], origin="lower")

	# Specify the limits.
	x1, x2, y1, y2 = 950, 1250, 300, 600
	# Apply the x-limits.
	axins.set_xlim(x1, x2)
	# Apply the y-limits.
	axins.set_ylim(y1, y2)

	plt.yticks(visible=False)
	plt.xticks(visible=False)

	# Make the line.
	base = os.getcwd()
	print(base)
	mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
	plt.tight_layout()
	plt.savefig(os.path.join(base, str(prefix) + "-" + title + ".png"))
	if showplot != 0:
		plt.show()

def get_LR_image(img, upscale_factor):
    """Returns a low-res image that will be used as model input"""
    return img.resize((img.size[0] // upscale_factor, img.size[1] // upscale_factor),PIL.Image.BICUBIC)


def upscale_image(model, img):
    """Upscales the given image and saves it in the Red_Green_Blue format"""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert("RGB")
    return out_img


class SRCallback(keras.callbacks.Callback):
    def __init__(self, test_paths, sr_factor, model_name):
        super(SRCallback, self).__init__()
        self.test_img = get_LR_image(load_img(test_paths[72]), sr_factor)
        self.model_name = model_name

    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Average value of PSNR for current epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 50 == 0:
            prediction = upscale_image(self.model, self.test_img)
            plot_results(prediction, "Epoch-" + str(epoch), "Prediction_"+str(self.model_name))

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(-10 * math.log10(logs["loss"]))



def inference(model, test_paths, sr_factor, model_name):
	total_bicubic_psnr = 0.0
	total_bicubic_ssim = 0.0
	total_test_psnr = 0.0
	total_test_ssim = 0.0
	inf_time = 0.0

	for index, test_img_path in enumerate(test_paths):
	    
	    num = len(test_paths)
	    img = load_img(test_img_path)
	    lowres_input = get_LR_image(img, sr_factor)
	    w = lowres_input.size[0] * sr_factor
	    h = lowres_input.size[1] * sr_factor
	    HR_img = img.resize((w, h))
	    t_start=t.time()
	    prediction = upscale_image(model, lowres_input)
	    t_end=t.time()

	    LR_img = lowres_input.resize((w, h))
	    LR_img_arr = img_to_array(LR_img)
	    HR_img_arr = img_to_array(HR_img)
	    predict_img_arr = img_to_array(prediction)
	    bicubic_psnr = tf.image.psnr(LR_img_arr, HR_img_arr, max_val=255)
	    bicubic_ssim = tf.image.ssim(LR_img_arr, HR_img_arr, max_val=255)
	    test_psnr = tf.image.psnr(predict_img_arr, HR_img_arr, max_val=255)
	    test_ssim = tf.image.ssim(predict_img_arr, HR_img_arr, max_val=255)

	    total_bicubic_psnr += bicubic_psnr
	    total_bicubic_ssim += bicubic_ssim
	    total_test_psnr += test_psnr
	    total_test_ssim += test_ssim
	    inf_time += t_end - t_start



	    if index ==85:
		    print("PSNR of LR and HR image: %.5f" % bicubic_psnr)
		    print("SSIM of LR and HR image: %.5f" % bicubic_ssim)
		    print("PSNR of Predicted and HR image: %.5f" % test_psnr)
		    print("SSIM of Predicted and HR image: %.5f" % test_ssim)
		    plot_results(LR_img, index, model_name + "_lowres")
		    plot_results(HR_img, index, model_name + "_highres")
		    plot_results(prediction, index, model_name + "_prediction")

	print("Avg. PSNR of LR and HR images is %.4f" % (total_bicubic_psnr / num))
	print("Avg. SSIM of LR and HR images is %.4f" % (total_bicubic_ssim / num))
	print("Avg. PSNR of Predictions and HR images is %.4f" % (total_test_psnr / num))
	print("Avg. SSIM of Predictions and HR images is %.4f" % (total_test_ssim / num))
	print("Avg. Inference Time is %.4f s" % (inf_time / num))


def training_stats(r, model_name):
	plt.rcParams.update({'lines.linewidth':3})
	plt.rcParams.update({'font.size': 18})
	fig, axs = plt.subplots(1,1, figsize=(10,7.5))
	y = r.history['loss']
	z = r.history['val_loss']
	axs.plot(y[1:], label='Loss')
	axs.plot(z[1:], label='Validation Loss')
	plt.legend()
	axs.set_title("Training Curves for "+ model_name)
	axs.set_xlabel("Number of Epochs")
	axs.set_ylabel("Losses")
	axs.grid('True')
	plt.tight_layout()
	plt.savefig(model_name+'_train.png')