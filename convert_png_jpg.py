from PIL import Image
import glob
import os

dir_out = 'training_jpg/'
dir = 'training/'

images = glob.glob(dir + '*/*/*.png')

dir_len = len(dir)
png_len = len('png')
for png_img_path in images:
    im = Image.open(png_img_path)
    rgb_im = im.convert('RGB')

    jpg_img_path_out = dir_out + png_img_path[dir_len:]  #swap directory
    jpg_img_path_out = jpg_img_path_out[0:-png_len] + 'jpg'  #swap extension to jpg
    last_dir_idx = jpg_img_path_out.rfind('/')
    dir_tree = jpg_img_path_out[0:last_dir_idx]
    if not os.path.isdir(dir_tree):
        os.makedirs(dir_tree)
    rgb_im.save(jpg_img_path_out)
