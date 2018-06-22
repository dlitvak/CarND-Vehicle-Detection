import glob

from vehicle_detection_pipeline import *

pipeline = Pipeline()

# dir_out = 'debugging/'
# frames = np.arange(554, 560)
f_imgs = glob.glob('test_images/test1.jpg')
for f in f_imgs:
    image = mpimg.imread(f)
    draw_img = pipeline.detect_cars(image)

    mpimg.imsave(f.replace('/test', '/out_win_test'), draw_img)
    # mpimg.imsave(dir_out + f_name.replace('.jpg', '_heat.jpg'), heatmap, cmap=plt.get_cmap('hot'))
    # mpimg.imsave(dir_out + f_name.replace('.jpg', '_marked.jpg'), draw_img)