import pickle
from scipy.ndimage.measurements import label
from utils import *

class Pipeline():
    def __init__(self):
        # load a pe-trained svc model from a serialized (pickle) file
        dist_pickle = pickle.load(open("svc_model_32_sqrt.p", "rb"))

        # get attributes of our svc object
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle["scaler"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.hist_bins = dist_pickle["hist_bins"]

        self.frame_cnt = 1

    def detect_cars(self, image):
        # Limit the are of the sliding window search to the horizon.
        # The scale makes the posterior window smaller to account for  the perspective.
        ystart = 400
        ystop = 550
        scale = 1.
        bbox_list = self._find_cars(image, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                                    self.cell_per_block, self.spatial_size, self.hist_bins)

        # The scale makes the front window larger to account for to the perspective
        ystart_front = 551
        ystop_front = 656
        scale_front = 1.5
        bbox_list_front = self._find_cars(image, ystart_front, ystop_front, scale_front, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                                    self.cell_per_block, self.spatial_size, self.hist_bins)

        bbox_list += bbox_list_front

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, bbox_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        labels = label(heatmap)

        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        # cv2.putText(draw_img, 'Frame:   ' + str(self.frame_cnt), (300, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        #             (255, 255, 255), 2,
        #             cv2.LINE_AA)
        # mpimg.imsave(dir_out + 'frame' + str(frame_cnt[0]) + '.jpg', image, format='jpeg')
        # mpimg.imsave(dir_marked + 'frame' + str(frame_cnt[0]) + '.jpg', draw_img, format='jpeg')
        self.frame_cnt += 1

        return draw_img



    # L35
    # Define a single function that can extract features using hog sub-sampling and make predictions
    def _find_cars(self, img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                  hist_bins):
        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv_color_space='YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        box_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    box_list.append(
                        ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

        return box_list


