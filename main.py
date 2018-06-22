# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

from vehicle_detection_pipeline import *

"""
Process movies
"""



def process_image(image):
    """Process movie image"""
    return Pipeline().detect_cars(image)


def process_movie(movie, movie_out):
    """
    Process movie image
    :param movie: movie
    :param movie_out: movie output
    :return:
    """
    clip = VideoFileClip(movie)
    mov_clip = clip.fl_image(process_image)  # NOTE: this function expects color images!!
    mov_clip.write_videofile(movie_out, audio=False)

process_movie('project_video.mp4', 'project_video_out.mp4')
