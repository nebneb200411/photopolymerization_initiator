import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import datetime
import glob
import re
import cv2 as cv
import shutil

def plot_3d(x, y, z, figsize, save_dir, special_plot=None, fontsize=18, to_movie=False):
    """
    【目的】
    3Dプロットの作成と3Dプロットの動画作成が可能
    x, y, zはlist型で入力
    special_plotは強調したいプロットを入力する
    【注意】
    ・special_plot
        special_plotは(強調したいプロットの数, 3)のshapeになるように入力
        [[x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn]]
        ※ 長さが8以上だとエラー
    ・to_movie
        動画にするか選べる.defaultがFalseなので動画にしたい時はTrueにする
    ・figsize
        タプルで入力. exp)(15, 15)
    """
    assert type(x) == list and type(y) == list and type(z) == list, 'You have to input x, y and z as for list type!!'
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    x_max = max(x)
    x_min = min(x)
    x_range = max(abs(x_max), abs(x_min))
    y_max = max(y)
    y_min = min(y)
    y_range = max(abs(y_max), abs(y_min))
    z_max = max(z)
    z_min = min(z)
    z_range = max(abs(z_max), abs(z_min))

    x_step = (x_max - x_min)//10
    y_step = (y_max - y_min)//10
    z_step = (z_max - z_min)//10

    ax.set_xticks(np.arange(-x_range, x_range, x_step), fontsize=fontsize)
    ax.set_yticks(np.arange(-y_range, y_range, y_step), fontsize=fontsize)
    ax.set_yticks(np.arange(-z_range, z_range, z_step), fontsize=fontsize)

    ax.scatter(x, y, z, s=50)

    for i, plot in enumerate(special_plot):
        x_s = plot[0]
        y_s = plot[1]
        z_s = plot[2]
        color = COLOR_LIST[i]
        print(i+1, 'の要素は{}'.format(color), '色となります')
        ax.scatter(x_s, y_s, z_s, s=500)
    
    plt.legend()
    plt.tick_params(labelsize=fontsize-2)

    now_datetime = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")

    save_path = os.path.join(save_dir, '{}_3dplot.png'.format(now_datetime))
    fig.savefig(save_path)

    if to_movie:
        movie_cache_dir = os.path.join(save_dir, 'movie_cache')
        os.makedirs(movie_cache_dir, exist_ok=True)
        for angle in range(0, 360):
            ax.view_init(elev=15, azim=angle)
            if angle % 1 == 0:
                filename = 'fig_' + str(angle) + ".png"
                fig.savefig(os.path.join(movie_cache_dir, filename))
        
        images = glob.glob(os.path.join(movie_cache_dir, '*.png'))
        #images = sorted(images, key=lambda x:int((re.search(r"[0-9]+", x)).group(0)))
        images = sorted(images, key=lambda s: int(re.search(r'\d+', s).group()))
        pictures = []
        for img in images:
            img = cv.imread(img)
            h, w, c = img.shape
            size = (w, h)
            pictures.append(img)

        movie_save_path = os.path.join(save_dir, 'graph_rotation_180speed.mp4')
        movie = cv.VideoWriter(movie_save_path, cv.VideoWriter_fourcc(*'MP4V'), 40.0, size)
        for i in range(len(pictures)):
            movie.write(pictures[i])
        shutil.rmtree(movie_cache_dir)


COLOR_LIST = [
    'b',
    'g',
    'r',
    'c',
    'm',
    'y',
    'k',
    'w'
]

def horizontal_bar(label, val, xlabel, bar_height=0.25, figisize=(50, 16), color='turquoise', save_path=None):
    sns.set()
    fig = plt.figure(figsize=figisize)
    ax = fig.add_subplot(1,1,1)

    ax.barh(label, val, height=bar_height, color=color, label='Liv Pass Backward')
    ax.set_yticks(label)
    ax.set_xlabel(xlabel)

    if save_path:
        fig.savefig(save_path)


