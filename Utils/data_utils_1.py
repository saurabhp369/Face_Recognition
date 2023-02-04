#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : data utils for task 1
#=======================================================

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data():
    mat = scipy.io.loadmat('Data/data.mat')
    mat1 = scipy.io.loadmat('Data/illumination.mat')
    mat2 = scipy.io.loadmat('Data/pose.mat')
    faces = np.array(mat['face'])
    illumination = np.array(mat1['illum'])
    poses = np.array(mat2['pose'])

    return faces,illumination, poses

def create_labels():
    face_labels = []
    illum_labels = []
    pose_labels = []
    for i in range(1, 201):
        label = [i]*3
        face_labels = face_labels + label
    face_labels = np.array(face_labels)

    for i in range(1, 69):
        label = [i]*13
        pose_labels = pose_labels + label
    pose_labels = np.array(pose_labels)

    for i in range(1,69):
        label = [i]*21
        illum_labels = illum_labels + label
    illum_labels = np.array(illum_labels)

    return face_labels, illum_labels, pose_labels

def train_test_split():
    train_ind = []
    test_ind = []

    for i in range(200):
        rand_nums = random.sample(range(0, 3), 2)
        train_ind.append(i*3+rand_nums[0])
        train_ind.append(i*3+rand_nums[1])

    for i in range(600):
        if i not in train_ind:
            test_ind.append(i)

    return train_ind, test_ind

def visualise_data(x,y):
    f = plt.figure()
    ax1 = f.add_subplot(1,2,1)
    plt.imshow(x.reshape((24,21)).squeeze(), cmap = 'gray')
    ax1.axis(False)
    ax2 = f.add_subplot(1,2,2)
    ax2.axis(False)
    plt.imshow(y.reshape((24,21)).squeeze(), cmap = 'gray')
    ax1.title.set_text('Original Image')
    ax2.title.set_text('Compressed Image')
    plt.show(block=True)
    
