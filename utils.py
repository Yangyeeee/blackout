import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import numpy as np
import torch

#_, term_width = os.popen('stty size', 'r').read().split()
term_width = 80 #int(term_width)

last_time = time.time()
begin_time = last_time
TOTAL_BAR_LENGTH = 65.


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_word2vec(word_counter):
    glove_path = "../../data/glove/glove.6B.100d.txt"
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes["6B"]
    word2vec_dict = {}
    word_counter = list(word_counter)

    with open(glove_path, 'r') as fh:
        for line in fh:
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector
            elif word == "maple":
                print(word)
                word2vec_dict["maple_tree"] = vector
            elif word == "palm":
                print(word)
                word2vec_dict["palm_tree"] = vector
            elif word == "pine":
                print(word)
                word2vec_dict["pine_tree"] = vector
            elif word == "willow":
                print(word)
                word2vec_dict["willow_tree"] = vector
            elif word == "oak":
                print(word)
                word2vec_dict["oak_tree"] = vector
            elif word == "mower":
                print(word)
                word2vec_dict["lawn_mower"] = vector
            elif word == "pepper":
                print(word)
                word2vec_dict["sweet_pepper"] = vector
            elif word == "fish":
                print(word)
                word2vec_dict["aquarium_fish"] = vector
            elif word == "truck":
                print(word)
                word2vec_dict["pickup_truck"] = vector


    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def generate_p_cifar100(p=0.5):
    fine2coarse = np.load("fine2coarse.npy")
    coarse2fine = np.array([
        [4, 30, 55, 72, 95],
        [1, 32, 67, 73, 91],
        [54, 62, 70, 82, 92],
        [9, 10, 16, 28, 61],
        [0, 51, 53, 57, 83],
        [22, 39, 40, 86, 87],
        [5, 20, 25, 84, 94],
        [6, 7, 14, 18, 24],
        [3, 42, 43, 88, 97],
        [12, 17, 37, 68, 76],
        [23, 33, 49, 60, 71],
        [15, 19, 21, 31, 38],
        [34, 63, 64, 66, 75],
        [26, 45, 77, 79, 99],
        [2, 11, 35, 46, 98],
        [27, 29, 44, 78, 93],
        [36, 50, 65, 74, 80],
        [47, 52, 56, 59, 96],
        [8, 13, 48, 58, 90],
        [41, 69, 81, 85, 89]])

    p_mat = np.ones((100, 100), dtype=np.single)
    q = (1-p)/95
    p = p/5
    p_mat *=  q
    for i in range(100):
        superclass = fine2coarse[i]
        sameclass = coarse2fine[superclass]
        p_mat[i][sameclass] = p

    # for debugging
    #classname = np.array(
    #    ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
    #     'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
    #     'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    #     'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
    #     'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
    #     'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
    #     'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
    #     'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
    #     'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    #     'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'])

    #for i in range(5):
    #    print(classname[coarse2fine[i]])
    #    print(p_mat[coarse2fine[i][0]][coarse2fine[i]])
    #    print(classname[coarse2fine[10]])
    #    print(p_mat[coarse2fine[i][0]][coarse2fine[10]])

    return p_mat
