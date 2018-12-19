import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def get_frames(folder):
    # ignore hidden files
    img_list = [x for x in os.listdir(folder) if not x.startswith('.')]
    # sort
    img_list = [folder + '/' + name for name in \
            sorted(img_list, key = lambda x: int(x[:-4]))]
    # load imgs
    frames = []
    for filename in img_list:
        frame = cv2.imread(filename)
        if frame is not None:
            frames.append(frame)

    return frames

def get_hist_score(frames):
    hists = []
    score = []

    for frame in frames:
        # R G B hist
        hist = [cv2.calcHist([frame], [i], None, [64], [0.0, 256.0]) for i in range(3)]
        # normalize hist
        hist = [h / np.sum(h) for h in hist]
        hists.append(np.array(hist).flatten())

    # calculate score
    for pair in zip(hists[1:], hists[:-1]):
        s = cv2.compareHist(pair[0], pair[1], cv2.HISTCMP_CHISQR)
        score.append(s)

    # plot score
    plt.figure(figsize = (8, 6))
    plt.plot(score)
    plt.savefig('score.png')

    return score, hists

def threshold_hist_score(score, threshold = 10.0):
    key_frames = []
    for i, s in enumerate(score):
        if s > threshold:
            key_frames.append(i)

    print(key_frames)
    return key_frames


if __name__ == "__main__":

    # load frames
    folder = './pics'
    frames = get_frames(folder)
    # calculate Hist score
    score, hists = get_hist_score(frames)
    # threshold hist score
    key_frames = threshold_hist_score(score)
