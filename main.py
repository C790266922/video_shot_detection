import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

'''
load image sequence from video
'''
def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if frame is not None:
            frames.append(frame)
        else:
            break

    cap.release()

    return frames

'''
calculate color histogram and score
'''
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

    return score, hists

'''
threshold hist score to find key frames
'''
def threshold_hist_score(score, threshold = 10.0):
    key_frames = []
    for i, s in enumerate(score):
        if s > threshold:
            key_frames.append(i)

    print(key_frames)
    return key_frames


if __name__ == "__main__":
    # load frames
    video_path = './movie.mp4'
    frames = get_frames(video_path)
    # calculate Hist score
    score, hists = get_hist_score(frames)
    # plot score
    plt.figure(figsize = (8, 6))
    plt.plot(score)
    plt.savefig('score.png')
    # threshold hist score
    key_frames = threshold_hist_score(score)
