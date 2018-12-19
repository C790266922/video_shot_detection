import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse

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

'''
calculate moment invariants feature difference between frames
'''
def get_mom_diff(frames):
    mom_features = []
    # traverse all frame and calculate moment feature
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mom = cv2.moments(gray)
        n20 = mom['nu20']
        n02 = mom['nu02']
        n11 = mom['nu11']
        n30 = mom['nu30']
        n03 = mom['nu03']
        n12 = mom['nu12']
        n21 = mom['nu21']

        # use 3 moment invarants to form a feature vector
        f1 = n20 + n02
        f2 = (n20 - n02) ** 2 + 4 * n11 ** 2
        f3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
        feat = [f1, f2, f3]

        mom_features.append(np.array(feat))

    # calculate feature difference between frames
    mom_diffs = []
    for i in range(1, len(mom_features)):
        diff = np.sum((mom_features[i] - mom_features[i - 1]) ** 2) 
        mom_diffs.append(diff)

    return mom_diffs

'''
threshold moment feature differences to get key frames
'''
def threshold_mom_diff(mom_diffs, threshold = 1.0):
    key_frames = []

    for i, diff in enumerate(mom_diffs):
        if diff > threshold:
            key_frames.append(i)

    return key_frames

'''
use color histogram to get shot (start, end) frames
'''
def get_shot_hist(frames):
    # calculate hist score
    score, hists = get_hist_score(frames)
    # plot score
    plt.figure(figsize = (8, 6))
    plt.plot(score)
    plt.savefig('score.png')
    # threshold score
    shots = threshold_hist_score(score)
    return shots

'''
use moment features to get shot (start, end) frames
'''
def get_shot_mom(frames):
    # calculate moment feature differences
    mom_diffs = get_mom_diff(frames)
    print(mom_diffs)
    print(np.mean(mom_diffs))
    print(min(mom_diffs), max(mom_diffs))
    # threshold diff
    shots = threshold_mom_diff(mom_diffs)
    print(shots)

    return shots

if __name__ == "__main__":
    # load frames
    video_path = './movie.mp4'
    frames = get_frames(video_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--hist', action = 'store_true')
    parser.add_argument('--moment', action = 'store_true')
    parser.add_argument('--mix')

    args = parser.parse_args()

    # use color histogram to identify key frames
    if args.hist:
        shots = get_shot_hist(frames)

    # use moment features to identify key frames
    elif args.moment:
        shots = get_shot_mom(frames)

    # use both algorithm to get 2 key frames sets, then union or intersect
    elif args.mix:
        shots_hist = get_shot_hist(frames)
        shots_mom = get_shot_mom(frames)
        
        shots = set()
        # union
        if args.mix == 1:
            shots = set(shots_hist) | set(shots_mom)
        # intersect
        elif args.mix == 2:
            shots = set(shots_hist) & set(shots_mom)
        else:
            printf("Wrong argument for --mix (1 or 2)")
            exit()
