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
threshold hist score to find cut frames
'''
def threshold_hist_score(score, threshold = 10.0, min_shot_size = 10):
    # use min_shot_size to avoid including too many fade/transition frames
    # initialize cuts to [0] to avoid cut[-1] index out of range error
    cuts = [0]
    for i, s in enumerate(score):
        if s > threshold and i - cuts[-1] > min_shot_size:
            cuts.append(i)

    # drop 0
    return cuts[1:]

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
threshold moment feature differences to get cut frames
'''
def threshold_mom_diff(mom_diffs, threshold = 0.1, min_shot_size = 10):
    # use min_shot_size to avoid including too many fade/transition frames
    # use 0 to aviod cuts[-1] index out of range
    cuts = [0]
    for i, diff in enumerate(mom_diffs):
        if diff > threshold and i - cuts[-1] > min_shot_size:
            cuts.append(i)

    # drop 0
    return cuts[1:]

'''
use color histogram to get cut frames
'''
def get_cuts_hist(frames):
    # calculate hist score
    score, hists = get_hist_score(frames)
    # plot score
    plt.figure(figsize = (8, 6))
    plt.plot(score)
    plt.savefig('score.png')
    # threshold score
    cuts = threshold_hist_score(score)
    return cuts

'''
use moment features to get cut frames
'''
def get_cuts_mom(frames):
    # calculate moment feature differences
    mom_diffs = get_mom_diff(frames)
    # plot diffs
    plt.figure(figsize = (8, 6))
    plt.plot(mom_diffs)
    plt.savefig('mom_diffs.png')
    # threshold diff
    cuts = threshold_mom_diff(mom_diffs)

    return cuts

if __name__ == "__main__":
    # load frames
    video_path = './movie.mp4'
    frames = get_frames(video_path)

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hist', action = 'store_true')
    parser.add_argument('--moment', action = 'store_true')
    parser.add_argument('--mix')

    args = parser.parse_args()

    cuts = []
    # use color histogram to identify key frames
    if args.hist:
        cuts = get_cuts_hist(frames)
        print(cuts)

    # use moment features to identify key frames
    elif args.moment:
        cuts = get_cuts_mom(frames)
        print(cuts)

    # use both algorithm to get 2 key frames sets, then union or intersect
    elif args.mix:
        cuts_hist = get_cuts_hist(frames)
        cuts_mom = get_cuts_mom(frames)
        
        cuts = set()
        # union
        if args.mix == '1':
            cuts = sorted(set(cuts_hist) | set(cuts_mom))
        # intersect
        elif args.mix == '2':
            cuts = sorted(set(cuts_hist) & set(cuts_mom))
        else:
            print("Wrong argument for --mix (1 or 2)")
            exit()

        print(cuts)

    # plot result
    plt.figure(figsize = (14, 6))
    plt.scatter(cuts, np.zeros_like(cuts))
    plt.legend('result')
    # plot truth
    truth = []
    with open('cuts.txt') as f:
        truth = [int(line.strip()) for line in f.readlines()]

    plt.scatter(truth, np.array(len(truth) * [0.1]))
    plt.legend('truth')
    plt.grid()
    plt.savefig('result.png')
