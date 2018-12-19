import cv2

def segmentation(filename, path):

    cap = cv2.VideoCapture(filename)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            cv2.imwrite(path + '/' + str(count) + '.png', frame)
            count += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    filename = './movie.mp4'
    path = './pics'

    segmentation(filename, path)



