import cv2
import imutils
import math
import numpy as np
from scipy import ndimage
import os


def rotate_card_vertically(image_to_rotate):
    image_to_rotate = imutils.resize(image_to_rotate, width=400)
    contrast = cv2.addWeighted(image_to_rotate, 1.2,
                               np.zeros(image_to_rotate.shape, image_to_rotate.dtype), 0, 0)
    gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=math.pi / 180.0, threshold=100, minLineLength=100, maxLineGap=50)
    angles = []

    for x1, y1, x2, y2, in lines[0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    image_rotated = ndimage.rotate(image_to_rotate, median_angle)
    image_rotated = cv2.rotate(image_rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_rotated


def pre_processing(image_card):
    gray = cv2.cvtColor(image_card, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (1, 1), 1000)
    thresh = cv2.threshold(blur, 199, 200, cv2.THRESH_BINARY)[1]

    copied_image = image_card.copy()
    contours_card, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_card = sorted(contours_card, key=cv2.contourArea, reverse=True)[:1]
    return copied_image, contours_card


def show_contours(copied_image, contours_card):
    for c in contours_card:
        cv2.drawContours(copied_image, [c], -1, (240, 0, 159), 3)

    text = "I found {} cards!".format(len(contours_card))
    cv2.putText(copied_image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (240, 0, 159), 2)


def crop_around_card(contours_card, image_card):
    for c in contours_card:
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approximation)
        image_cropped = image_card[y:y + h, x:x + w]
        return image_cropped


def get_corner_card(image_card_cropped):
    height, width, channels = image_card_cropped.shape
    height_card = height / 3.5
    width_card = width / 6
    corner = image_card_cropped[0:int(height_card), 0:int(width_card)]
    return corner


if __name__ == '__main__':
    liste_card = os.listdir('photos')
    number_failed = 0
    for card in liste_card:
        try:
            image = cv2.imread("photos/" + card)
            image = rotate_card_vertically(image_to_rotate=image)
            output, contours = pre_processing(image_card=image)
            show_contours(copied_image=output, contours_card=contours)
            cropped = crop_around_card(contours_card=contours, image_card=image)
            corner = get_corner_card(cropped)
            cv2.imwrite("images_cropped_png/" + str(card).split('.')[0] + ".bmp", corner)
        except:
            print(str(card) + " failed")
            number_failed += 1
    print(number_failed)

    cv2.destroyAllWindows()
