import cv2
import imutils

image = cv2.imread("carte.jpg")
image_resized = imutils.resize(image, width=400)

gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (1, 1), 1000)
thresh = cv2.threshold(blur, 180, 200, cv2.THRESH_BINARY)[1]

edged = cv2.Canny(thresh, 100, 200)

output = image_resized.copy()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

for c in contours:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)

text = "I found {} cards!".format(len(contours))
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

rects = []

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    x, y, w, h = cv2.boundingRect(approximation)
    cropped = image_resized[y:y + h, x:x + w]
    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)

cv2.destroyAllWindows()
