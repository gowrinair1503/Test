import cv2
import numpy as np

def analyze_test_result(test_img_path, test_type):
    img = cv2.imread(test_img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    if len(contours) < 2:
        return "Negative (only one line detected)"

    x1, y1, w1, h1 = cv2.boundingRect(contours[0])
    x2, y2, w2, h2 = cv2.boundingRect(contours[1])
    roi1 = gray[y1:y1+h1, x1:x1+w1]
    roi2 = gray[y2:y2+h2, x2:x2+w2]

    intensity1 = np.mean(roi1)
    intensity2 = np.mean(roi2)

    if test_type.lower() == "ezeefind":
        return "Positive" if intensity2 < 240 else "Negative"
    elif test_type.lower() == "ovufind":
        return "Positive" if intensity2 >= intensity1 else "Negative"
    elif test_type.lower() == "menofind":
        return "Positive" if intensity2 > intensity1 else "Negative"
    else:
        return "Unknown test type"
