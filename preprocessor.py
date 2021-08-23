from operator import mul
import cv2 as cv
import numpy as np
from scipy.signal import convolve2d as conv2d
from skimage.feature import hog
from skimage.transform import resize
from skimage import restoration

def hist_normalize(img: np.ndarray) -> np.ndarray:
    hist: np.ndarray
    hist, _ = np.histogram(img.flatten(), 256, (0, 256))
    cdf = hist.cumsum()
    cdf_masked: np.ndarray = np.ma.masked_equal(cdf, 0)
    cdf_min = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    new_cdf = np.ma.filled(cdf_min, 0).astype('uint8')
    return new_cdf[img]

def color_filter(img: np.ndarray) -> np.ndarray:
    if len(img.shape) > 2:
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 0], dtype=np.uint8)
        upper_bound = np.array([150, 255, 255], dtype=np.uint8)
        mask = cv.inRange(img_hsv, lower_bound, upper_bound)
        img = cv.bitwise_and(img, img, mask=mask)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def sharpening(img: np.ndarray) -> np.ndarray:
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    psf = np.ones((5, 5)) / 35
    conved = conv2d(img, psf, 'same')
    denoise_img = restoration.richardson_lucy(conved, psf, 15, clip=False)
    return np.uint8(denoise_img)

def canny_edge_threshold(img: np.ndarray) -> tuple[np.ndarray, float]:
    threshold1 = 200
    threshold2 = 45
    edges = cv.Canny(img, threshold1, threshold2)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv.contourArea)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [max_contour], -1, (255, 255, 255), cv.FILLED)
    masked_img = cv.bitwise_and(img, img, mask=mask)
    non_zero = cv.countNonZero(mask)
    area = non_zero / mul(*img.shape[:2])
    return masked_img, area

def hog_descriptor(img: np.ndarray) -> np.ndarray:
    resized_img = resize(img, (128*4, 64*4))
    _, hog_img = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=True)
    return hog_img # feature descriptor

def hough_circling(img: np.ndarray) -> np.ndarray:
    img_clone = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = np.zeros(img_clone.shape, dtype=np.uint8)
    width, _ = img_clone.shape
    circles = cv.HoughCircles(img_clone, cv.HOUGH_GRADIENT_ALT, 1, width // 3, param1=300, param2=0.65)
    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')
        x, y, r = max(circles, key=(lambda x: x[2]))
        cv.circle(mask, (x, y), r, 255, cv.FILLED)
        img_clone = cv.bitwise_and(img_clone, img_clone, mask=mask)
        return img_clone
    return None