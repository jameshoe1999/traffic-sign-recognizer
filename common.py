import cv2 as cv
import os
import numpy as np
import random
import preprocessor as pp

def load_data(dir: str, limit: int = 0) -> tuple[np.ndarray, list[str]]:
    raws: list[np.ndarray] = []
    labels: list[str] = []
    files = os.listdir(dir)
    if limit in [0, None]:
        limit = len(files)
    count = 0
    for name in files:
        if (name.endswith('.ppm')):
            filepath = os.path.join(dir, name)
            img = read_image(filepath)
            if (img is not None):
                raws.append(img)
                # filename convention: ..._50.ppm
                # where between -6 and -4 is the label
                labels.append(str(name[-6:-4]))
                count += 1
                if (count >= limit):
                    break
    return raws, labels

def read_image(filepath: str) -> np.ndarray:
    img = cv.imread(filepath)
    if (img is not None):
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = cv.resize(img, (75, 75))
        img = np.uint8(img)
        return img
    return None

def preprocess_image(img: np.ndarray):
    normalized = pp.hist_normalize(img)
    houghed = pp.hough_circling(normalized)
    if houghed is not None:
        hog_img = pp.hog_descriptor(houghed)
        result: np.ndarray = np.array(hog_img)
        result = result.reshape(*result.shape, 1)
        return result
    return None

def files_rename(dir: str):
    files = os.listdir(dir)
    filepath = os.path.join(dir) + os.sep
    for file in files:
        if (file.endswith(".ppm") and file[2] == '_'): # only rename file start with "XX_....ppm"
            speed = file[:2]
            while (True): # continue renaming until no duplicate filename
                try:
                    ram = random.randint(1, 99999)
                    new_filename = "%s_%s.ppm" % (str(ram).rjust(5, "0"), speed)
                    os.rename(filepath + file, filepath + new_filename)
                    break
                except WindowsError:
                    print('Duplicate filename: ', new_filename)
                    continue
            print(f"Renamed {file} to {new_filename}")
