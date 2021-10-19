import numpy as np
from skimage.draw import polygon
import cv2
import json


def json2mask(json_file, img_file, out_dir):
    img = cv2.imread(img_file) 
    with open('data.json','r') as JSON:
        data = json.loads(JSON)
        image_name = data['imagePath']
        image_height = data['imageHeight']
        image_width = data['imageWidth']
        shapes = data["shapes"]
        for s in shapes:
            label = s['label']
            points = s['points']
            xs = []
            ys = []
            for p in points:
                xs.append(p[0])
                ys.append(p[1])
            mask = points2Mask(xs, ys, img)        


def points2Mask(xs, ys, img):
    mask = np.zeros(img.shape)
    rr, cc = polygon(xs, ys, img.shape)
    mask[rr, cc, 1] = 1
