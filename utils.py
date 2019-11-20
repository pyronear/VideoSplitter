# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pytesseract
import re
import difflib
import math
from functools import partial


def apply_threshold(image, threshold=30, invert=True):
    """
    Apply threshold to image and invert the colors if requested
    """
    # Convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply threshold: pixels below (above) become black (white)
    bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    return cv2.bitwise_not(bw) if invert else bw


def filter_contour(contour, image,
                   min_height=3e-3, max_height=0.05, min_width=0.3, max_width=1,
                   relative=True, location='bottom-left'):
    """
    Return true if contour is within min/max width and height,
    relative to image size if relative = True, and close to given location
    (top, bottom, right, left, bottom-right, etc)

    contour: cv2 contour or tuple (x, y, w, h)
    image: image
    min_height: float (default: 3e-3), minimum image height for contour
    max_height: float (default: 0.05), maximum image height for contour
    min_width: float (default: 0.3), minimum image width for contour
    max_width: float (default: 1), maximum image width for contour
    relative: bool (default: True), compute min and max values relative to image size
    location: str (default: 'bottom-left')
    """
    img_h, img_w = image.shape[:2]
    try:
        x, y, w, h = cv2.boundingRect(contour)
    except cv2.error:
        x, y, w, h = contour

    H, W = (h, w) if not relative else 1. * np.array((h, w)) / image.shape[:2]
    return (min_height <= H <= max_height) and \
           (min_width <= W <= max_width) and \
           (x < 0.1 * w if 'left' in location else True) and \
           (img_w - (x + w) < 0.1 * w if 'right' in location else True) and \
           (y < 0.1 * h if 'top' in location else True) and \
           (img_h - (y + h) < 0.1 * h if 'bottom' in location else True)


def findMSER(image, **kw):
    """
    Return a generator with the rectangular contours around maximally stable extremal
    regions in the given image

    Args:
        image: image
        kw: keyword arguments for filter_contour
    """
    bw = apply_threshold(image)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(bw)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    for contour in hulls:
        if filter_contour(contour, image, **kw):
            yield cv2.boundingRect(contour)


def findContours(image, **kw):
    """
    Return a generator with rectangular contours of the given image

    Args:
        image: image
        kw: keyword arguments for filter_contour
    """
    # Apply adaptiveThreshold at the bitwise_not of gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 15, -2)

    # Contours
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if filter_contour(contour, image, **kw):
            yield cv2.boundingRect(contour)


def findTextBoxes(image, min_height=5e-3, max_height=1, min_width=5e-3, max_width=1, relative=True):
    """
    Return a generator with all parts of the image that contain text and are within min/max width and height
    from https://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html

    Args:
        image: image
        min_height: float (default: 5e-3), minimum image height for text box
        max_height: float (default: 0.9), maximum image height for text box
        min_width: float (default: 5e-3), minimum image width for text box
        max_width: float (default: 0.9), maximum image width for text box
        relative: bool (default: True), compute min and max values relative to image size

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
    #_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=13)  # dilate
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        # get rectangle bounding contour
        x, y, w, h = cv2.boundingRect(contour)
        # discard areas that are too large
        H, W = (h, w) if not relative else 1. * np.array((h, w)) / image.shape[:2]
        if (min_height <= H <= max_height) and (min_width <= W <= max_width):
            yield x, y, w, h
        #else: print (W,H, image.shape)


def findFirstTextBox(image, **kw):
    """
    Return first text box in image

    Args:
        image: image
        kw: keyword arguments for findTextBoxes
    """
    try:
        x, y, w, h = next(findTextBoxes(image, **kw))
        return image[x:x + w + 1, y:y + h + 1]
    except StopIteration:
        return image


def findLargestTextBox(image, **kw):
    """
    """
    try:
        x, y, w, h = max(findTextBoxes(image, **kw), key=lambda x: x[2] * x[3])
        return image[x:x + w + 1, y:y + h + 1]
    except ValueError:
        return image


def findBlackBand(frame, threshold=10):
    """
    Return image from the first row where the mean of the pixels is < threshold

    Args:
        frame: image
        threshold: float (default: 10)
    """
    mean = frame.mean(axis=(1, 2))
    ymin = np.argmax(mean < threshold)
    return frame[ymin:]


def bottomFraction(image, fraction=0.05):
    """
    Return bottom fraction of image

    Args:
        image: image
        fraction: float (default: 0.05)
    """
    return image[int((1 - fraction) * image.shape[0]):]


def resizeImage(image, ratio=1, min_width=None, min_height=None):
    """
    Resize image by ratio or to obtain at least a minimum width or height

    Args:
        ratio: float, resizing ratio (default: 1)
        min_width: int, minimum width in pixels or None (default)
        min_heigth: int, minimum heigth in pixels or None (default)
    """
    height, width = image.shape[:2]
    ratio = max(ratio or 1, (min_width or width) / width, (min_height or height) / height)
    new_size = math.ceil(ratio * width), math.ceil(ratio * height)
    return cv2.resize(image, new_size)


def prepareOCR(frame, min_height=100, threshold=10,
               grayscale=False, invert_colors=False,
               cropFcn=lambda x: findBlackBand(bottomFraction(x))):
    """
    Return a pre-processed image for applying OCR

    Args:
        frame: image, 3D array (height, width, channels)

        min_height: int, minimum height of extraction image in pixels (default: 100).
            The image is resized to reach the desired height
            N.B.: The OCR was found to perform poorly in relatively small images

        grayscale: bool (default: True). Convert image to grayscale

        invert_colors: bool (default: True). Invert image colors (use 255 - values)

        cropFcn: function to crop image (or None)
    """
    if cropFcn is not None:
        frame = cropFcn(frame)

    height, width = frame.shape[:2]
    if min_height is not None and height < min_height:
        # Resize by a factor 2 until it reaches min_height
        ratio = min_height / height
        new_size = math.ceil(ratio * width), math.ceil(ratio * height)
        frame = cv2.resize(frame, new_size)

    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if invert_colors:
        frame = 255 - frame

    return frame


frame_to_string = partial(pytesseract.image_to_string, config='--psm 7')  # single line


def extract_coordinates(caption, pattern=r'(X|x)+:(\S+) (¥|Y)+:(\S+) (Z|2|7)+:(\S+)',
                        indices=[2, 4, 6, 7],
                        possible_matches=[],
                        fixes={'\n': ''}):
    """
    Extract and return the coordinates from the given caption. ValueError is raised if extraction fails

    Args:
        caption: str, the caption to be processed
        pattern: str, the pattern used to extract the coordinates
        indices: list of ints, the indices to retain from the pattern matching
        possible_matches: list of possible near-matches to be used as reference
        fixes: dictionary with replacements, used to correct (known) problems with OCR
    """
    fixed_caption = str(caption)
    for fix in fixes.items():
        fixed_caption = caption.replace(*fix)
    try:
        coordinates = tuple(map(re.split(pattern, fixed_caption).__getitem__, indices))
    except IndexError:
        raise ValueError('Problem extracting coordinates from caption', caption)

    # Remove datetime information from location name (last 2 words)
    coordinates = coordinates[:-1] + (' '.join(coordinates[-1].split()[:-2]),)

    # Replace location name (last item) by close match if (x,y,z) matches
    possibilities = set(i[-1] for i in possible_matches if i[:-1] == coordinates[:-1])
    matches = difflib.get_close_matches(coordinates[-1], possibilities, n=1, cutoff=0.9)
    if matches:
        coordinates = coordinates[:-1] + tuple(matches)
    return coordinates


def extract_timestamp(caption, pattern=r'(\d{4}(:|/)\d{2}(:|/)\d{2})', split=True):
    """
    Return a string with date and time extracted from the given string
    """
    match = re.split(pattern, caption)
    try:
        date = match[1]
        time = match[-1].split()[0]
    except IndexError:
        raise ValueError('Problem extracting timestamp from caption', caption)
    return ' '.join([date, time])


# unittests
import unittest


class UtilsTester(unittest.TestCase):
    """
    Test caption manipulation
    """
    def setUp(self):
        import urllib
        import imutils
        url = 'https://gist.github.com/blenzi/1cf8d14fd01494f7d9c0e34714f35c29/raw'
        self.ref = eval(urllib.request.urlopen(url).read())
        self.img = imutils.url_to_image(self.ref['url'])

    def test_findMSER(self):
        "Test if at least one region is found with MSER"
        self.assertTrue(list(findMSER(self.img)))

    def test_findContours(self):
        "Test if at least one contour is found"
        self.assertTrue(list(findContours(self.img)))

    def test_filter_contour(self):
        for contour in findMSER(self.img):
            self.assertFalse(filter_contour(contour, self.img, location='top'))

    def test_extract_coordinates(self):
        caption = self.ref['caption']
        coordinates = self.ref['coordinates']
        self.assertEqual(extract_coordinates(caption), coordinates)

    def test_extract_timestamp(self):
        caption = self.ref['caption']
        timestamp = self.ref['timestamp']
        self.assertEqual(extract_timestamp(caption), timestamp)

    def test_shape(self):
        self.assertEqual(self.ref['shape'], self.img.shape)


if __name__ == '__main__':
    unittest.main()
