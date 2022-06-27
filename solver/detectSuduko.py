from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt



def find_puzzle(image, debug=False):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  kernel = np.ones((5,5), np.uint8)
  blurred = cv2.GaussianBlur(gray, (5,5), 3)
  thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
  # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


  if debug:
    plt.imshow(thresh, 'gray')
    plt.show()

  cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]

  puzzleCnt = None

  

  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 15, True)
    
    # output = img.copy()
    # cv2.drawContours(output, [c], -1, (0,255,0), 15)
    # plt.imshow(output)
    # plt.show()

    if len(approx) == 4:
      puzzleCnt = approx
      break
    
  if puzzleCnt is None:
    raise Exception("Could not find Sudoku !!")

  if debug:
    output = image.copy()
    cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 20)
    plt.imshow(output)
    plt.show()
  
  puzzle  = four_point_transform(image, puzzleCnt.reshape(4, 2))
  warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

  if debug:
    plt.imshow(puzzle)
    plt.show()

  return puzzle, warped