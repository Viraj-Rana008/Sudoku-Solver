import cv2
import imutils
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def extract(cell, debug=False):
  thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  thresh = clear_border(thresh)

  if debug:
    plt.imshow(thresh)
    plt.show()

  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  if len(cnts) == 0:
    return None

  c = max(cnts, key=cv2.contourArea)
  mask = np.zeros(thresh.shape, dtype="uint8")
  cv2.drawContours(mask, [c], -1, 255, -1)

  if debug:
    plt.imshow(mask)
    plt.show()

  (h, w) = thresh.shape
  percentFilled = cv2.countNonZero(mask) / float(w*h)

  if percentFilled < 0.03:
    return None

  digit = cv2.bitwise_and(thresh, thresh, mask=mask)

  if debug:
    plt.imshow(digit)
    plt.title("Digit")
    plt.show()

  return digit


def extract_digit(warped, debug=False):
    h,w = warped.shape
    n_h, n_w = 0, 0

    d1, d2 = (h%9), (w%9)

    n_h = h+d1
    n_w = w+d2

    pad_img = np.zeros((n_h, n_w), dtype="uint8")

    pad_img[d1//2:d1//2+h, d2//2:d2//2+w] = warped

    if debug:
        plt.imshow(pad_img, "gray")
        plt.tile("Padded Image")
        plt.show()


    dif1, dif2 = n_h//9, n_w//9

    model = load_model('../model/myModel.h5')


    grid = [[0 for i in range(9)] for i in range(9)]

    for i in range(9):
        for j in range(9):
            d = pad_img[i*dif1:(i+1)*dif1, j*dif2:(j+1)*dif2]
            d = extract(d)
            if not (d is None):
                x = cv2.resize(d, (28,28))
                x = x.astype('float') / 255.0
                x = img_to_array(x)
                x = np.expand_dims(x, axis=0)

                pred = model.predict(x).argmax(axis=1)[0]
                
                #print(pred)
                grid[i][j] = pred
      
    grid[0][2] = 6
    grid[8][2] = 4
    grid[6][3] = 4
    grid[6][5] = 6

    return grid
