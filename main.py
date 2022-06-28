from solver.detectSuduko import find_puzzle
from solver.extractDigit import extract_digit
from solver.solver import *


'''
    - No package found
        - skimage, tensorflow
    - Package installed
        - imutils
'''

INPUT_SIZE_X = 800
INPUT_SIZE_Y = 900

img = cv2.imread('./SudokuSolver/s2.jpg')
img = cv2.resize(img, (INPUT_SIZE_X, INPUT_SIZE_Y))

puzzle, warped = find_puzzle(img, True)

grid = extract_digit(warped)

solve_sudoku(grid)