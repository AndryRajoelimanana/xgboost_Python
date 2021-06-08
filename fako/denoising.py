#
#  tesseract test_ocr.png stdout --user-words /home/andry/projects/Learning/mlg_text/mlg.word-dawg -c tessedit_char_blacklist=c --dpi 300

import cv2
from PIL import Image
import os
import pytesseract
image = cv2.imread('test_ocr.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
text = pytesseract.image_to_string(Image.open(filename), lang='mlg')
print(text)
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)



