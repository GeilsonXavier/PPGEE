import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ler imagem 320x240
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Equaliza imagem
img_equalizada = cv.equalizeHist(img_grayscale)

# Calcula histogramas
hist_original = cv.calcHist(img_grayscale, channels=[0], mask=None, histSize=[256], ranges=[0,256])
hist_equalizado = cv.calcHist(img_equalizada, channels=[0], mask=None, histSize=[256], ranges=[0,256])

# Mostra histogramas
plt.figure(1)
plt.subplot(221)
plt.imshow(img_grayscale, cmap='gray')
plt.subplot(222)
plt.hist(img_grayscale.ravel(), 256)
plt.subplot(223)
plt.imshow(img_equalizada, cmap='gray')
plt.subplot(224)
plt.hist(img_equalizada.ravel(), 256)
plt.show()
