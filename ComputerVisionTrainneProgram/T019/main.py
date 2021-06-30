import cv2 as cv
import matplotlib.pyplot as plt

# ler imagem 320x240
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Filtro de Sobel
Gx = cv.Sobel(img_grayscale, dx=1, dy=0, ddepth=cv.CV_64F, ksize=3)
Gy = cv.Sobel(img_grayscale, dx=0, dy=1, ddepth=cv.CV_64F, ksize=3)
img_sobel = (Gx**2 + Gy**2)**(1/2)
img_sobel = cv.convertScaleAbs(img_sobel)

# mostra imagem original e resuldados
cv.imshow('Imagem original', img)

plt.figure(1)
plt.subplot(221)
plt.imshow(img_grayscale, cmap='gray')
plt.subplot(222)
plt.hist(img_grayscale.ravel(), 256, [0, 256])
plt.subplot(223)
plt.imshow(img_sobel, cmap='gray')
plt.subplot(224)
plt.hist(img_sobel.ravel(), 256, [0, 256])
plt.show()