import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ler imagem 320x240
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Cria histograma da imagem - frequência de distribuição
def histograma(img):
    hist = np.zeros([256], np.uint8)
    linhas, colunas = img_grayscale.shape
    for linha in range(linhas):
        for coluna in range(colunas):
            hist[img[linha, coluna]] += 1
    return hist

# cumulative distribution frequency
def cdf(hist):
    tam_hist = len(hist)
    cdf = np.zeros([tam_hist])
    cdf[0] = hist[0]
    for i in range(1, tam_hist):
        cdf[i] = cdf[i-1] + hist[i]
    cdf = [elem*255/cdf[-1] for elem in cdf] # normaliza o histograma
    return cdf

# Equalização da imagem
def equalizar(img):
    cdf_hist = cdf(histograma(img))
    img_eq = np.interp(img, range(0,256), cdf_hist)
    return img_eq

img_equalizada = equalizar(img_grayscale)

plt.figure(1)
plt.subplot(221)
plt.imshow(img_grayscale, cmap='gray')
plt.subplot(222)
plt.hist(img_grayscale.ravel(), 256, [0, 256])
plt.subplot(223)
plt.imshow(img_equalizada, cmap='gray')
plt.subplot(224)
plt.hist(img_equalizada.ravel(), 256, [0, 256])
plt.show()