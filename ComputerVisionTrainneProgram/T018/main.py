import cv2 as cv

# ler imagem 320x240
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Filtro laplaciano
img_laplace = cv.Laplacian(img_grayscale, ddepth=cv.CV_64F, ksize=3)
img_laplace = cv.convertScaleAbs(img_laplace)

# Equaliza a imagem
img_laplace_eq = cv.equalizeHist(img_laplace)

# mostra imagem original e resuldados
cv.imshow('Imagem original', img)
cv.imshow('Imagem em tons de cinza', img_grayscale)
cv.imshow('Imagem com filtro Laplaciano', img_laplace)
cv.imshow('Imagem com filtro Laplaciano equalizado', img_laplace_eq)

cv.waitKey(0)