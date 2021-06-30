import cv2 as cv

# ler imagem
img = cv.imread('Imagens/original.jpg')

# transforma para HSV
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Separa canais HSV
canal_H, canal_S, canal_V = cv.split(img_hsv)

# mostra imagem original, imagem HSV e canais separados
cv.imshow('Imagem original', img)
cv.imshow('Imagem HSV', img_hsv)
cv.imshow('Imagem HSV canal H', canal_H)
cv.imshow('Imagem HSV canal S', canal_S)
cv.imshow('Imagem HSV canal V', canal_V)
cv.waitKey(0)

#salva imagem
cv.imwrite('Imagens/imagem_HSV.jpg', img_hsv)
cv.imwrite('Imagens/canal_h.jpg', canal_H)
cv.imwrite('Imagens/canal_s.jpg', canal_S)
cv.imwrite('Imagens/canal_v.jpg', canal_V)
