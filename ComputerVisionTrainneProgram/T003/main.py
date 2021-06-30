import cv2 as cv

# ler imagem
img = cv.imread('Imagens/original.jpg')

# Separa canais RGB
canal_B, canal_G, canal_R = cv.split(img)

# mostra imagem e canais separados
cv.imshow('Imagem original', img)
cv.imshow('Imagem canal Red', canal_R)
cv.imshow('Imagem canal Green', canal_G)
cv.imshow('Imagem canal Blue', canal_B)
cv.waitKey(0)

#salva imagem
cv.imwrite('Imagens/canal_red.jpg', canal_R)
cv.imwrite('Imagens/canal_green.jpg', canal_G)
cv.imwrite('Imagens/canal_blue.jpg', canal_B)
