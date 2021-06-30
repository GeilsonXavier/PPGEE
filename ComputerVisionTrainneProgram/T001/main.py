import cv2 as cv

# ler imagem
img = cv.imread('Imagens/logo.jpg')

# mostra imagem
cv.imshow('Imagem', img)
cv.waitKey(0)

#salva imagem
cv.imwrite('Imagens/saved_logo.jpg', img)
