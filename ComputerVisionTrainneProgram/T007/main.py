import cv2 as cv

# ler imagem
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# aplica limiarização
_, img_threshold = cv.threshold(img_grayscale, 100, 255, cv.THRESH_BINARY)

# mostra imagem original e resuldados
cv.imshow('Imagem original', img)
cv.imshow('Imagem em nível de cinza', img_grayscale)
cv.imshow('Resultado da limiarização', img_threshold)
cv.waitKey(0)

#salva imagens
cv.imwrite('Imagens/grayscale.jpg', img_grayscale)
cv.imwrite('Imagens/limiarizacao.jpg', img_threshold)
