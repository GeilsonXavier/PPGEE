import cv2 as cv

# ler imagem
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Filtro passa baixa mediana
img_median = cv.medianBlur(img_grayscale, ksize=5)

# Filtro passa baixa média
img_mean = cv.blur(img_grayscale, ksize=(5,5))

# mostra imagem original e resuldado dos filtros
cv.imshow('Imagem original', img)
cv.imshow('Imagem em nível de cinza', img_grayscale)
cv.imshow('Resultado com filtro passa baixa mediana', img_median)
cv.imshow('Resultado com filtro passa baixa média', img_mean)
cv.waitKey(0)

#salva imagem
cv.imwrite('Imagens/grayscale.jpg', img_grayscale)
cv.imwrite('Imagens/filtro_mediana.jpg', img_median)
cv.imwrite('Imagens/filtro_media.jpg', img_mean)
