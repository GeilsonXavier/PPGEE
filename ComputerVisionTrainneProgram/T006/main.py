import cv2 as cv

# ler imagem
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Filtro passa alta de Canny
img_canny = cv.Canny(img_grayscale, 50, 150) # as diferenças são percebidas alterando-se os limites inferior e superior


# mostra imagem original e resuldado dos filtros
cv.imshow('Imagem original', img)
cv.imshow('Imagem em nível de cinza', img_grayscale)
cv.imshow('Resultado com filtro passa alta de Canny', img_canny)
cv.waitKey(0)

#salva imagem
cv.imwrite('Imagens/grayscale.jpg', img_grayscale)
cv.imwrite('Imagens/filtro_canny.jpg', img_canny)
