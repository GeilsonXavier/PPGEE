import cv2 as cv

# ler imagem 320x240
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# resgata quantidade de linhas e colunas
linhas, colunas = img_grayscale.shape

# redimensiona imagem para 160x120
img_reduzida = cv.resize(img_grayscale, (int(linhas/2), int(colunas/2)))

# redimensiona imagem para 640x480
img_ampliada = cv.resize(img_grayscale, (linhas*2, colunas*2))

# mostra imagem original e resuldados
cv.imshow('Imagem original', img)
cv.imshow('Imagem em nível de cinza', img_grayscale)
cv.imshow('Resultado imagem reduzida', img_reduzida)
cv.imshow('Resultado imagem ampliada', img_ampliada)
cv.waitKey(0)

#salva imagens
cv.imwrite('Imagens/grayscale.jpg', img_grayscale)
cv.imwrite('Imagens/reduzida.jpg', img_reduzida)
cv.imwrite('Imagens/ampliada.jpg', img_ampliada)
