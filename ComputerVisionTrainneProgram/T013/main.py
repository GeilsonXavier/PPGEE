import cv2 as cv
import numpy as np

# ler imagem
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# resgata quantidade de linhas e colunas
linhas, colunas = img_grayscale.shape

# Cria matriz do tamanho da imagem
img_saida = np.zeros((linhas, colunas), dtype=np.uint8)

# Operadores Sobel
gx = [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]]

gy = [[-1, -2, -1],
      [0, 0, 0],
      [+1, 2, 1]]

# Convolução com kernel de Sobel - Descartando linhas e colunas zeradas
for linha in range(1, linhas-1):
    for coluna in range(1, colunas-1):
        Gx = img_grayscale[linha-1, coluna-1] * gx[0][0] + img_grayscale[linha, coluna-1] * gx[1][0] + \
             img_grayscale[linha+1, coluna-1] * gx[2][0] + img_grayscale[linha-1, coluna+1] * gx[0][2] + \
             img_grayscale[linha, coluna+1] * gx[1][2] + img_grayscale[linha+1, coluna+1] * gx[2][2]

        Gy = img_grayscale[linha-1, coluna-1] * gy[0][0] + img_grayscale[linha-1, coluna] * gy[0][1] + \
             img_grayscale[linha-1, coluna+1] * gy[0][2] + img_grayscale[linha+1, coluna-1] * gy[2][0] + \
             img_grayscale[linha+1, coluna] * gy[2][1] + img_grayscale[linha+1, coluna+1] * gy[2][2]

        img_saida[linha, coluna] = (Gx**2 + Gy**2)**(1/2)

# mostra imagem original e resuldados
cv.imshow('Imagem original', img)
cv.imshow('Imagem em nível de cinza', img_grayscale)
cv.imshow('Imagem filtro Sobel', img_saida)
cv.waitKey(0)

#salva imagens
cv.imwrite('Imagens/grayscale.jpg', img_grayscale)
cv.imwrite('Imagens/sobel.jpg', img_saida)