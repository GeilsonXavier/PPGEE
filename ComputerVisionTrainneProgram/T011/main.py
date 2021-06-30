import cv2 as cv
import numpy as np

# ler imagem
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# resgata quantidade de linhas e colunas
linhas, colunas = img_grayscale.shape

# Cria matriz para limiarização do tamanho da imagem
img_aux = np.zeros((linhas, colunas), dtype=np.uint8)

# Salva todos os pixels da imagem na matriz
for linha in range(linhas):
    for coluna in range(colunas):
        img_aux[linha, coluna] = img_grayscale[linha, coluna]

# Inicializa e encontra os centróides
xc = yc = cont = 0
for linha in range(linhas):
    for coluna in range(colunas):
        if img_aux[linha, coluna] == 0:
            xc += linha
            yc += coluna
            cont += 1
xc = int(xc/cont)
yc = int(yc/cont)

# Desenha um círculo no centróide
cv.circle(img_aux, (xc, yc), 3, (255,255,255), -1)

# mostra imagem original e resuldados
cv.imshow('Imagem original', img)
cv.imshow('Imagem em nível de cinza', img_grayscale)
cv.imshow('Imagem com centróide', img_aux)
cv.waitKey(0)

#salva imagens
cv.imwrite('Imagens/grayscale.jpg', img_grayscale)
cv.imwrite('Imagens/centroide.jpg', img_aux)