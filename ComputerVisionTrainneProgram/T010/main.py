import cv2 as cv
import numpy as np

# ler imagem 320x240
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# resgata quantidade de linhas e colunas
linhas, colunas = img_grayscale.shape

# Cria matriz para limiarização do tamanho da imagem
matriz_limiar = np.zeros((linhas, colunas), dtype=np.uint8)

# Salva todos os pixels da imagem na matriz
for linha in range(linhas):
    for coluna in range(colunas):
        matriz_limiar[linha, coluna] = img_grayscale[linha, coluna]

# Aplica limiarização e Salva todos os pixels em arquivo txt
with open('Imagens/pixels_matrix_limiarizacao.txt', 'w') as arquivo:
    for linha in range(linhas):
        for coluna in range(colunas):
            if matriz_limiar[linha, coluna] < 120:
                matriz_limiar[linha, coluna] = 0
            else:
                matriz_limiar[linha, coluna] = 255
            arquivo.write(str(matriz_limiar[linha,coluna]) + ' ')
        arquivo.write('\n')
    arquivo.close()

# mostra imagem original e resuldados
cv.imshow('Imagem original', img)
cv.imshow('Imagem em nível de cinza', img_grayscale)
cv.waitKey(0)

#salva imagens
cv.imwrite('Imagens/grayscale.jpg', img_grayscale)

""""# Para comparação...
_, img_limiar = cv.threshold(img_grayscale, 120, 255, cv.THRESH_BINARY)
cv.imshow('imagem limiarizada', img_limiar) # para comparação
cv.waitKey(0)
cv.imwrite('Imagens/grayscale.jpg', img_grayscale) # para comparação

# Aplica limiarização e Salva todos os pixels em arquivo txt
with open('Imagens/pixels_limiarizacao.txt', 'w') as arquivo:
    for linha in range(linhas):
        for coluna in range(colunas):
            arquivo.write(str(img_limiar[linha, coluna]) + ' ')
        arquivo.write('\n')
    arquivo.close()"""