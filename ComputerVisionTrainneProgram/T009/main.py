import cv2 as cv

# ler imagem 320x240
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# resgata quantidade de linhas e colunas
linhas, colunas = img_grayscale.shape

# mostra imagem original e resuldados
cv.imshow('Imagem original', img)
cv.imshow('Imagem em nível de cinza', img_grayscale)
cv.waitKey(0)

#salva imagens
cv.imwrite('Imagens/grayscale.jpg', img_grayscale)

# Salva todos os pixels em arquivo txt de acordo com a disposição na imagem
with open('Imagens/pixels.txt', 'w') as arquivo:
    for linha in range(linhas):
        for coluna in range(colunas):
            arquivo.write(str(img_grayscale[linha,coluna]) + ' ')
        arquivo.write('\n')