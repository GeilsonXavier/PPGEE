import cv2 as cv

# ler imagem
img = cv.imread('Imagens/original.jpg')

# transforma em nível de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# mostra imagem em nível de cinza
cv.imshow("Imagem em nível de cinza.", img_grayscale)
cv.waitKey(0)

# salva imagem
cv.imwrite('Imagens/grayscale.jpg', img_grayscale)