import cv2 as cv
import numpy as np

if __name__ == '__main__':
    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Filtro de Canny
    img_canny = cv.Canny(img_grayscale, 80, 180)

    # Encontra os contornos
    contornos, hierarquia = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # print(hierarquia)

    img_canny_contornos = img.copy()

    # Desenha os contornos encontrados
    cv.drawContours(img_canny_contornos, contornos, -1, (0,0,255), 3)

    # Calcula a área para cada contorno encontrado
    for i, contorno in enumerate(contornos):
        print('Área ' + str(i+1) + '= ' + str(cv.contourArea(contorno)))


    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
    cv.imshow('Imagem cinza', img_grayscale)
    cv.imshow('Imagem detecção de contornos - Filtro de Canny', img_canny_contornos)
    cv.waitKey(0)
