import cv2 as cv
import numpy as np

if __name__ == '__main__':
    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Cria uma copia e apaga imagem original
    img_canny_contornos = img.copy()
    del img

    # Filtro de Canny
    img_canny = cv.Canny(img_grayscale, 80, 180)

    # Encontra os contornos e apaga variaveis da memoria
    contornos, hierarquia = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    del img_canny
    del hierarquia

    # Encontra a delimitação dos contornos
    pol_contornos = [None] * len(contornos)
    ret_delimitador = [None] * len(contornos)
    for i, contorno in enumerate(contornos):
        pol_contornos[i] = cv.approxPolyDP(contorno, 2, True)
        ret_delimitador[i] = cv.boundingRect(pol_contornos[i])

    # Desenha os retangulos para cada contorno encontrado e apaga
    for i, contorno in enumerate(pol_contornos):
        cv.rectangle(img_canny_contornos, (int(ret_delimitador[i][0]), int(ret_delimitador[i][1])),
                     (int(ret_delimitador[i][0]) + int(ret_delimitador[i][2]),
                      int(ret_delimitador[i][1]) + ret_delimitador[i][3]),
                     (0,255,0), 2)

        recorte = img_canny_contornos[int(ret_delimitador[i][1]):int(ret_delimitador[i][1]) + ret_delimitador[i][3],
                  int(ret_delimitador[i][0]):int(ret_delimitador[i][0]) + ret_delimitador[i][2]]

        cv.imshow('Objeto ' + str(i+1), recorte)
        cv.waitKey(10)
        del recorte

    # mostra imagem original e resuldados e apaga imagem
    cv.imshow('Imagem cinza', img_grayscale)
    del img_grayscale
    cv.imshow('Imagem detecção de contornos - Filtro de Canny', img_canny_contornos)
    cv.waitKey(0)
