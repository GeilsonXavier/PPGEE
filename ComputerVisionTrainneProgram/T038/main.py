import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Limiarização de Otsu
    _, img_lim_otsu = cv.threshold(img_grayscale, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Elemento estruturante
    kernels = [cv.MORPH_RECT, cv.MORPH_ELLIPSE, cv.MORPH_CROSS]

    # Dilatação
    titulos = ['kernel retangular', 'kernel elíptico', 'kernel em cruz']
    for i, elemento in enumerate(kernels):
        kernel = cv.getStructuringElement(elemento, (7,7))
        img_dilatada = cv.dilate(img_lim_otsu, kernel, iterations=5)
        cv.imshow(titulos[i], img_dilatada)
        cv.waitKey(10)

    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
    cv.imshow('Imagem cinza', img_grayscale)
    cv.imshow('Imagem Limiarização Otsu', img_lim_otsu)
    cv.waitKey(0)
