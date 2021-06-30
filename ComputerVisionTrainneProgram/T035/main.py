import cv2 as cv
import numpy as np

if __name__ == '__main__':
    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Limiarização de Otsu
    _, img_lim_otsu = cv.threshold(img_grayscale, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Elemento estruturante
    kernel = np.ones((5,5), np.uint8)

    # Dilatação
    for i in range(7):
        img_dilatada = cv.dilate(img_lim_otsu, kernel, iterations=i)
        cv.imshow('Imagem dilatada', img_dilatada)
        cv.waitKey(1000)

    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
    cv.imshow('Imagem cinza', img_grayscale)
    cv.imshow('Imagem Limiarização Otsu', img_lim_otsu)
    cv.waitKey(0)
