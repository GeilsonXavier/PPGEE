import cv2 as cv
import numpy as np

if __name__ == '__main__':
    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Transformada de Hough
    img_circ_hough = cv.HoughCircles(img_grayscale, cv.HOUGH_GRADIENT, 1, 20,
                                param1=150, param2=24, minRadius=0, maxRadius=0)

    try:
        img_circ_hough = np.uint16(np.around(img_circ_hough))

    except:
        print("Não existem circulos na imagem.")
        exit()

    img_circ = np.copy(img)
    for i in img_circ_hough[0,:]:
        cv.circle(img_circ, (i[0],i[1]), i[2], (0,0,255), 2)

    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
    cv.imshow('Imagem cinza', img_grayscale)
    cv.imshow('Imagem detecção de circulos - Transformada de Hough', img_circ)
    cv.waitKey(0)
