import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Filtro de Canny
    img_canny = cv.Canny(img_grayscale, 80, 180)

    # Parametros Blob
    blob_param = cv.SimpleBlobDetector_Params()

    # Filtros blob
    blob_param.filterByArea = False
    blob_param.minArea = 10
    blob_param.maxArea = 50000

    blob_param.filterByConvexity = False
    blob_param.minConvexity = 0.7

    blob_param.filterByInertia = False
    blob_param.minInertiaRatio = 0.3

    blob_param.filterByCircularity = True
    blob_param.minCircularity = 0.4

    blob_param.minDistBetweenBlobs = 20

    # Blob detector
    blob_detec = cv.SimpleBlobDetector_create(blob_param)

    obj_blobs = blob_detec.detect(img_canny)

    print('Total de objetos encontrados: ',len(obj_blobs))

    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
    cv.imshow('Imagem cinza', img_grayscale)
    cv.imshow('Imagem Limiarização Otsu', img_canny)
    cv.waitKey(0)
