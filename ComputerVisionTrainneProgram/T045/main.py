import cv2 as cv
import numpy as np
import random

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
    blob_param.filterByArea = True
    blob_param.minArea = 10
    blob_param.maxArea = 50000

    blob_param.filterByConvexity = True
    blob_param.minConvexity = 0.7

    blob_param.filterByInertia = True
    blob_param.minInertiaRatio = 0.3

    blob_param.filterByCircularity = False
    blob_param.minCircularity = 0.4

    blob_param.minDistBetweenBlobs = 20

    # Blob detector
    blob_detec = cv.SimpleBlobDetector_create(blob_param)

    obj_blobs = blob_detec.detect(img_canny)

    print('Total de objetos encontrados: ',len(obj_blobs))

    # Pinta objetos detectados
    linhas, colunas = img.shape[:2]
    img_edit = img.copy()
    for i, b in enumerate(obj_blobs):
        # pega coordenadas
        x_sup_esq = int(b.pt[0] - b.size)
        y_sup_esq = int(b.pt[1] - b.size)
        x_inf_dir = int(b.pt[0] + b.size)
        y_inf_dir = int(b.pt[1] + b.size)

        print(b.pt, b.size)

        # Verifica se estão dentro da imagem
        if x_sup_esq < 0:
            x_sup_esq = 0
        if y_sup_esq < 0:
            y_sup_esq = 0
        if x_inf_dir > colunas:
            x_inf_dir = colunas
        if y_inf_dir > linhas:
            y_inf_dir = linhas

        # recorte nos objetos
        recorte = img_edit[y_sup_esq:y_inf_dir, x_sup_esq:x_inf_dir]
        cv.imshow('Recorte ' + str(i+1), recorte)
        cv.waitKey(10)

    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
    cv.imshow('Imagem cinza', img_grayscale)
    cv.imshow('Imagem Limiarização Otsu', img_canny)
    cv.waitKey(0)
