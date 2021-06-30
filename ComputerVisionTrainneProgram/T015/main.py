import cv2 as cv
import numpy as np

# Inicializa a câmera
cam = cv.VideoCapture(0)

# Aguarda a tecla 'q' para finalizar
while cv.waitKey(1) != ord('q'):
    # Captura o frame
    _, frame = cam.read()

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Aplica o filtro de Canny
    img_canny = cv.Canny(img_grayscale, 30, 100)

    # mostra imagem original e resuldados
    cv.imshow('Imagem da câmera original', frame)
    cv.imshow('Imagem da câmera em cinza', img_grayscale)
    cv.imshow('Imagem da câmera com filtro de Canny', img_canny)