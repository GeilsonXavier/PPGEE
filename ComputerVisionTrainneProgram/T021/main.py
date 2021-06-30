import cv2 as cv
import numpy as np
from numba import njit

seed = (0,0)

def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global seed
    seed = (y,x)

@njit
def region_growing(img, seed):
    linhas, colunas = img.shape
    xc, yc = seed # semente
    img_seg = np.zeros_like(img) # matriz da região segmentada
    img_seg[xc,yc] = 255 # posiciona a semente na imagem

    # controle de crescimento
    px_atual = 0
    px_anterior = 1

    # loop até o final do crescimento da região
    while px_anterior != px_atual:
        px_anterior = px_atual
        px_atual = 0
        for linha in range(linhas):
            for coluna in range(colunas):
                if img_seg[linha, coluna] == 255:
                    if img[linha-1, coluna-1] < 127:
                        img_seg[linha-1, coluna-1] = 255
                        px_atual += 1
                    if img[linha-1, coluna] < 127:
                        img_seg[linha-1, coluna] = 255
                        px_atual += 1
                    if img[linha-1, coluna+1] < 127:
                        img_seg[linha-1, coluna+1] = 255
                        px_atual += 1
                    if img[linha, coluna-1] < 127:
                        img_seg[linha, coluna-1] = 255
                        px_atual += 1
                    if img[linha, coluna+1] < 127:
                        img_seg[linha, coluna+1] = 255
                        px_atual += 1
                    if img[linha+1, coluna-1] < 127:
                        img_seg[linha+1, coluna-1] = 255
                        px_atual += 1
                    if img[linha+1, coluna] < 127:
                        img_seg[linha+1, coluna] = 255
                        px_atual += 1
                    if img[linha+1, coluna+1] < 127:
                        img_seg[linha+1, coluna+1] = 255
                        px_atual += 1
    return img_seg

if __name__ == '__main__':

    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Abre iamgem para seleção da semente e aguarda tecla para seguir
    cv.imshow('Imagem original', img)
    cv.setMouseCallback('Imagem original', mouse_click)
    cv.waitKey(0)

    # Marca posição da semente na imagem cinza
    img_aux = img_grayscale.copy()
    img_aux[seed] = 255

    # Crescimento de regiões
    img_seg = region_growing(img_grayscale, seed)

    # mostra imagem original e resuldados
    cv.imshow('Imagem em tons de cinza', img_grayscale)
    cv.imshow('Imagem com semente posicionada', img_aux)
    cv.imshow('Imagem Segmentada', img_seg)
    cv.waitKey(0)