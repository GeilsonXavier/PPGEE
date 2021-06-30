import cv2 as cv
import numpy as np
from numba import njit

# ler imagem 320x240
img = cv.imread('Imagens/original.jpg')

# transforma para níveis de cinza
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

matriz_seg = np.zeros_like(img_grayscale)  # matriz da região segmentada
seed = (0,0)
cont_event = 0

def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global seed
        global cont_event
        global matriz_seg
        cont_event += 1
        matriz_seg[y,x] = cont_event

@njit
def region_growing(img, img_marc, seed=None):
    # valores max e min para segmentação
    min_c = 0
    max_c = 230

    linhas, colunas = img.shape[:2]

    # controle de crescimento
    px_atual = 0
    px_anterior = 1

    # loop até o final do crescimento da região
    while px_anterior != px_atual:
        px_anterior = px_atual
        px_atual = 0
        for linha in range(linhas):
            for coluna in range(colunas):
                # Marcação 01
                if img_marc[linha, coluna] == 1:
                    if img[linha-1, coluna-1] < max_c:
                        img_marc[linha-1, coluna-1] = 1
                        px_atual += 1
                    if img[linha-1, coluna] < max_c:
                        img_marc[linha-1, coluna] = 1
                        px_atual += 1
                    if img[linha-1, coluna+1] < max_c:
                        img_marc[linha-1, coluna+1] = 1
                        px_atual += 1
                    if img[linha, coluna-1] < max_c:
                        img_marc[linha, coluna-1] = 1
                        px_atual += 1
                    if img[linha, coluna+1] < max_c:
                        img_marc[linha, coluna+1] = 1
                        px_atual += 1
                    if img[linha+1, coluna-1] < max_c:
                        img_marc[linha+1, coluna-1] = 1
                        px_atual += 1
                    if img[linha+1, coluna] < max_c:
                        img_marc[linha+1, coluna] = 1
                        px_atual += 1
                    if img[linha+1, coluna+1] < max_c:
                        img_marc[linha+1, coluna+1] = 1
                        px_atual += 1

                # Marcação 02
                if img_marc[linha, coluna] == 2:
                    if img[linha-1, coluna-1] < max_c:
                        img_marc[linha-1, coluna-1] = 2
                        px_atual += 1
                    if img[linha-1, coluna] < max_c:
                        img_marc[linha-1, coluna] = 2
                        px_atual += 1
                    if img[linha-1, coluna+1] < max_c:
                        img_marc[linha-1, coluna+1] = 2
                        px_atual += 1
                    if img[linha, coluna-1] < max_c:
                        img_marc[linha, coluna-1] = 2
                        px_atual += 1
                    if img[linha, coluna+1] < max_c:
                        img_marc[linha, coluna+1] = 2
                        px_atual += 1
                    if img[linha+1, coluna-1] < max_c:
                        img_marc[linha+1, coluna-1] = 2
                        px_atual += 1
                    if img[linha+1, coluna] < max_c:
                        img_marc[linha+1, coluna] = 2
                        px_atual += 1
                    if img[linha+1, coluna+1] < max_c:
                        img_marc[linha+1, coluna+1] = 2
                        px_atual += 1

                # Marcação 03
                if img_marc[linha, coluna] == 3:
                    if img[linha-1, coluna-1] < max_c:
                        img_marc[linha-1, coluna-1] = 3
                        px_atual += 1
                    if img[linha-1, coluna] < max_c:
                        img_marc[linha-1, coluna] = 3
                        px_atual += 1
                    if img[linha-1, coluna+1] < max_c:
                        img_marc[linha-1, coluna+1] = 3
                        px_atual += 1
                    if img[linha, coluna-1] < max_c:
                        img_marc[linha, coluna-1] = 3
                        px_atual += 1
                    if img[linha, coluna+1] < max_c:
                        img_marc[linha, coluna+1] = 3
                        px_atual += 1
                    if img[linha+1, coluna-1] < max_c:
                        img_marc[linha+1, coluna-1] = 3
                        px_atual += 1
                    if img[linha+1, coluna] < max_c:
                        img_marc[linha+1, coluna] = 3
                        px_atual += 1
                    if img[linha+1, coluna+1] < max_c:
                        img_marc[linha+1, coluna+1] = 3
                        px_atual += 1

    return img_marc

if __name__ == '__main__':

    # Abre iamgem para seleção da semente e aguarda tecla para seguir
    cv.imshow('Seleção de sementes - Imagem cinza', img_grayscale)
    cv.setMouseCallback('Seleção de sementes - Imagem cinza', mouse_click)
    cv.waitKey(0)

    # Crescimento de regiões
    img_seg = region_growing(img_grayscale, matriz_seg)

    # Cria imagem RGB com marcações
    linhas, colunas = img_seg.shape[:2]
    img_edit = np.zeros([linhas, colunas, 3], np.uint8)
    img_edit[np.where(img_seg == 1)] = [0,0,255]
    img_edit[np.where(img_seg == 2)] = [255,0,0]
    img_edit[np.where(img_seg == 3)] = [0,255,0]

    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
    cv.imshow('Imagem Segmentada com marcações', img_edit)
    cv.waitKey(0)