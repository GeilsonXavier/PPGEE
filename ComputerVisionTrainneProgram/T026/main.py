import cv2 as cv
import numpy as np
from numba import njit
import random


@njit
def region_growing(img):
    # valores max e min para segmentação
    min_c = 0
    max_c = 230

    linhas, colunas = img.shape[:2]

    # matriz de seguimentação
    matriz_seg = np.zeros_like(img)

    n_objetos = 0

    for linha_ext in range(linhas):
        for coluna_ext in range(colunas):
            if matriz_seg[linha_ext, coluna_ext] == 0 and \
                    img[linha_ext, coluna_ext] < 230:
                n_objetos += 1
                matriz_seg[linha_ext, coluna_ext] = n_objetos

                # controle de crescimento
                px_atual = 0
                px_anterior = 1

                # loop até o final do crescimento da região
                while px_anterior != px_atual:
                    px_anterior = px_atual
                    px_atual = 0
                    for linha in range(linhas):
                        for coluna in range(colunas):
                            if matriz_seg[linha, coluna] == n_objetos:
                                if img[linha - 1, coluna - 1] < max_c:
                                    matriz_seg[linha - 1, coluna - 1] = n_objetos
                                    px_atual += 1
                                if img[linha - 1, coluna] < max_c:
                                    matriz_seg[linha - 1, coluna] = n_objetos
                                    px_atual += 1
                                if img[linha - 1, coluna + 1] < max_c:
                                    matriz_seg[linha - 1, coluna + 1] = n_objetos
                                    px_atual += 1
                                if img[linha, coluna - 1] < max_c:
                                    matriz_seg[linha, coluna - 1] = n_objetos
                                    px_atual += 1
                                if img[linha, coluna + 1] < max_c:
                                    matriz_seg[linha, coluna + 1] = n_objetos
                                    px_atual += 1
                                if img[linha + 1, coluna - 1] < max_c:
                                    matriz_seg[linha + 1, coluna - 1] = n_objetos
                                    px_atual += 1
                                if img[linha + 1, coluna] < max_c:
                                    matriz_seg[linha + 1, coluna] = n_objetos
                                    px_atual += 1
                                if img[linha + 1, coluna + 1] < max_c:
                                    matriz_seg[linha + 1, coluna + 1] = n_objetos
                                    px_atual += 1
    return matriz_seg, n_objetos

if __name__ == '__main__':
    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Crescimento de regiões
    img_seg, n_objetos = region_growing(img_grayscale)

    # Cria imagem RGB com marcações
    linhas, colunas = img_seg.shape[:2]
    img_edit = np.zeros([linhas, colunas, 3], np.uint8)

    # pinta cada objeto de uma cor
    for n in range(n_objetos):
        cor = lambda: random.randint(0, 255)
        img_edit[np.where(img_seg == n + 1)] = [cor(), cor(), cor()]

    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
    cv.imshow('Imagem cinza', img_grayscale)
    cv.imshow('Imagem Segmentada com marcações', img_edit)
    cv.waitKey(0)
