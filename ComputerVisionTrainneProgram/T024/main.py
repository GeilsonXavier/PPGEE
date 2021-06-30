import cv2 as cv
import numpy as np

seed = (0,0)

def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global seed
    seed = (y,x)

def region_growing(img, seed):
    # crescimento de região no fucinho
    min_c = 10
    max_c = 70
    
    linhas, colunas = img.shape[:2]
    xc, yc = seed # semente
    cor_ref = img[xc,yc]
    img_seg = np.zeros_like(img) # matriz da região segmentada
    img_seg[xc,yc] = cor_ref # posiciona a semente na imagem

    # controle de crescimento
    px_atual = 0
    px_anterior = 1

    # loop até o final do crescimento da região
    while px_anterior != px_atual:
        px_anterior = px_atual
        px_atual = 0
        for linha in range(linhas):
            for coluna in range(colunas):
                if np.array_equal(img[linha, coluna], cor_ref):
                    if np.array_equal(img[linha-1, coluna-1], cor_ref):
                        img_seg[linha-1, coluna-1] = cor_ref
                        px_atual += 1
                    if np.array_equal(img[linha-1, coluna], cor_ref):
                        img_seg[linha-1, coluna] = cor_ref
                        px_atual += 1
                    if np.array_equal(img[linha-1, coluna+1], cor_ref):
                        img_seg[linha-1, coluna+1] = cor_ref
                        px_atual += 1
                    if np.array_equal(img[linha, coluna-1], cor_ref):
                        img_seg[linha, coluna-1] = cor_ref
                        px_atual += 1
                    if np.array_equal(img[linha, coluna+1], cor_ref):
                        img_seg[linha, coluna+1] = cor_ref
                        px_atual += 1
                    if np.array_equal(img[linha+1, coluna-1], cor_ref):
                        img_seg[linha+1, coluna-1] = cor_ref
                        px_atual += 1
                    if np.array_equal(img[linha+1, coluna], cor_ref):
                        img_seg[linha+1, coluna] = cor_ref
                        px_atual += 1
                    if np.array_equal(img[linha+1, coluna+1], cor_ref):
                        img_seg[linha+1, coluna+1] = cor_ref
                        px_atual += 1
    return img_seg

if __name__ == '__main__':

    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')
    img = cv.resize(img, (0,0), fx=0.5, fy=0.5)

    # Abre iamgem para seleção da semente e aguarda tecla para seguir
    cv.imshow('Imagem original', img)
    cv.setMouseCallback('Imagem original', mouse_click)
    cv.waitKey(0)

    # Crescimento de regiões
    img_seg = region_growing(img, seed)

    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
#    cv.imshow('Imagem com semente posicionada', img_aux)
    cv.imshow('Imagem Segmentada', img_seg)
    cv.waitKey(0)