import cv2 as cv

if __name__ == '__main__':
    # ler imagem 320x240
    img = cv.imread('Imagens/original.jpg')

    # transforma para níveis de cinza
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Limiarização automática
    img_lim_auto = cv.adaptiveThreshold(img_grayscale, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)


    # mostra imagem original e resuldados
    cv.imshow('Imagem original', img)
    cv.imshow('Imagem cinza', img_grayscale)
    cv.imshow('Imagem limiarização adaptativa', img_lim_auto)
    cv.waitKey(0)
