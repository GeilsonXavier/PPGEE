import cv2 as cv


if __name__ == '__main__':
    img1 = cv.imread('Imagens/original.jpg')
    img2 = cv.imread('Imagens/logo.jpg')

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Cria o detector
    sift = cv.SIFT_create()
    # orb = cv.ORB_create()

    # acha os keypoints e descritores
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    # descritores coincidentes
    matches = bf.match(desc1, desc2)

    # marcações
    img_match = cv.drawMatches(img1, kp1, img2, kp2, matches[:5], None, flags=2)

    #mostra resultados
    cv.imshow('Resultado', img_match)
    cv.waitKey(0)

    # salva resultado
    cv.imwrite('Imagens/sift.jpg', img_match)
