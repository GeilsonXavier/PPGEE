import cv2 as cv
import os, csv, glob

def momentos_invariantes(imgs):
    momentos_hu_total = []
    for i, img in enumerate(imgs):
        print('Extraindo atributos da imagem {} de {}'.format(i+1, len(imgs)))
        arq = cv.imread(img)
        arq = cv.cvtColor(arq, cv.COLOR_BGR2GRAY)
        momentos = cv.moments(arq)

        momentos_hu = cv.HuMoments(momentos)
        momentos_n = [momento[0] for momento in momentos_hu]

        momentos_hu_total.append(momentos_n)

    print('\n')

    return momentos_hu_total

def salvar(nome_extrator, atributos):
    for atributo in atributos:
        print(atributo)
    # Salva em *.csv
    with open(nome_extrator + '.csv', 'w', newline='') as arquivo:
        gravar = csv.writer(arquivo)
        gravar.writerows(atributos)

if __name__ == '__main__':
    dataset = 'Imagens/'
    lista_img = glob.glob(os.path.join(dataset, '*.jpg'))
    atributos = momentos_invariantes(lista_img)
    salvar(dataset + 'Momentos invariantes', atributos)