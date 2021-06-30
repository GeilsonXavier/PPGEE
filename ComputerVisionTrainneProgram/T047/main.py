import cv2 as cv
import os, csv, glob

def momentos_centrais(imgs):
    momentos_cent = []
    for i, img in enumerate(imgs):
        print('Extraindo atributos da imagem {} de {}'.format(i+1, len(imgs)))
        arq = cv.imread(img)
        arq = cv.cvtColor(arq, cv.COLOR_BGR2GRAY)
        momentos = cv.moments(arq)

        # lista com os atributos
        momentos_cent.append([momentos['mu20'], momentos['mu11'], momentos['mu02'], momentos['mu30'],
                              momentos['mu21'], momentos['mu12'], momentos['mu03']])
    print('\n')

    return momentos_cent

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
    atributos = momentos_centrais(lista_img)
    salvar(dataset + 'Momentos centrais', atributos)