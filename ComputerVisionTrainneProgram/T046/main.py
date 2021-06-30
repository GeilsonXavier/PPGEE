import cv2 as cv
import os, csv, glob

def momentos_estatiscos(imgs):
    momentos_sta = []
    for i, img in enumerate(imgs):
        print('Extraindo atributos da imagem {} de {}'.format(i+1, len(imgs)))
        arq = cv.imread(img)
        arq = cv.cvtColor(arq, cv.COLOR_BGR2GRAY)
        momentos = cv.moments(arq)

        # lista com os atributos
        momentos_sta.append([momentos['m00'], momentos['m10'], momentos['m01'], momentos['m20'], momentos['m11'],
                             momentos['m02'], momentos['m30'], momentos['m21'], momentos['m12'], momentos['m03']])
    print('\n')

    return momentos_sta

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
    atributos = momentos_estatiscos(lista_img)
    salvar(dataset + 'Momentos estatisticos', atributos)