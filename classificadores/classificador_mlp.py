# %%
import numpy as np
import os
from pathlib import Path
from matplotlib.image import imread
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# %%
dataset_path = Path(os.path.join('C:',os.sep, 'Users', 'Gabriel', 'Desktop', 'FACUL 6ª SEM', 'APS-PI-6SEM', 'dataset', 'V20220930_partial'))

# %%
path_labels = {}

path_labels['a'] = dataset_path.joinpath('a_l/train_61')
path_labels['e'] = dataset_path.joinpath('e_l/train_65')
path_labels['i'] = dataset_path.joinpath('i_l/train_69')
path_labels['o'] = dataset_path.joinpath('o_l/train_6f')
path_labels['u'] = dataset_path.joinpath('u_l/train_75')
path_labels['A'] = dataset_path.joinpath('A_u/train_41')
path_labels['E'] = dataset_path.joinpath('E_u/train_45')
path_labels['I'] = dataset_path.joinpath('I_u/train_49')
path_labels['O'] = dataset_path.joinpath('O_u/train_4f')
path_labels['U'] = dataset_path.joinpath('U_u/train_55')

# %%
label_to_file = {
    k: [dataset_path / path_label / file for file in os.listdir(dataset_path / path_label)]  # NOQA:E501
    for k, path_label in path_labels.items()
}

# %%
X = []
y = []
for key, files in label_to_file.items():
    for path in files:
        y.append(key)
        image = imread(path)
        binary_matrix = []
        for line in image:
            binary_matrix.append([int(cell[0]) for cell in line])

        X.append(np.array(binary_matrix))

# %%
X_reshaped = [x.reshape(-1) for x in X]
X_reshaped = np.array(X_reshaped)

# %%
y = np.array(y)

# %%
y = ((y=='i')|(y=='I'))

# %%
split_test_threshold = 0.2

# %%
#train_index, test_index = next(selection_iter.split(X, y))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# %%
y_train_i = (y_train==True)
y_test_i = (y_test==True)

# %% [markdown]
# Rede Neural:

# %%
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='sgd', random_state=42)
mlp_clf.fit(X_train, y_train_i)

# %%
#mlp_clf.predict([d2_X_test[0]])

# %%
y_train_pred_RN = cross_val_predict(mlp_clf, X_train, y_train_i, cv=2)

# %%
print('Matriz de confusão do classificador MLP:')
print(confusion_matrix(y_train_i, y_train_pred_RN))

# %%
acuracia = cross_val_score(mlp_clf, X_train, y_train_i, cv=3, scoring='accuracy')
print(f'Valor da acurácia: {acuracia}')

# %%
print(f'Valor da precisão: {precision_score(y_train_i, y_train_pred_RN)}')

# %%
print(f'Valor do recall: {recall_score(y_train_i, y_train_pred_RN)}')

# %%
print(f'Valor do f1 score: {f1_score(y_train_i, y_train_pred_RN)}')

# %%
# Função para classificar uma imagem
def pretty_print_mnist_number(number: np.array):
    res = ''
    for linha in number.reshape(128, 128):
        for p in linha:
            res += f'{int(p):>3}'
        res += '\n'
    print(res)

# %%
def imagem_to_cinza(matrix_colorida: np.array) -> np.array:
    return cv2.cvtColor(matrix_colorida, cv2.COLOR_RGB2GRAY)
'''
def somar_valores_antes_depois_indice(dicionario, indice, valor_maximo):
    valores_antes = sum(v for i, v in dicionario.items() if i < indice)
    valores_depois = sum(v for i, v in dicionario.items() if i > indice)

    if (abs((valores_antes + valor_maximo) - (valores_depois)) < abs((valores_antes) - (valores_depois + valor_maximo))):
        return indice-1
    else:
        return indice+1
'''

# %%
def plotar_histograma_imagem(imagem):

    # Calcular o histograma usando a função cv2.calcHist()
    histograma = cv2.calcHist([imagem], [0], None, [256], [0, 256])

    # Criar um dicionário para armazenar o número de ocorrências para cada valor de intensidade
    #ocorrencias_por_valor = {i: int(hist) for i, hist in enumerate(histograma)}

    # Encontrar a intensidade com o maior número de ocorrências
    #maior_ocorrencia = max(ocorrencias_por_valor, key=ocorrencias_por_valor.get)
    #valor_maximo = ocorrencias_por_valor[maior_ocorrencia]

    limiar, _ = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar o valor do limiar baseado no pico do histograma
    #limiar = np.argmax(histograma)  # Índice do pico do histograma

    # Binarizar a imagem usando o limiar
    _, imagem_binarizada = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY)

     # Mostrar a imagem binarizada
    plt.imshow(imagem_binarizada, cmap='gray')
    plt.title('Imagem Binarizada')
    plt.axis('off')
    plt.show()

    # Plotar o histograma
    plt.plot(histograma)
    plt.title('Histograma da Imagem em Tons de Cinza')
    plt.xlabel('Valores de Pixel')
    plt.ylabel('Número de Pixels')
    plt.show()
    
    return limiar

# %%
def classificar_imagem_mlp(classificador, caminho_da_imagem):
    
    # Carregar a imagem
    imagem = Image.open(caminho_da_imagem)

    # Redimensionar para uma nova largura e altura
    nova_largura = 128
    nova_altura = 128
    imagem_redimensionada = imagem.resize((nova_largura, nova_altura))

    imagem_redimensionada.save("img/nova_imagem_redimensionada.jpg")

    imagem2 = imread("img/nova_imagem_redimensionada.jpg")

    imagem_cinza = imagem_to_cinza(imagem2)

    limiar = plotar_histograma_imagem(imagem_cinza) # Valor entre 0 e 255

    # Converter para matriz binária
    matriz_binaria = (imagem_cinza > limiar).astype(int)
    #_, matriz_binaria = cv2.threshold(imagem_cinza, limiar, 255, cv2.THRESH_BINARY)
    #plt.imshow(matriz_binaria, cmap='gray')

    imagem_processada = matriz_binaria.reshape(1, -1)
    previsao = classificador.predict(imagem_processada)
    
    pretty_print_mnist_number(matriz_binaria)
    print(f'Resultado da analise da imagem utilizando o classificador {classificador}: {previsao}')
