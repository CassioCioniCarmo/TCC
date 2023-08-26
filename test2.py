import os
import numpy as np
import cv2
import joblib
from skimage.feature import hog
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from skimage.feature import hog
import cv2
import joblib
import skfuzzy as fuzz
from skfuzzy import membership
from scipy.stats import norm
from skfuzzy import gaussmf, gbellmf
from sklearn.metrics import confusion_matrix




# Carregar o modelo treinado
modelo_arquivo = "modelo_rede_neural.joblib"
clf = joblib.load(modelo_arquivo)

# Diretório com as imagens de teste
diretorio_teste = "./imagens_teste"



# Pesos para os recursos e valores de pertinência fuzzy (valores arbitrários, ajuste conforme necessário)
fuzzy_weights = [0.05, 0.3, 0.05, 0.05, 0.05, 0.05, 0,15]


def extract_features(image):
    image_cin = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image_cin = cv2.resize(image_cin, (64, 64))
    resized_image = cv2.resize(image, (64, 64))

    hog_features = hog(resized_image_cin, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    # Converter a imagem para o espaço de cores HSV
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # Extrair histograma de cores dos canais H, S e V da imagem HSV
    color_hist_features_h = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    color_hist_features_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    color_hist_features_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Normalizar os histogramas de cores
    color_hist_features_h = cv2.normalize(color_hist_features_h, color_hist_features_h).flatten()
    color_hist_features_s = cv2.normalize(color_hist_features_s, color_hist_features_s).flatten()
    color_hist_features_v = cv2.normalize(color_hist_features_v, color_hist_features_v).flatten()

    color_hist_features_r = cv2.calcHist([resized_image], [0], None, [256], [0, 256])
    color_hist_features_g = cv2.calcHist([resized_image], [1], None, [256], [0, 256])
    color_hist_features_b = cv2.calcHist([resized_image], [2], None, [256], [0, 256])

    color_hist_features_r = cv2.normalize(color_hist_features_r, color_hist_features_r).flatten()
    color_hist_features_g = cv2.normalize(color_hist_features_g, color_hist_features_g).flatten()
    color_hist_features_b = cv2.normalize(color_hist_features_b, color_hist_features_b).flatten()

    #print(len(hog_features))
    #print(len(color_hist_features_r))
    #print(len(color_hist_features_g))
    #print(len(color_hist_features_b))
    #print(len(color_hist_features_h))
    #print(len(color_hist_features_s))
    #print(len(color_hist_features_v))

    # Combine the HOG and color histogram features in a dictionary
    features = {
        'hog': hog_features,
        'color_hist_h': color_hist_features_h,
        'color_hist_s': color_hist_features_s,
        'color_hist_v': color_hist_features_v,
        'color_hist_r': color_hist_features_r,
        'color_hist_g': color_hist_features_g,
        'color_hist_b': color_hist_features_b,
    }

    return features


# Carregar o modelo treinado a partir do arquivo
modelo_arquivo = "modelo_rede_neural.joblib"
clf = joblib.load(modelo_arquivo)

def calculate_fuzzy_memberships(features, fuzzy_weights):
    fuzzy_memberships = {}

    # HOG feature
    hog_fuzzy = gaussmf(features['hog'], np.mean(features['hog']), np.std(features['hog']))
    fuzzy_memberships['hog'] = hog_fuzzy * fuzzy_weights[0]

    # color_hist_features_h
    color_h_fuzzy = gaussmf(features['color_hist_h'], np.mean(features['color_hist_h']), np.std(features['color_hist_h']))
    fuzzy_memberships['color_hist_h'] = color_h_fuzzy * fuzzy_weights[1]

    # color_hist_features_s
    color_s_fuzzy = gaussmf(features['color_hist_s'], np.mean(features['color_hist_s']), np.std(features['color_hist_s']))
    fuzzy_memberships['color_hist_s'] = color_s_fuzzy * fuzzy_weights[2]

    # color_hist_features_v
    color_v_fuzzy = gaussmf(features['color_hist_v'], np.mean(features['color_hist_v']), np.std(features['color_hist_v']))
    fuzzy_memberships['color_hist_v'] = color_v_fuzzy * fuzzy_weights[3]

    # color_hist_features_r
    color_r_fuzzy = gaussmf(features['color_hist_r'], np.mean(features['color_hist_r']), np.std(features['color_hist_r']))
    fuzzy_memberships['color_hist_r'] = color_r_fuzzy * fuzzy_weights[4]

    # color_hist_features_g
    color_g_fuzzy = gaussmf(features['color_hist_g'], np.mean(features['color_hist_g']), np.std(features['color_hist_g']))
    fuzzy_memberships['color_hist_g'] = color_g_fuzzy * fuzzy_weights[5]

    # color_hist_features_b
    color_b_fuzzy = gaussmf(features['color_hist_b'], np.mean(features['color_hist_b']), np.std(features['color_hist_b']))
    fuzzy_memberships['color_hist_b'] = color_b_fuzzy * fuzzy_weights[6]

    return fuzzy_memberships


import os
import numpy as np
import cv2
import joblib
from skimage.feature import hog
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Carregar o modelo treinado
modelo_arquivo = "modelo_rede_neural.joblib"
clf = joblib.load(modelo_arquivo)

# Diretório com as imagens de teste
diretorio_teste = "./imagens_teste"

X_test = []
y_test = []

# Loop sobre as subpastas, cada uma representando uma classe
for classe in os.listdir(diretorio_teste):
    classe_path = os.path.join(diretorio_teste, classe)
    if os.path.isdir(classe_path):
        # Carregar imagens de teste da classe
        for imagem_nome in os.listdir(classe_path):
            imagem_path = os.path.join(classe_path, imagem_nome)
            image = cv2.imread(imagem_path)

            # Extrair recursos HOG da imagem de teste
            hog_features_test = extract_features(image)

            fuzzy_memberships_test = calculate_fuzzy_memberships(hog_features_test, fuzzy_weights)

            # Converter os valores de fuzzy_memberships em uma lista para concatená-los
            valores_fuzzy_test = list(fuzzy_memberships_test.values())

            # Converter hog_features em uma lista
            valores_hog_test = list(hog_features_test.values())

            # Combinar os valores de pertinência fuzzy e recursos HOG ponderados
            combined_features_test = np.hstack(valores_fuzzy_test)

            # Adicionar recursos e rótulo às listas de teste X_test e y_test
            X_test.append(combined_features_test)
            y_test.append(classe)

# Converter para arrays numpy
X_test = np.array(X_test)
y_test = np.array(y_test)

# Prever classes para as imagens de teste
predicted_values = clf.predict(X_test)

# Calcular a acurácia usando a função accuracy_score
accuracy = accuracy_score(y_test, predicted_values)
print("Acurácia:", accuracy)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, predicted_values)
print("Matriz de Confusão:")
print(conf_matrix)

# Calcular a precisão (Precision)
precision = precision_score(y_test, predicted_values, average='weighted')
print("Precisão:", precision)

# Calcular o recall (Recall)
recall = recall_score(y_test, predicted_values, average='weighted')
print("Recall:", recall)

# Calcular o F1-Score
f1 = f1_score(y_test, predicted_values, average='weighted')
print("F1-Score:", f1)

# Imprimir o relatório de classificação
class_report = classification_report(y_test, predicted_values)
print("Relatório de Classificação:")
print(class_report)
