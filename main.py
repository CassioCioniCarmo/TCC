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
import matplotlib.pyplot as plt


def extract_hog_features(image):
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

    color_hist_features_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    color_hist_features_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    color_hist_features_b = cv2.calcHist([image], [2], None, [256], [0, 256])

    color_hist_features_r = cv2.normalize(color_hist_features_r, color_hist_features_r).flatten()
    color_hist_features_g = cv2.normalize(color_hist_features_g, color_hist_features_g).flatten()
    color_hist_features_b = cv2.normalize(color_hist_features_b, color_hist_features_b).flatten()

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

    #plt.subplot(1, 3,1)
    #plt.bar(np.arange(256), color_hist_features_g, width=1, color='g')
    #plt.xlabel("Intensidade")
    #plt.ylabel("Frequência")
    #plt.title(f"Histograma Verde")
    #plt.legend()
    
    #plt.subplot(1, 3, 2)
    #plt.bar(np.arange(256), color_hist_features_b, width=1, color='b')
    #plt.xlabel("Intensidade")
    #plt.ylabel("Frequência")
    #plt.title(f"Histograma Azul")
    #plt.legend()

    #plt.subplot(1, 3, 3)
    #plt.bar(np.arange(256), color_hist_features_r, width=1, color='r')
    #plt.xlabel("Intensidade")
    #plt.ylabel("Frequência")
    #plt.title(f"Histograma Vermelho")
    #plt.legend()

    #plt.tight_layout()

    # Mostrar a imagem original com os destaques
    #plt.figure(figsize=(8, 8))
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.title("Imagem Original com Regiões Destacadas")
    #plt.axis('off')
    #plt.show()


    return features


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


# Diretório onde as imagens estão armazenadas
diretorio_dados = "./imagens"

# Listas para armazenar os recursos (X) e os rótulos (y)
X = []
y = []

# Pesos para os recursos e valores de pertinência fuzzy (valores arbitrários, ajuste conforme necessário)
fuzzy_weights = [0.05, 0.3, 0.05, 0.05, 0.05, 0.05, 0,15]

# Loop sobre as subpastas, cada uma representando uma classe
for classe in os.listdir(diretorio_dados):
    classe_path = os.path.join(diretorio_dados, classe)
    if os.path.isdir(classe_path):
        # Carregar imagens da classe
        for imagem_nome in os.listdir(classe_path):
            imagem_path = os.path.join(classe_path, imagem_nome)
            image = cv2.imread(imagem_path)

            # Extrair recursos HOG da imagem
            hog_features = extract_hog_features(image)

            fuzzy_memberships = calculate_fuzzy_memberships(hog_features, fuzzy_weights)

            # Converter os valores de fuzzy_memberships em uma lista para garantir que possamos concatená-los corretamente
            valores_fuzzy = list(fuzzy_memberships.values())

            # Converter hog_features em uma lista
            valores_hog = list(hog_features.values())

            # Combine the values of pertinence fuzzy and weighted HOG features
            combined_features = np.hstack(valores_fuzzy)

            # Adicionar recursos e rótulo às listas X e y
            X.append(combined_features)
            y.append(classe)

# Converter para arrays numpy
X = np.array(X)
y = np.array(y)


# Step 1: Import necessary libraries (if not already done)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# ... (your existing code)

# Step 2: Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(100,), (150, 100), (200, 100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
}



# Treinamento da rede neural
clf = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=100000,activation = 'relu',solver='adam',random_state=1)
clf.fit(X, y)


# Salvar o modelo em um arquivo
modelo_arquivo = "modelo_rede_neural.joblib"
joblib.dump(clf, modelo_arquivo)

# Carregar uma nova imagem de teste
imagem_teste = cv2.imread("./DSC_0137.jpg")  # Substitua pelo caminho da imagem de teste
features_teste = extract_hog_features(imagem_teste)

# Calcular valores de pertinência fuzzy ponderados para os recursos da imagem de teste
fuzzy_memberships_test = calculate_fuzzy_memberships(features_teste, fuzzy_weights)

# Converter os valores de fuzzy_memberships em uma lista para garantir que possamos concatená-los corretamente
valores_fuzzy_test = list(fuzzy_memberships_test.values())

# Converter hog_features em uma lista
valores_hog_test = list(features_teste.values())

# Combine the values of pertinence fuzzy and weighted HOG features
combined_features_test = np.hstack(valores_fuzzy_test)


# Classificação pela rede neural
classe_predita = clf.predict([combined_features_test])[0]

# Probabilidades de pertencer a cada classe
probabilidades = clf.predict_proba([combined_features_test])[0]

# Imprimir a classe predita e as porcentagens de pertencer a cada classe
print("Classe predita pela rede neural:", classe_predita)
# Imprimir as probabilidades de pertencer a cada classe
for classe, probabilidade in enumerate(probabilidades):
    print(f"Probabilidade de pertencer à classe {classe}: {probabilidade:.2f}")


# Step 4: Perform Grid Search
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X, y)

# Step 5: Access the best hyperparameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best")
print("Params")
print(best_params)
print("Model")
print(best_model)

clf = best_model
clf.fit(X, y)

print(len(features_teste))
print(len(combined_features_test))
print(len(combined_features_test))

# Saving the model to a file (if not already done)
modelo_arquivo = "modelo_rede_neural.joblib"
joblib.dump(clf, modelo_arquivo)




