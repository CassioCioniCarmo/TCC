import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skfuzzy import gaussmf
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


# Pesos para os recursos e valores de pertinência fuzzy (valores arbitrários, ajuste conforme necessário)
fuzzy_weights = [0.05, 0.3, 0.05, 0.05, 0.05, 0.05, 0,15]

# Variáveis globais para armazenar informações sobre a seleção retangular
ref_pt = []
cropping = False

def extract_hog_features(image):
    image_cin = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image_cin = cv2.resize(image_cin, (64, 64))
    resized_image = cv2.resize(image, (64, 64))

    hog_features = hog(resized_image_cin, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    # Converter a imagem para o espaço de cores HSV
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # Extrair histograma de cores dos canais H, S e V da imagem HSV
    color_hist_features_h = cv2.calcHist([hsv_image], [0], None, [8], [0, 256])
    color_hist_features_s = cv2.calcHist([hsv_image], [1], None, [8], [0, 256])
    color_hist_features_v = cv2.calcHist([hsv_image], [2], None, [8], [0, 256])

    # Normalizar os histogramas de cores
    color_hist_features_h = cv2.normalize(color_hist_features_h, color_hist_features_h).flatten()
    color_hist_features_s = cv2.normalize(color_hist_features_s, color_hist_features_s).flatten()
    color_hist_features_v = cv2.normalize(color_hist_features_v, color_hist_features_v).flatten()

    color_hist_features_r = cv2.calcHist([image], [0], None, [8], [0, 256])
    color_hist_features_g = cv2.calcHist([image], [1], None, [8], [0, 256])
    color_hist_features_b = cv2.calcHist([image], [2], None, [8], [0, 256])

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

    return features


# Carregar o modelo treinado a partir do arquivo
modelo_arquivo = "modelo_rede_neural.joblib"
clf = joblib.load(modelo_arquivo)

# Function to handle mouse click events
def on_click(event, image):
    global ref_pt

    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        ref_pt.append((x, y))

    if len(ref_pt) == 2:
        # Draw the rectangle on the image
        x_start, y_start = ref_pt[0]
        x_end, y_end = ref_pt[1]
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        print(ref_pt)

        # Display the updated image
        plt.imshow(image)
        plt.title("Região selecionada")
        plt.axis('off')
        plt.show()


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


def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.*")])
    if file_path:
        image = cv2.imread(file_path)
        global image_copy
        image_copy = image.copy()
        global ref_pt
        ref_pt = []


        # Convert OpenCV image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        image_copy_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        # Prompt the user to click two points to select the region
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, image_copy_copy))

        # Display the image using matplotlib
        plt.imshow(image_copy_copy)
        plt.title("Selecione uma região")
        plt.axis('off')  # Turn off axis ticks and labels
        plt.show()

        # Obter as coordenadas do retângulo selecionado
        x_start, y_start = ref_pt[0]
        x_end, y_end = ref_pt[1]

        # Recortar a região selecionada da imagem original
        selected_region = image[y_start:y_end, x_start:x_end]
        cv2.imwrite("filename.jpg", selected_region)

        features_teste = extract_hog_features(selected_region)


        # Calcular valores de pertinência fuzzy ponderados para os recursos da imagem de teste
        fuzzy_memberships_test = calculate_fuzzy_memberships(features_teste, fuzzy_weights)

        # Converter os valores de fuzzy_memberships em uma lista para garantir que possamos concatená-los corretamente
        valores_fuzzy_test = list(fuzzy_memberships_test.values())

        # Converter hog_features em uma lista
        valores_hog_test = list(features_teste.values())

        # Combine the values of pertinence fuzzy and weighted HOG features
        combined_features_test = np.hstack(valores_fuzzy_test + valores_hog_test)

        # Classificação pela rede neural
        classe_predita = clf.predict([combined_features_test])[0]

        # Probabilidades de pertencer a cada classe
        probabilidades = clf.predict_proba([combined_features_test])[0]

          # Adicionar a classe predita e as probabilidades ao Tkinter
        result_text = f"Classe predita: {classe_predita}"
        for classe, probabilidade in enumerate(probabilidades):
            result_text += f"\nProbabilidade de pertencer à classe {classe}: {probabilidade:.2f}"

        result_label.config(text=result_text)

# Cria a janela principal
root = tk.Tk()
root.title("Seleção de Imagem")

# Botão para abrir a imagem
btn_open_image = tk.Button(root, text="Escolha imagem e selecione uma parte utiliando dois cliques", command=open_image)
btn_open_image.pack()

# Label para mostrar o resultado da classificação
result_label = tk.Label(root, text="", justify="left", font=("Helvetica", 12), anchor="w", bd=5)
result_label.pack()

# Inicia o loop do Tkinter
root.mainloop()