import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skfuzzy import gaussmf
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from collections import Counter

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
    predita = []
    file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.*")])
    if file_path:
        imagem = cv2.imread(file_path)

        """image_copy = image.copy()
 
        # Convert OpenCV image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        image_copy_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        # Display the image using matplotlib
        plt.imshow(image_copy_copy)
        plt.title("Selecione uma região")
        plt.axis('off')  # Turn off axis ticks and labels
        plt.show()"""

        # Resize the image for display
        max_display_width = 800
        scale_factor = max_display_width / imagem.shape[1]
        display_height = int(imagem.shape[0] * scale_factor)
        resized_image = cv2.resize(imagem, (max_display_width, display_height))

        #cv2.imshow("Original", resized_image)
        #cv2.waitKey(0)  


        # Converter a imagem para o espaço de cores HSV
        imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

        # Resize the image for display
        max_display_width = 800
        scale_factor = max_display_width / imagem_hsv.shape[1]
        display_height = int(imagem_hsv.shape[0] * scale_factor)
        resized_image = cv2.resize(imagem_hsv, (max_display_width, display_height))

        #cv2.imshow("Original", resized_image)
        #cv2.waitKey(0)  

        
        # Extrair o canal S (saturação)
        canal_s = imagem_hsv[:, :, 1]

        # Resize the image for display
        max_display_width = 800
        scale_factor = max_display_width / canal_s.shape[1]
        display_height = int(canal_s.shape[0] * scale_factor)
        resized_image = cv2.resize(canal_s, (max_display_width, display_height))

        #cv2.imshow("Original", resized_image)
        #cv2.waitKey(0)  

        # Normalizar o canal S para valores entre 0 e 1
        canal_s_normalizado = canal_s / 255.0

        # Definir os valores de limiar para a segmentação
        limiar_inferior = 0.2
        limiar_superior = 1.0

        # Criar máscara baseada nos valores de limiar
        mascara = ((canal_s_normalizado >= limiar_inferior) & (canal_s_normalizado < limiar_superior)).astype(np.uint8) * 255

        # Pós-processamento para eliminar ruídos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara_segmentada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara_segmentada = cv2.morphologyEx(mascara_segmentada, cv2.MORPH_OPEN, kernel)

        # Resize the image for display
        max_display_width = 800
        scale_factor = max_display_width / mascara_segmentada.shape[1]
        display_height = int(mascara_segmentada.shape[0] * scale_factor)
        resized_image = cv2.resize(mascara_segmentada, (max_display_width, display_height))

        #cv2.imshow("Original", resized_image)
        #cv2.waitKey(0)  
        # Remover objetos indesejados fora da região de interesse
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(mascara_segmentada)
        # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
        # here, we're interested only in the size of the blobs, contained in the last column of stats.
        sizes = stats[:, -1]
        # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
        # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
        sizes = sizes[1:]
        nb_blobs -= 1
        tamanho_limite = 1000000

        # Encontrar o índice do maior componente
        maior_componente_id = np.argmax(sizes) + 1

        # Criar uma máscara para isolar o maior componente
        mascara_maior_componente = (im_with_separated_blobs == maior_componente_id).astype(np.uint8) * 255

        # Aplicar a máscara ao original
        partes_destacadas_s = cv2.bitwise_and(imagem, imagem, mask=mascara_maior_componente)

        """# Visualizar a máscara após remover objetos indesejados
        cv2.namedWindow("Máscara Após Remoção", cv2.WINDOW_NORMAL)
        cv2.imshow("Máscara Após Remoção", partes_destacadas_s) 
        cv2.waitKey(0)"""

        # Reduzir o tamanho das imagens para exibição
        altura, largura = imagem.shape[:2]
        nova_largura = 800
        nova_altura = int((nova_largura / largura) * altura)
        partes_destacadas_s_reduzidas = cv2.resize(partes_destacadas_s, (nova_largura, nova_altura))

        # Mostrar resultado
        #cv2.imshow("Partes Destacadas (Canal S)", partes_destacadas_s_reduzidas)
        #cv2.waitKey(0)

        # Tamanho dos quadrados em pixels
        tamanho_quadrado = 50

        # Obtém as dimensões da imagem resultante
        altura, largura, _ = partes_destacadas_s.shape

        # Create a copy of the original image to draw rectangles on
        image_with_rectangles = imagem.copy()

        # Itera sobre os quadrados na imagem resultante
        for y in range(0, altura, tamanho_quadrado):
            for x in range(0, largura, tamanho_quadrado):
                # Define as coordenadas dos cantos do quadrado
                canto_sup_esq = (x, y)
                canto_inf_dir = (x + tamanho_quadrado, y + tamanho_quadrado)
                
                # Extrai o quadrado da máscara
                quadrado_mascara = mascara_maior_componente[y:y+tamanho_quadrado, x:x+tamanho_quadrado]
                
                # Calcula a porcentagem de pixels não pretos no quadrado
                pixels_nao_pretos = np.count_nonzero(quadrado_mascara)
                total_pixels = tamanho_quadrado * tamanho_quadrado
                porcentagem_nao_pretos = (pixels_nao_pretos / total_pixels) * 100

                
                # Verifica se a porcentagem de pixels não pretos é maior que 80%
                if porcentagem_nao_pretos > 80.0:

                    # Extrai o quadrado da imagem resultante
                    quadrado = partes_destacadas_s[y:y+tamanho_quadrado, x:x+tamanho_quadrado]

                    features_teste = extract_features(quadrado)


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
                    #result_text = f"Classe predita: {classe_predita}"
                    predita.append(classe_predita)
                    #for classe, probabilidade in enumerate(probabilidades):
                    #    result_text += f"\nProbabilidade de pertencer à classe {classe}: {probabilidade:.2f}"
                    #result_label.config(text=result_text)
                    
                    # Draw a rectangle on the image_with_rectangles
                    cv2.rectangle(image_with_rectangles, canto_sup_esq, canto_inf_dir, (255, 0, 0), 2)
        # Resize the image for display
        max_display_width = 800
        scale_factor = max_display_width / image_with_rectangles.shape[1]
        display_height = int(image_with_rectangles.shape[0] * scale_factor)
        resized_image = cv2.resize(image_with_rectangles, (max_display_width, display_height))

        # Display the resized image with rectangles
        #cv2.imshow("Original Image with Rectangles", resized_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
                                # Mostra o quadrado
                    #cv2.imshow("Quadrado", quadrado)
                    #cv2.waitKey(500)  # Espera 500 milissegundos entre cada quadrado
                
    print("Soja Carijo:" + str(predita.count("Soja Carijo")))
    print("Soja Crestamento:" + str(predita.count("Soja Crestamento")))
    print("Soja Ferrugem:" + str(predita.count("Soja Ferrugem")))
    print("Soja Mildio:" + str(predita.count("Soja Mildio")))
    print("Soja Oidio:" + str(predita.count("Soja Oidio")))
    print("Soja Saudavel:" + str(predita.count("Soja Saudavel")))

    # Encontra a classe com a maior probabilidade, exceto "Soja Saudável"
    classes = clf.classes_
    classe_maior_prob = classes[1]

    # Contagem das ocorrências de cada classe
    contador = Counter(predita)

    # Cálculo do total de amostras
    total_amostras = len(predita)

    # Criação do dicionário com a ordem crescente das classes e a porcentagem do total
    dicionario_resultado = {}
    for classe, quantidade in sorted(contador.items()):
        porcentagem = (quantidade / total_amostras) * 100
        dicionario_resultado[classe] = porcentagem

    print(dicionario_resultado)

    # Encontre a classe com a maior porcentagem no dicionário
    classe_maior_porcentagem = max(dicionario_resultado, key=dicionario_resultado.get)

    print("Classe com maior porcentagem:", classe_maior_porcentagem)


    result_text = ""

    # Encontre a classe com a maior porcentagem no dicionário
    classe_maior_porcentagem = max(dicionario_resultado, key=dicionario_resultado.get)

    if classe_maior_porcentagem != "Soja Saudavel":
        result_text += f"Classe com maior probabilidade: {classe_maior_porcentagem} (Probabilidade: {dicionario_resultado[classe_maior_porcentagem]:.2f}%)\n\n"
    else:
        # Encontra a segunda classe com maior probabilidade, excluindo "Soja Saudável"
        segunda_maior_porcentagem = 0
        classe_segunda_maior_prob = None

        for classe, porcentagem in dicionario_resultado.items():
            if classe != "Soja Saudavel" and porcentagem > segunda_maior_porcentagem:
                segunda_maior_porcentagem = porcentagem
                classe_segunda_maior_prob = classe

        result_text += f"Classe com segunda maior probabilidade: {classe_segunda_maior_prob} (Probabilidade: {segunda_maior_porcentagem:.2f}%)\n\n"

    
    result_text += "Frames classificados como Soja Carijo:" + str(predita.count("Soja Carijo")) + "\n"
    result_text += "Frames classificados como Soja Crestamento:" + str(predita.count("Soja Crestamento")) + "\n"
    result_text += "Frames classificados como Soja Ferrugem:" + str(predita.count("Soja Ferrugem")) + "\n"
    result_text += "Frames classificados como Soja Mildio:" + str(predita.count("Soja Mildio")) + "\n"
    result_text += "Frames classificados como Soja Oidio:" + str(predita.count("Soja Oidio")) + "\n"
    result_text += "Frames classificados como Soja Saudavel:" + str(predita.count("Soja Saudavel")) + "\n"

    result_label.config(text=result_text)

# Cria a janela principal
root = tk.Tk()
root.title("Seleção de Imagem")

# Botão para abrir a imagem
btn_open_image = tk.Button(root, text="Escolha imagem para classificação", command=open_image)
btn_open_image.pack()

# Label para mostrar o resultado da classificação
result_label = tk.Label(root, text="", justify="left", font=("Helvetica", 12), anchor="w", bd=5)
result_label.pack()

# Inicia o loop do Tkinter
root.mainloop()