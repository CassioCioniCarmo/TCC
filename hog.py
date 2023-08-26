import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

# Coordenadas dos retângulos destacados
highlight_coords = [
    (0, 0, 100, 100),
    (200, 100, 300, 200),  # Adicione outras coordenadas aqui
    # ...
]

# Carregar a imagem
image_path = './DSC_0141_3.png'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Definir regiões de interesse (ROI)
regions = [
    image[y0:y1, x0:x1] for x0, y0, x1, y1 in highlight_coords
]

# Calcular o HOG e mostrar os histogramas das regiões
num_bins = 9

plt.figure(figsize=(15, 5 * len(regions)))

for i, region in enumerate(regions):
    # Histograma de cores RGB
    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

    color_hist_features_r = cv2.calcHist([region], [0], None, [256], [0, 256])
    color_hist_features_g = cv2.calcHist([region], [1], None, [256], [0, 256])
    color_hist_features_b = cv2.calcHist([region], [2], None, [256], [0, 256])

    color_hist_features_r = cv2.normalize(color_hist_features_r, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
    color_hist_features_g = cv2.normalize(color_hist_features_g, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
    color_hist_features_b = cv2.normalize(color_hist_features_b, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

    
    # Histograma de cores HSV
    region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
    color_hist_features_h = cv2.calcHist([region_hsv], [0], None, [256], [0, 256])
    color_hist_features_s = cv2.calcHist([region_hsv], [1], None, [256], [0, 256])
    color_hist_features_v = cv2.calcHist([region_hsv], [2], None, [256], [0, 256])

    color_hist_features_h = cv2.normalize(color_hist_features_h, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
    color_hist_features_s = cv2.normalize(color_hist_features_s, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
    color_hist_features_v = cv2.normalize(color_hist_features_v, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()


    gray_image = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hog_features, hog_image = hog(gray_image, block_norm='L2-Hys', visualize=True)
    histogram_orientations = np.zeros(num_bins)
    orientations = hog_features[:num_bins]
    for j in range(num_bins):
        histogram_orientations[j] = orientations[j]
    
    #plt.subplot(len(regions), 3, i * 3 + 1)
    #plt.imshow(region, cmap='gray')
    #plt.title(f"Região de Interesse {i + 1}")
    #plt.axis('off')
    
    #plt.subplot(len(regions), 3, i * 3 + 2)
    #plt.bar(np.arange(num_bins) * 20, histogram_orientations, width=20)
    #plt.xlabel("Orientação (graus)")
    #plt.ylabel("Magnitude")
    #plt.title(f"Histograma da Região {i + 1}")
    
    #plt.subplot(len(regions), 3, i * 3 + 3)
    #plt.imshow(hog_image, cmap='gray')
    #plt.title(f"HOG da Região {i + 1}")
    #plt.axis('off')

    #plt.subplot(len(regions), 5, i * 5 + 3)
    #plt.bar(np.arange(256), color_hist_features_r, width=20, color='r')
    #plt.xlabel("Intensidade")
    #plt.ylabel("Frequência")
    #plt.title(f"Histograma Vermelho da Região {i + 1}")
    #plt.legend()
    
    plt.subplot(len(regions), 5, i * 5 + 4)
    plt.bar(np.arange(256), color_hist_features_g, width=1, color='g')
    plt.xlabel("Intensidade")
    plt.ylabel("Frequência")
    plt.title(f"Histograma Verde")
    plt.legend()
    
    plt.subplot(len(regions), 5, i * 5 + 5)
    plt.bar(np.arange(256), color_hist_features_b, width=1, color='b')
    plt.xlabel("Intensidade")
    plt.ylabel("Frequência")
    plt.title(f"Histograma Azul")
    plt.legend()

    plt.subplot(len(regions), 5, i * 5 + 3)
    plt.bar(np.arange(256), color_hist_features_r, width=1, color='r')
    plt.xlabel("Intensidade")
    plt.ylabel("Frequência")
    plt.title(f"Histograma Vermelho")
    plt.legend()
    
    
    #plt.subplot(len(regions), 5, i * 5 + 3)
    #plt.bar(np.arange(256), color_hist_features_h, width=1, color='y')
    #plt.xlabel("Intensidade")
    #plt.ylabel("Frequência")
    #plt.title(f"Histograma Matiz")
    #plt.legend()

    #plt.subplot(len(regions), 5, i * 5 + 4)
    #plt.bar(np.arange(256), color_hist_features_s, width=1, color='m')
    #plt.xlabel("Intensidade")
    #plt.ylabel("Frequência")
    #plt.title(f"Histograma Saturação")
    #plt.legend()

    #plt.subplot(len(regions), 5, i * 5 + 5)
    #plt.bar(np.arange(256), color_hist_features_v, width=1, color='k')
    #plt.xlabel("Intensidade")
    #plt.ylabel("Frequência")
    #plt.title(f"Histograma Valor")
    #plt.legend()


   

    # Adicionar destaque na imagem original colorida
    x0, y0, x1, y1 = highlight_coords[i]
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)  # Destaque em vermelho

plt.tight_layout()

# Mostrar a imagem original com os destaques
#plt.figure(figsize=(8, 8))
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.title("Imagem Original com Regiões Destacadas")
#plt.axis('off')
plt.show()
