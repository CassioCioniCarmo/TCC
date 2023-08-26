import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('DSC_0141_3.png')

# Converter para HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Separar os canais H, S e V
h_channel, s_channel, v_channel = cv2.split(hsv_image)

# Calcular os histogramas
h_hist = cv2.calcHist([h_channel], [0], None, [256], [0, 256])
s_hist = cv2.calcHist([s_channel], [0], None, [256], [0, 256])
v_hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])

# Normalizar os histogramas
h_hist /= h_hist.sum()
s_hist /= s_hist.sum()
v_hist /= v_hist.sum()

# Configurar os subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Histogramas de Cores HSV')

# Histograma de Matiz (Hue)
axs[0].bar(range(256), h_hist[:,0], color='r', width=1)
axs[0].set_title('Matiz (Hue)')
axs[0].set_xlim([0, 256])
axs[0].set_xlabel('Valor de H')
axs[0].set_ylabel('Frequência Normalizada')

# Histograma de Saturação (Saturation)
axs[1].bar(range(256), s_hist[:,0], color='g', width=1)
axs[1].set_title('Saturação (Saturation)')
axs[1].set_xlim([0, 256])
axs[1].set_xlabel('Valor de S')
axs[1].set_ylabel('Frequência Normalizada')

# Histograma de Valor (Value)
axs[2].bar(range(256), v_hist[:,0], color='b', width=1)
axs[2].set_title('Valor (Value)')
axs[2].set_xlim([0, 256])
axs[2].set_xlabel('Valor de V')
axs[2].set_ylabel('Frequência Normalizada')

# Ajustar o layout
plt.tight_layout()
plt.show()
