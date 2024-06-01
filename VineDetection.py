import cv2
import numpy as np


def adjust_colors(image):
    b = image[:, :, 0] * 1.3  # Aumenta o canal azul
    g = image[:, :, 1] * 1.3  # Aumenta o canal verde
    r = image[:, :, 2] * 1.3  # Aumenta o canal vermelho

    # Garante que os valores permaneçam no intervalo válido de 0 a 255
    b = np.clip(b, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    r = np.clip(r, 0, 255).astype(np.uint8)

    # Combina os canais de volta em uma imagem
    result = cv2.merge((b, g, r))

    return result


def detect_color(frame, color_lower, color_upper, color_name):
    # Convertendo a imagem para o espaço de cores HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Criando uma máscara usando os intervalos de cores especificados
    mask = cv2.inRange(hsv_frame, color_lower, color_upper)

    # Aplicando operação de abertura para reduzir o ruído
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Encontrando contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 800:  # Ajuste o limite da área conforme necessário
            detected_contours.append(contour)
            color = (0, 255, 0) if color_name == 'Leaf' else (255, 0, 0)  # Verde para Folhas, Azul para Uvas
            cv2.drawContours(frame, [contour], -1, color, 2)
            cv2.putText(frame, color_name, tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, detected_contours


video_path = 'G:/Nova pasta (2)/video5.mp4'
output_path = 'G:/Nova pasta (2)/video5_processed.mp4'

cap = cv2.VideoCapture(video_path)

# Obtendo informações do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Configurando o objeto de gravação de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ajuste de cores
    frame = adjust_colors(frame)

    # Definindo intervalo de cores para folhas verdes e uvas azuis
    lower_green_leaf = np.array([20, 50, 50])  # Verde escuro
    upper_green_leaf = np.array([100, 255, 255])  # Verde claro
    lower_purple_grape = np.array([50, 0, 0])  # Roxo escuro
    upper_purple_grape = np.array([255, 255, 255])  # Roxo claro

    # Detectando folhas
    frame, leaf_contours = detect_color(frame, lower_green_leaf, upper_green_leaf, 'Leaf')

    # Detectando cachos de uvas azuis
    frame, blue_grape_contours = detect_color(frame, lower_purple_grape, upper_purple_grape, 'Grape')

    # Gravando o frame processado
    out.write(frame)

    cv2.imshow('Color Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
