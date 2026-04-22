import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def calculate_histogram(image):
    hist = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        hist[pixel] += 1
    return hist


def otsu_threshold(image):
    hist = calculate_histogram(image)
    total_pixels = image.size

    max_sc = 0
    optimal_threshold = 0

    for t in range(256):
        # DISP(0, t) - дисперсия для пикселей [0..t]
        disp_0_t = 0
        count_0_t = np.sum(hist[0:t + 1])

        if count_0_t > 0:
            mo_0_t = np.sum([i * hist[i] for i in range(t + 1)]) / count_0_t
            disp_0_t = np.sum([hist[i] * (i - mo_0_t) ** 2 for i in range(t + 1)])

        # DISP(t+1, 255) - дисперсия для пикселей [t+1..255]
        disp_t1_255 = 0
        count_t1_255 = np.sum(hist[t + 1:256])

        if count_t1_255 > 0:
            mo_t1_255 = np.sum([i * hist[i] for i in range(t + 1, 256)]) / count_t1_255
            disp_t1_255 = np.sum([hist[i] * (i - mo_t1_255) ** 2 for i in range(t + 1, 256)])

        # DISP(0, 255) - общая дисперсия
        mo_total = np.sum([i * hist[i] for i in range(256)]) / total_pixels
        disp_total = np.sum([hist[i] * (i - mo_total) ** 2 for i in range(256)])

        # Критерий разделимости SC(t)
        if disp_total > 0:
            sc = 1 - (disp_0_t + disp_t1_255) / disp_total

            if sc > max_sc:
                max_sc = sc
                optimal_threshold = t

    return optimal_threshold


def binarize_image(image, threshold):
    binary = np.zeros_like(image)
    binary[image > threshold] = 255
    return binary


def save_histogram(image, threshold, filename, title):
    plt.figure(figsize=(10, 6))
    plt.hist(image.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Порог = {threshold}')
    plt.title(title)
    plt.xlabel('Значение яркости')
    plt.ylabel('Количество пикселей')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def process_image(input_path, output_folder, global_thresh=128):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {input_path}")
        return
    print(f"Изображение загружено: {image.shape}")
    cv2.imwrite(os.path.join(output_folder, '1_original.png'), image)

    binary_global = binarize_image(image, global_thresh)
    cv2.imwrite(os.path.join(output_folder, '2_binary_global.png'), binary_global)
    save_histogram(image, global_thresh,
                   os.path.join(output_folder, '3_histogram_global.png'),
                   f'Гистограмма (Глобальный порог = {global_thresh})')
    print(f"Глобальная бинаризация: порог = {global_thresh}")

    otsu_thresh = otsu_threshold(image)
    binary_otsu = binarize_image(image, otsu_thresh)
    cv2.imwrite(os.path.join(output_folder, '4_binary_otsu.png'), binary_otsu)
    save_histogram(image, otsu_thresh,
                   os.path.join(output_folder, '5_histogram_otsu.png'),
                   f'Гистограмма (Метод Отсу, порог = {otsu_thresh})')
    print(f"Метод Отсу: оптимальный порог = {otsu_thresh}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Исходное изображение', fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(binary_global, cmap='gray')
    axes[0, 1].set_title(f'Глобальная бинаризация (порог={global_thresh})', fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(binary_otsu, cmap='gray')
    axes[0, 2].set_title(f'Бинаризация Отсу (порог={otsu_thresh})', fontsize=14)
    axes[0, 2].axis('off')

    axes[1, 0].hist(image.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    axes[1, 0].set_title('Гистограмма исходного изображения', fontsize=14)
    axes[1, 0].set_xlabel('Яркость')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(image.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    axes[1, 1].axvline(global_thresh, color='r', linestyle='--', linewidth=2, label=f'Порог={global_thresh}')
    axes[1, 1].set_title('Гистограмма с глобальным порогом', fontsize=14)
    axes[1, 1].set_xlabel('Яркость')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(image.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)
    axes[1, 2].axvline(otsu_thresh, color='r', linestyle='--', linewidth=2, label=f'Порог={otsu_thresh}')
    axes[1, 2].set_title('Гистограмма с порогом Отсу', fontsize=14)
    axes[1, 2].set_xlabel('Яркость')
    axes[1, 2].set_ylabel('Частота')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '6_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    input_image = 'in/input_image.png'
    output_folder = 'output_results'
    process_image(input_image, output_folder, 60)