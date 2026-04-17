import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize_image_lut(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError("Изображение должно быть 8-битным.")

    i_min = int(image.min())
    i_max = int(image.max())

    if i_min == i_max:
        return image.copy()

    lut = np.zeros(256, dtype=np.uint8)

    for i in range(i_min, i_max + 1):
        value = 255 * (i - i_min) / (i_max - i_min)
        lut[i] = int(round(value))

    normalized = lut[image]
    return normalized


def build_histogram(image: np.ndarray) -> np.ndarray:
    return np.bincount(image.ravel(), minlength=256)


def plot_histogram(hist: np.ndarray, title: str, save_path: str = None):
    plt.figure(figsize=(10, 5))
    plt.bar(range(256), hist, width=1.0, color='gray', edgecolor='black')
    plt.title(title)
    plt.xlabel("Яркость")
    plt.ylabel("Количество пикселей")
    plt.xlim(0, 255)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def find_input_image(input_dir: str) -> Path:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Папка '{input_dir}' не найдена.")

    for file in input_path.iterdir():
        if file.is_file() and file.suffix.lower() in exts:
            return file

    raise FileNotFoundError(f"В папке '{input_dir}' нет изображений.")


def main():
    input_dir = "in"
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)

    input_image_path = find_input_image(input_dir)
    image = cv2.imread(str(input_image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Не удалось загрузить изображение")

    hist_in = build_histogram(image)
    plot_histogram(
        hist_in,
        "Гистограмма исходного изображения",
        os.path.join(output_dir, "hist_input.png")
    )

    normalized = normalize_image_lut(image)

    output_image_path = os.path.join(output_dir, "normalized_image.png")
    cv2.imwrite(output_image_path, normalized)

    hist_out = build_histogram(normalized)
    plot_histogram(
        hist_out,
        "Гистограмма нормализованного изображения",
        os.path.join(output_dir, "hist_output.png")
    )

    print("Обработка завершена.")
    print("Вход:", input_image_path)
    print("Выходное изображение:", output_image_path)


if __name__ == "__main__":
    main()
