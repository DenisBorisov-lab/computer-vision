import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT_DIR = "in"
OUTPUT_DIR = "out"
OUTPUT_IMAGE_NAME = "equalized.png"
OUTPUT_HIST_BEFORE = "hist_before.png"
OUTPUT_HIST_AFTER = "hist_after.png"


def ensure_output_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def find_input_image(input_dir: str) -> str:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(exts):
            return os.path.join(input_dir, file_name)
    raise FileNotFoundError(f"В папке '{input_dir}' не найдено изображение.")


def compute_histogram(image: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for value in image.ravel():
        hist[value] += 1
    return hist


def equalize_image(image: np.ndarray) -> np.ndarray:
    hist = compute_histogram(image)

    cdf = np.cumsum(hist)

    nonzero = np.nonzero(cdf)[0]
    if len(nonzero) == 0:
        return image.copy()

    cdf_min = cdf[nonzero[0]]
    total_pixels = image.size

    lut = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        if cdf[i] < cdf_min:
            lut[i] = 0
        else:
            value = round((cdf[i] - cdf_min) * 255 / (total_pixels - cdf_min)) if total_pixels != cdf_min else 0
            lut[i] = np.clip(value, 0, 255)

    equalized = lut[image]
    return equalized


def save_histogram(hist: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(256), hist, width=1.0, color="gray")
    plt.title(title)
    plt.xlabel("Яркость")
    plt.ylabel("Количество пикселей")
    plt.xlim(0, 255)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    ensure_output_dir(OUTPUT_DIR)

    image_path = find_input_image(INPUT_DIR)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    hist_before = compute_histogram(image)
    save_histogram(hist_before, "Гистограмма исходного изображения", os.path.join(OUTPUT_DIR, OUTPUT_HIST_BEFORE))

    equalized_image = equalize_image(image)

    output_image_path = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE_NAME)
    cv2.imwrite(output_image_path, equalized_image)

    hist_after = compute_histogram(equalized_image)
    save_histogram(hist_after, "Гистограмма эквализованного изображения", os.path.join(OUTPUT_DIR, OUTPUT_HIST_AFTER))

    print(f"Исходное изображение: {image_path}")
    print(f"Эквализованное изображение сохранено: {output_image_path}")
    print(f"Гистограмма до эквализации: {os.path.join(OUTPUT_DIR, OUTPUT_HIST_BEFORE)}")
    print(f"Гистограмма после эквализации: {os.path.join(OUTPUT_DIR, OUTPUT_HIST_AFTER)}")


if __name__ == "__main__":
    main()
