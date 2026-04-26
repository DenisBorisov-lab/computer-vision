import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_grayscale_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    return image.astype(np.float32)


def add_additive_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 20.0) -> np.ndarray:
    noise = np.random.normal(loc=mean, scale=sigma, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def save_image(image: np.ndarray, path: str):
    success = cv2.imwrite(path, image)
    if not success:
        raise ValueError(f"Не удалось сохранить изображение: {path}")


def show_images(original: np.ndarray, noisy: np.ndarray):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray', vmin=0, vmax=255)
    plt.title('Исходное изображение')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(noisy, cmap='gray', vmin=0, vmax=255)
    plt.title('Изображение с аддитивным шумом')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    input_path = 'image.png'
    output_path = 'noisy_image.png'

    image = load_grayscale_image(input_path)
    noisy_image = add_additive_gaussian_noise(image, mean=0, sigma=50)

    save_image(noisy_image, output_path)
    print(f'Зашумленное изображение сохранено: {output_path}')

    show_images(image.astype(np.uint8), noisy_image)


if __name__ == '__main__':
    main()
