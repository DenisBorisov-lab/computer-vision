import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_grayscale_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    return image.astype(np.float32)


def save_image(image: np.ndarray, path: str):
    success = cv2.imwrite(path, image)
    if not success:
        raise ValueError(f"Не удалось сохранить изображение: {path}")


def add_additive_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    noise = np.random.normal(loc=mean, scale=sigma, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def apply_linear_filter(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'reflect') -> np.ndarray:
    image = image.astype(np.float32)

    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=padding_mode)
    result = np.zeros_like(image, dtype=np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            window = padded[y:y + kh, x:x + kw]
            result[y, x] = np.sum(window * kernel)

    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def mean_kernel(size: int) -> np.ndarray:
    if size % 2 == 0 or size < 1:
        raise ValueError("Размер окна должен быть положительным нечётным числом")
    return np.ones((size, size), dtype=np.float32) / (size * size)


def gaussian_kernel(size: int, sigma: float = 1.2) -> np.ndarray:
    if size % 2 == 0 or size < 1:
        raise ValueError("Размер окна должен быть положительным нечётным числом")

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def middle_horizontal_profile(image: np.ndarray) -> np.ndarray:
    middle_row = image.shape[0] // 2
    return image[middle_row, :]


def middle_vertical_profile(image: np.ndarray) -> np.ndarray:
    middle_col = image.shape[1] // 2
    return image[:, middle_col]


def save_comparison_figure_with_profiles(original: np.ndarray, filtered: np.ndarray, title: str, output_path: str):
    h_profile_orig = middle_horizontal_profile(original)
    h_profile_filt = middle_horizontal_profile(filtered)

    v_profile_orig = middle_vertical_profile(original)
    v_profile_filt = middle_vertical_profile(filtered)

    diff = cv2.absdiff(original, filtered)

    middle_row = original.shape[0] // 2
    middle_col = original.shape[1] // 2

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(title, fontsize=14)

    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].axhline(middle_row, color='yellow', linestyle='--', linewidth=1, label='Центральная строка')
    axes[0, 0].axvline(middle_col, color='cyan', linestyle='--', linewidth=1, label='Центральный столбец')
    axes[0, 0].set_title('Исходное зашумленное изображение')
    axes[0, 0].axis('off')
    axes[0, 0].legend(loc='lower right', fontsize=8)

    axes[0, 1].imshow(filtered, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].axhline(middle_row, color='yellow', linestyle='--', linewidth=1)
    axes[0, 1].axvline(middle_col, color='cyan', linestyle='--', linewidth=1)
    axes[0, 1].set_title('Отфильтрованное изображение')
    axes[0, 1].axis('off')

    axes[1, 0].plot(h_profile_orig, color='gray', label='Исходное')
    axes[1, 0].plot(h_profile_filt, color='blue', label='После фильтрации')
    axes[1, 0].set_title('Профиль центральной строки')
    axes[1, 0].set_xlabel('Номер столбца')
    axes[1, 0].set_ylabel('Яркость')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(v_profile_orig, color='gray', label='Исходное')
    axes[1, 1].plot(v_profile_filt, color='red', label='После фильтрации')
    axes[1, 1].set_title('Профиль центрального столбца')
    axes[1, 1].set_xlabel('Номер строки')
    axes[1, 1].set_ylabel('Яркость')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    input_path = 'in/img.png'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    original = load_grayscale_image(input_path)

    noisy = add_additive_gaussian_noise(original, mean=0, sigma=40, seed=42)
    save_image(noisy, os.path.join(output_dir, '01_noisy_image.png'))

    avg_kernel = mean_kernel(5)
    avg_filtered = apply_linear_filter(noisy, avg_kernel)
    save_image(avg_filtered, os.path.join(output_dir, '02_mean_filtered_5x5.png'))

    gauss_kernel = gaussian_kernel(size=5, sigma=1.2)
    gauss_filtered = apply_linear_filter(noisy, gauss_kernel)
    save_image(gauss_filtered, os.path.join(output_dir, '03_gaussian_filtered_5x5.png'))

    save_comparison_figure_with_profiles(
        noisy,
        avg_filtered,
        'Скользящее среднее 5x5',
        os.path.join(output_dir, '04_comparison_mean_5x5.png')
    )

    save_comparison_figure_with_profiles(
        noisy,
        gauss_filtered,
        'Гауссова фильтрация 5x5',
        os.path.join(output_dir, '05_comparison_gaussian_5x5.png')
    )


if __name__ == '__main__':
    main()