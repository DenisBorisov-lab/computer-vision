import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_grayscale_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def save_grayscale_image(image: np.ndarray, path: str) -> None:
    Image.fromarray(image).save(path)


def find_median_from_hist(hist: np.ndarray, window_area: int) -> int:
    target = window_area // 2 + 1
    cumulative = 0

    for value in range(256):
        cumulative += hist[value]
        if cumulative >= target:
            return value
    return 255


def median_filter_histogram(image: np.ndarray, ksize: int = 3, padding_mode: str = "edge") -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Ожидается полутоновое изображение.")
    if image.dtype != np.uint8:
        raise ValueError("Ожидается изображение типа uint8.")
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError("Размер окна должен быть положительным нечетным числом.")

    pad = ksize // 2
    h, w = image.shape
    padded = np.pad(image, pad_width=pad, mode=padding_mode)
    result = np.empty((h, w), dtype=np.uint8)

    window_area = ksize * ksize

    for i in range(h):
        hist = np.zeros(256, dtype=np.int32)

        for r in range(ksize):
            for c in range(ksize):
                hist[padded[i + r, c]] += 1

        result[i, 0] = find_median_from_hist(hist, window_area)

        for j in range(1, w):
            left_col = j - 1
            right_col = j + ksize - 1

            for r in range(ksize):
                hist[padded[i + r, left_col]] -= 1
                hist[padded[i + r, right_col]] += 1

            result[i, j] = find_median_from_hist(hist, window_area)

    return result


def get_central_column_profile(image: np.ndarray) -> np.ndarray:
    _, w = image.shape
    center_col = w // 2
    return image[:, center_col]


def get_central_row_profile(image: np.ndarray) -> np.ndarray:
    h, _ = image.shape
    center_row = h // 2
    return image[center_row, :]


def save_profile_plot(
    original_profile: np.ndarray,
    filtered_profile: np.ndarray,
    output_path: str,
    title: str,
    x_label: str
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(
        original_profile,
        label="Зашумленное изображение",
        color="black",
        linewidth=1.4
    )
    plt.plot(
        filtered_profile,
        label="После медианной фильтрации",
        color="red",
        linewidth=1.4
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Яркость")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_image_comparison(noisy: np.ndarray, filtered: np.ndarray, output_path: str, ksize: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Зашумленное изображение")
    axes[0].axis("off")

    axes[1].imshow(filtered, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Медианная фильтрация {ksize}x{ksize}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_profile_comparison_figure(
    noisy: np.ndarray,
    filtered: np.ndarray,
    noisy_col_profile: np.ndarray,
    filtered_col_profile: np.ndarray,
    noisy_row_profile: np.ndarray,
    filtered_row_profile: np.ndarray,
    output_path: str,
    ksize: int
) -> None:
    h, w = noisy.shape
    center_col = w // 2
    center_row = h // 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(noisy, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].axvline(center_col, color="lime", linewidth=1.5, label="Центральный столбец")
    axes[0, 0].axhline(center_row, color="magenta", linewidth=1.5, label="Центральная строка")
    axes[0, 0].set_title("Зашумленное изображение")
    axes[0, 0].axis("off")
    axes[0, 0].legend(loc="lower right", fontsize=8)

    axes[0, 1].imshow(filtered, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].axvline(center_col, color="lime", linewidth=1.5, label="Центральный столбец")
    axes[0, 1].axhline(center_row, color="magenta", linewidth=1.5, label="Центральная строка")
    axes[0, 1].set_title(f"После медианной фильтрации {ksize}x{ksize}")
    axes[0, 1].axis("off")
    axes[0, 1].legend(loc="lower right", fontsize=8)

    axes[1, 0].plot(
        noisy_col_profile,
        label="Зашумленное изображение",
        color="black",
        linewidth=1.4
    )
    axes[1, 0].plot(
        filtered_col_profile,
        label="После медианной фильтрации",
        color="red",
        linewidth=1.4
    )
    axes[1, 0].set_title("Профиль по центральному столбцу")
    axes[1, 0].set_xlabel("Номер строки")
    axes[1, 0].set_ylabel("Яркость")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend()

    axes[1, 1].plot(
        noisy_row_profile,
        label="Зашумленное изображение",
        color="black",
        linewidth=1.4
    )
    axes[1, 1].plot(
        filtered_row_profile,
        label="После медианной фильтрации",
        color="red",
        linewidth=1.4
    )
    axes[1, 1].set_title("Профиль по центральной строке")
    axes[1, 1].set_xlabel("Номер столбца")
    axes[1, 1].set_ylabel("Яркость")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    input_path = "in/img.png"
    output_dir = "out"
    ksize = 5

    os.makedirs(output_dir, exist_ok=True)

    noisy_image = load_grayscale_image(input_path)
    filtered_image = median_filter_histogram(noisy_image, ksize=ksize, padding_mode="edge")

    noisy_col_profile = get_central_column_profile(noisy_image)
    filtered_col_profile = get_central_column_profile(filtered_image)

    noisy_row_profile = get_central_row_profile(noisy_image)
    filtered_row_profile = get_central_row_profile(filtered_image)

    filtered_image_path = os.path.join(output_dir, f"filtered_median_{ksize}x{ksize}.png")
    comparison_image_path = os.path.join(output_dir, f"comparison_images_{ksize}x{ksize}.png")
    col_profile_plot_path = os.path.join(output_dir, f"profile_central_column_{ksize}x{ksize}.png")
    row_profile_plot_path = os.path.join(output_dir, f"profile_central_row_{ksize}x{ksize}.png")
    summary_figure_path = os.path.join(output_dir, f"summary_{ksize}x{ksize}.png")

    save_grayscale_image(filtered_image, filtered_image_path)
    save_image_comparison(noisy_image, filtered_image, comparison_image_path, ksize)

    save_profile_plot(
        noisy_col_profile,
        filtered_col_profile,
        col_profile_plot_path,
        title="Профиль по центральному столбцу",
        x_label="Номер строки"
    )

    save_profile_plot(
        noisy_row_profile,
        filtered_row_profile,
        row_profile_plot_path,
        title="Профиль по центральной строке",
        x_label="Номер столбца"
    )

    save_profile_comparison_figure(
        noisy_image,
        filtered_image,
        noisy_col_profile,
        filtered_col_profile,
        noisy_row_profile,
        filtered_row_profile,
        summary_figure_path,
        ksize
    )

    print("Обработка завершена.")
    print(f"Сохранено отфильтрованное изображение: {filtered_image_path}")
    print(f"Сохранено сравнение изображений: {comparison_image_path}")
    print(f"Сохранен профиль по центральному столбцу: {col_profile_plot_path}")
    print(f"Сохранен профиль по центральной строке: {row_profile_plot_path}")
    print(f"Сохранена итоговая сводка: {summary_figure_path}")


if __name__ == "__main__":
    main()
