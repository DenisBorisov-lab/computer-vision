import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_DIR = Path("in")
OUTPUT_DIR = Path("out")

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def prepare_binary_image(gray_image):
    """
    Приводит изображение к бинарному виду.
    Если фон белый, а объекты чёрные — изображение инвертируется,
    чтобы объекты стали белыми, а фон чёрным.
    """

    if gray_image is None:
        raise ValueError("Изображение не удалось прочитать")

    # Бинаризация методом Оцу
    _, binary = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Предполагаем, что объект занимает меньшую часть изображения.
    # Если белых пикселей больше половины, значит фон, скорее всего, белый.
    white_pixels = np.count_nonzero(binary == 255)
    total_pixels = binary.size

    if white_pixels > total_pixels / 2:
        binary = cv2.bitwise_not(binary)

    return binary


def colorize_labels(labels, background_labels=None):
    """
    Преобразует карту меток сегментов в цветное изображение.
    """

    if background_labels is None:
        background_labels = {0}

    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label in background_labels:
            continue

        if label == -1:
            # Границы watershed
            colored[labels == label] = [255, 0, 0]
            continue

        # Детерминированная генерация цвета по номеру метки
        r = (37 * int(label)) % 255
        g = (91 * int(label)) % 255
        b = (173 * int(label)) % 255

        colored[labels == label] = [r, g, b]

    return colored


def segmentation_connected_components(binary, connectivity=4):
    """
    Сегментация методом связных компонент.
    connectivity=4 или connectivity=8.
    """

    number_of_labels, labels = cv2.connectedComponents(
        binary,
        connectivity=connectivity
    )

    colored = colorize_labels(labels, background_labels={0})

    return labels, colored, number_of_labels - 1


def segmentation_contours(binary):
    """
    Сегментация через поиск контуров.
    Каждый внешний контур считается отдельным сегментом.
    """

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    labels = np.zeros(binary.shape, dtype=np.int32)

    for i, contour in enumerate(contours, start=1):
        cv2.drawContours(
            labels,
            [contour],
            contourIdx=-1,
            color=i,
            thickness=-1
        )

    colored = colorize_labels(labels, background_labels={0})

    return labels, colored, len(contours)


def segmentation_watershed(gray_image, binary):
    """
    Сегментация методом watershed.
    Может разделять соприкасающиеся объекты.
    """

    # Убираем мелкий шум
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Определяем уверенный фон
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Определяем уверенный передний план через distance transform
    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    if dist_transform.max() == 0:
        labels = np.zeros(binary.shape, dtype=np.int32)
        colored = colorize_labels(labels)
        return labels, colored, 0

    _, sure_fg = cv2.threshold(
        dist_transform,
        0.35 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    # Неизвестная область
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Маркеры
    number_of_labels, markers = cv2.connectedComponents(sure_fg)

    # Чтобы фон был не 0, а 1
    markers = markers + 1

    # Неизвестную область помечаем 0
    markers[unknown == 255] = 0

    # Watershed требует цветное изображение
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    markers = cv2.watershed(color_image, markers)

    colored = colorize_labels(
        markers,
        background_labels={0, 1}
    )

    # Количество сегментов: метки больше 1, кроме границы -1
    segment_labels = [
        label for label in np.unique(markers)
        if label not in {-1, 0, 1}
    ]

    return markers, colored, len(segment_labels)


def save_image_rgb(path, image_rgb):
    """
    Сохраняет RGB-изображение через OpenCV.
    """

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def create_comparison(
    original_gray,
    binary,
    results,
    output_path,
    title
):
    """
    Создаёт общее изображение-сравнение:
    исходное изображение, бинарное изображение и результаты алгоритмов.
    """

    columns = 2 + len(results)

    plt.figure(figsize=(4 * columns, 5))

    plt.subplot(1, columns, 1)
    plt.imshow(original_gray, cmap="gray")
    plt.title("Исходное")
    plt.axis("off")

    plt.subplot(1, columns, 2)
    plt.imshow(binary, cmap="gray")
    plt.title("Бинарное")
    plt.axis("off")

    for i, result in enumerate(results, start=3):
        algorithm_name = result["name"]
        segmented_image = result["image"]
        count = result["count"]

        plt.subplot(1, columns, i)
        plt.imshow(segmented_image)
        plt.title(f"{algorithm_name}\nСегментов: {count}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def process_image(image_path):
    """
    Обрабатывает одно изображение.
    """

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"Не удалось открыть файл: {image_path}")
        return

    binary = prepare_binary_image(gray)

    results = []

    labels_4, colored_4, count_4 = segmentation_connected_components(
        binary,
        connectivity=4
    )

    results.append({
        "name": "CC 4-связность",
        "labels": labels_4,
        "image": colored_4,
        "count": count_4
    })

    labels_8, colored_8, count_8 = segmentation_connected_components(
        binary,
        connectivity=8
    )

    results.append({
        "name": "CC 8-связность",
        "labels": labels_8,
        "image": colored_8,
        "count": count_8
    })

    labels_contours, colored_contours, count_contours = segmentation_contours(
        binary
    )

    results.append({
        "name": "Контуры",
        "labels": labels_contours,
        "image": colored_contours,
        "count": count_contours
    })

    labels_watershed, colored_watershed, count_watershed = segmentation_watershed(
        gray,
        binary
    )

    results.append({
        "name": "Watershed",
        "labels": labels_watershed,
        "image": colored_watershed,
        "count": count_watershed
    })

    stem = image_path.stem

    # Сохраняем бинарное изображение
    cv2.imwrite(
        str(OUTPUT_DIR / f"{stem}_binary.png"),
        binary
    )

    # Сохраняем результат каждого алгоритма отдельно
    for result in results:
        safe_name = (
            result["name"]
            .replace(" ", "_")
            .replace("-", "_")
            .replace("связность", "svyaznost")
        )

        save_image_rgb(
            OUTPUT_DIR / f"{stem}_{safe_name}.png",
            result["image"]
        )

    # Сохраняем общее сравнение
    create_comparison(
        original_gray=gray,
        binary=binary,
        results=results,
        output_path=OUTPUT_DIR / f"{stem}_comparison.png",
        title=f"Сегментация изображения: {image_path.name}"
    )

    print(f"Обработано: {image_path.name}")

    for result in results:
        print(f"  {result['name']}: сегментов = {result['count']}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not INPUT_DIR.exists():
        print("Папка 'in' не найдена.")
        print("Создай папку 'in' и помести туда изображения.")
        return

    image_files = [
        file for file in INPUT_DIR.iterdir()
        if file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print("В папке 'in' нет изображений.")
        return

    for image_path in image_files:
        process_image(image_path)

    print("\nГотово. Результаты сохранены в папку 'out'.")


if __name__ == "__main__":
    main()
