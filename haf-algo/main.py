import cv2
import numpy as np
import os


INPUT_DIR = "in"
OUTPUT_DIR = "out"

CANNY_LOW = 50
CANNY_HIGH = 150

THETA_STEP = 1
THRESHOLD_RATIO = 0.5
MAX_LINES = 20


def find_input_image(input_dir):
    """
    Поиск первого изображения в папке in.
    """

    allowed_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    for filename in os.listdir(input_dir):
        lower_name = filename.lower()

        if any(lower_name.endswith(ext) for ext in allowed_extensions):
            return os.path.join(input_dir, filename)

    return None


def hough_transform_lines(edge_image, theta_step=1):
    """
    Ручная реализация преобразования Хафа для поиска прямых.
    """

    height, width = edge_image.shape

    diagonal = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))

    rhos = np.arange(-diagonal, diagonal + 1)
    thetas = np.deg2rad(np.arange(0, 180, theta_step))

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    y_indexes, x_indexes = np.nonzero(edge_image)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for x, y in zip(x_indexes, y_indexes):
        rho_values = x * cos_t + y * sin_t
        rho_indexes = np.round(rho_values).astype(int) + diagonal

        for theta_index, rho_index in enumerate(rho_indexes):
            accumulator[rho_index, theta_index] += 1

    return accumulator, rhos, thetas


def create_hough_space_image(accumulator, peaks=None):
    """
    Создаёт удобную визуализацию пространства Хафа.

    По вертикали: rho.
    По горизонтали: theta.
    Яркие области — большое количество голосов.
    Красные кружки — найденные максимумы.
    """

    THETA_SCALE = 8
    RHO_SCALE = 1

    # Логарифмическое преобразование, чтобы слабые голоса тоже были видны
    accumulator_log = np.log1p(accumulator.astype(np.float32))

    # Обрезаем слишком яркие значения, чтобы один максимум не "убивал" всю картинку
    max_value = np.percentile(accumulator_log, 99.8)

    if max_value > 0:
        accumulator_log = np.clip(accumulator_log, 0, max_value)

    normalized = cv2.normalize(
        accumulator_log,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    height, width = normalized.shape

    # Увеличиваем изображение по горизонтали,
    # потому что theta обычно всего 180 значений
    resized = cv2.resize(
        normalized,
        (width * THETA_SCALE, height * RHO_SCALE),
        interpolation=cv2.INTER_NEAREST
    )

    # Цветовая карта для лучшей видимости
    colored = cv2.applyColorMap(resized, cv2.COLORMAP_INFERNO)

    # Отмечаем найденные пики
    if peaks is not None:
        for votes, rho_index, theta_index in peaks:
            x = theta_index * THETA_SCALE + THETA_SCALE // 2
            y = rho_index * RHO_SCALE + RHO_SCALE // 2

            cv2.circle(colored, (x, y), 8, (0, 255, 0), 2)
            cv2.circle(colored, (x, y), 2, (255, 255, 255), -1)

    return colored



def find_hough_peaks(accumulator, threshold_ratio=0.5, neighborhood_size=15, max_lines=20):
    """
    Поиск локальных максимумов в аккумуляторе Хафа.
    """

    max_value = accumulator.max()
    threshold = max_value * threshold_ratio

    accumulator_float = accumulator.astype(np.float32)

    kernel = np.ones((neighborhood_size, neighborhood_size), np.uint8)
    local_max = cv2.dilate(accumulator_float, kernel)

    peaks_mask = (accumulator_float == local_max) & (accumulator_float >= threshold)

    rho_indexes, theta_indexes = np.where(peaks_mask)

    peaks = []

    for rho_index, theta_index in zip(rho_indexes, theta_indexes):
        votes = accumulator[rho_index, theta_index]
        peaks.append((votes, rho_index, theta_index))

    peaks.sort(reverse=True, key=lambda item: item[0])

    return peaks[:max_lines]


def draw_hough_lines(image, peaks, rhos, thetas):
    """
    Отрисовка найденных прямых на исходном изображении.
    """

    result = image.copy()

    for votes, rho_index, theta_index in peaks:
        rho = rhos[rho_index]
        theta = thetas[theta_index]

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x0 = cos_t * rho
        y0 = sin_t * rho

        x1 = int(x0 + 1000 * (-sin_t))
        y1 = int(y0 + 1000 * cos_t)

        x2 = int(x0 - 1000 * (-sin_t))
        y2 = int(y0 - 1000 * cos_t)

        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return result


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = find_input_image(INPUT_DIR)

    if input_path is None:
        print("В папке in не найдено изображение.")
        print("Положи туда файл .jpg, .jpeg, .png или .bmp.")
        return

    image = cv2.imread(input_path)

    if image is None:
        print("Не удалось открыть изображение.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    accumulator, rhos, thetas = hough_transform_lines(edges, THETA_STEP)

    peaks = find_hough_peaks(
        accumulator,
        threshold_ratio=THRESHOLD_RATIO,
        neighborhood_size=15,
        max_lines=MAX_LINES
    )

    hough_space_image = create_hough_space_image(accumulator, peaks)

    result = draw_hough_lines(image, peaks, rhos, thetas)

    contour_path = os.path.join(OUTPUT_DIR, "contour.png")
    hough_space_path = os.path.join(OUTPUT_DIR, "hough_space.png")
    hough_result_path = os.path.join(OUTPUT_DIR, "hough_result.png")

    cv2.imwrite(contour_path, edges)
    cv2.imwrite(hough_space_path, hough_space_image)
    cv2.imwrite(hough_result_path, result)

    print("Готово.")
    print(f"Входное изображение: {input_path}")
    print(f"Контур: {contour_path}")
    print(f"Преобразование Хафа: {hough_space_path}")
    print(f"Результат с прямыми: {hough_result_path}")
    print(f"Найдено прямых: {len(peaks)}")


if __name__ == "__main__":
    main()
