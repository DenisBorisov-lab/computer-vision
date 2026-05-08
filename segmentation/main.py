from pathlib import Path
import cv2
import numpy as np

try:
    from skimage.filters import threshold_multiotsu
except ImportError:
    threshold_multiotsu = None


# =========================
# Настройки программы
# =========================

IN_DIR = Path("in")
OUT_DIR = Path("out")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Для мультипороговой сегментации
MULTI_CLASSES = 4

# Для диапазонной сегментации.
# Используются процентили яркости изображения.
RANGE_PERCENTILES = (35, 65)

# Для метода слияния областей
MERGE_BLOCK_SIZE = 8
MERGE_MEAN_DIFF = 10

# Для метода разбиения областей
SPLIT_MIN_SIZE = 16
SPLIT_STD_THRESHOLD = 12

# Для метода слияния/разбиения
SPLIT_MERGE_DIFF = 10


# =========================
# Вспомогательные функции
# =========================

def read_image(path: Path):
    """
    Чтение изображения с поддержкой путей с кириллицей.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def save_image(path: Path, image: np.ndarray):
    """
    Сохранение изображения с поддержкой путей с кириллицей.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix
    success, encoded = cv2.imencode(ext, image)
    if success:
        encoded.tofile(str(path))


def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Перевод изображения в полутоновый формат.
    """
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gray_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Перевод полутонового изображения в BGR для сохранения сравнений.
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def add_caption(image: np.ndarray, text: str) -> np.ndarray:
    """
    Добавляет подпись над изображением.
    """
    img = gray_to_bgr(image)
    h, w = img.shape[:2]

    caption_height = 45
    result = np.full((h + caption_height, w, 3), 255, dtype=np.uint8)
    result[caption_height:, :] = img

    cv2.putText(
        result,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return result


def make_comparison(original_gray: np.ndarray, segmented: np.ndarray, method_name: str) -> np.ndarray:
    """
    Создает изображение сравнения: исходное слева, результат справа.
    """
    left = add_caption(original_gray, "Original")
    right = add_caption(segmented, method_name)

    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]

    if h1 != h2:
        right = cv2.resize(right, (w2, h1))

    return cv2.hconcat([left, right])


def render_labels_by_mean(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Преобразует карту меток областей в полутоновое изображение,
    где каждая область закрашена средней яркостью этой области.
    """
    labels = labels.astype(np.int32)
    max_label = int(labels.max()) + 1

    sums = np.bincount(labels.ravel(), weights=gray.ravel(), minlength=max_label)
    counts = np.bincount(labels.ravel(), minlength=max_label)

    means = np.zeros(max_label, dtype=np.float32)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]

    result = means[labels]
    return np.clip(result, 0, 255).astype(np.uint8)


def collect_neighbor_pairs(labels: np.ndarray):
    """
    Собирает пары соседних областей по горизонтальным и вертикальным границам.
    """
    pairs = set()

    # Горизонтальные соседи
    left = labels[:, :-1]
    right = labels[:, 1:]
    mask = left != right

    for a, b in zip(left[mask].ravel(), right[mask].ravel()):
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        pairs.add((a, b))

    # Вертикальные соседи
    top = labels[:-1, :]
    bottom = labels[1:, :]
    mask = top != bottom

    for a, b in zip(top[mask].ravel(), bottom[mask].ravel()):
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        pairs.add((a, b))

    return list(pairs)


# =========================
# 1. Пороговая сегментация
# =========================

def threshold_segmentation(gray: np.ndarray) -> np.ndarray:
    """
    Пороговая сегментация по методу Оцу.

    g(i,j) = 1, если f(i,j) >= T
    g(i,j) = 0, если f(i,j) < T
    """
    _, result = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return result


# =========================
# 2. Диапазонная сегментация
# =========================

def range_threshold_segmentation(gray: np.ndarray) -> np.ndarray:
    """
    Диапазонная пороговая сегментация.

    Пиксель относится к области, если его яркость входит в диапазон D.
    Диапазон выбирается автоматически через процентили яркости.
    """
    low, high = np.percentile(gray, RANGE_PERCENTILES)

    result = np.zeros_like(gray, dtype=np.uint8)
    result[(gray >= low) & (gray <= high)] = 255

    return result


# =========================
# 3. Мультипороговая сегментация
# =========================

def multi_threshold_segmentation(gray: np.ndarray, classes: int = MULTI_CLASSES) -> np.ndarray:
    """
    Мультипороговая сегментация.

    Яркости разбиваются на несколько диапазонов D1, D2, ..., Dn.
    """
    if classes < 2:
        raise ValueError("Количество классов должно быть не меньше 2.")

    if threshold_multiotsu is not None:
        thresholds = threshold_multiotsu(gray, classes=classes)
    else:
        # Запасной вариант без scikit-image
        percentiles = np.linspace(0, 100, classes + 1)[1:-1]
        thresholds = np.percentile(gray, percentiles)

    regions = np.digitize(gray, bins=thresholds)

    result = regions.astype(np.float32)
    result = result * (255 / (classes - 1))

    return result.astype(np.uint8)


# =========================
# 4. Слияние областей
# =========================

class DSU:
    """
    Структура данных для объединения областей.
    Disjoint Set Union / система непересекающихся множеств.
    """
    def __init__(self, n: int, sums: np.ndarray, counts: np.ndarray):
        self.parent = np.arange(n, dtype=np.int32)
        self.sums = sums.astype(np.float64)
        self.counts = counts.astype(np.float64)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def mean(self, x: int) -> float:
        root = self.find(x)
        if self.counts[root] == 0:
            return 0.0
        return self.sums[root] / self.counts[root]

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return False

        # Присоединяем меньшую область к большей
        if self.counts[ra] < self.counts[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.sums[ra] += self.sums[rb]
        self.counts[ra] += self.counts[rb]

        return True


def make_grid_labels(height: int, width: int, block_size: int) -> np.ndarray:
    """
    Начальная пресегментация изображения на квадратные блоки.
    """
    labels = np.zeros((height, width), dtype=np.int32)

    label = 0
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            labels[y:y + block_size, x:x + block_size] = label
            label += 1

    return labels


def merge_regions_from_labels(
    gray: np.ndarray,
    labels: np.ndarray,
    mean_diff_threshold: float,
    max_iterations: int = 100
):
    """
    Итеративное слияние соседних областей.

    Две области объединяются, если разница их средних яркостей
    не превышает mean_diff_threshold.
    """
    labels = labels.astype(np.int32)
    n_labels = int(labels.max()) + 1

    sums = np.bincount(labels.ravel(), weights=gray.ravel(), minlength=n_labels)
    counts = np.bincount(labels.ravel(), minlength=n_labels)

    dsu = DSU(n_labels, sums, counts)
    neighbor_pairs = collect_neighbor_pairs(labels)

    for _ in range(max_iterations):
        changed = False

        for a, b in neighbor_pairs:
            ra = dsu.find(a)
            rb = dsu.find(b)

            if ra == rb:
                continue

            mean_a = dsu.mean(ra)
            mean_b = dsu.mean(rb)

            if abs(mean_a - mean_b) <= mean_diff_threshold:
                dsu.union(ra, rb)
                changed = True

        if not changed:
            break

    root_map = np.array([dsu.find(i) for i in range(n_labels)], dtype=np.int32)
    merged_labels = root_map[labels]

    result = render_labels_by_mean(gray, merged_labels)

    return result, merged_labels


def region_merging_segmentation(gray: np.ndarray) -> np.ndarray:
    """
    Метод слияния областей.

    1. Выполняется начальная сегментация на блоки.
    2. Соседние области объединяются, если они похожи по средней яркости.
    """
    h, w = gray.shape[:2]
    initial_labels = make_grid_labels(h, w, MERGE_BLOCK_SIZE)

    result, _ = merge_regions_from_labels(
        gray,
        initial_labels,
        mean_diff_threshold=MERGE_MEAN_DIFF
    )

    return result


# =========================
# 5. Разбиение областей
# =========================

def quadtree_split_labels(
    gray: np.ndarray,
    min_size: int = SPLIT_MIN_SIZE,
    std_threshold: float = SPLIT_STD_THRESHOLD
) -> np.ndarray:
    """
    Разбиение изображения с использованием дерева квадрантов.

    Если область неоднородна, то есть стандартное отклонение яркости
    больше std_threshold, область делится на 4 части.
    """
    h, w = gray.shape[:2]
    labels = np.full((h, w), -1, dtype=np.int32)
    current_label = 0

    def split(y0: int, y1: int, x0: int, x1: int):
        nonlocal current_label

        region = gray[y0:y1, x0:x1]
        region_h = y1 - y0
        region_w = x1 - x0

        if region_h <= 0 or region_w <= 0:
            return

        is_small = region_h <= min_size or region_w <= min_size
        is_homogeneous = np.std(region) <= std_threshold

        if is_small or is_homogeneous:
            labels[y0:y1, x0:x1] = current_label
            current_label += 1
            return

        ym = (y0 + y1) // 2
        xm = (x0 + x1) // 2

        # Если область уже нельзя нормально разделить
        if ym == y0 or ym == y1 or xm == x0 or xm == x1:
            labels[y0:y1, x0:x1] = current_label
            current_label += 1
            return

        split(y0, ym, x0, xm)
        split(y0, ym, xm, x1)
        split(ym, y1, x0, xm)
        split(ym, y1, xm, x1)

    split(0, h, 0, w)

    return labels


def region_splitting_segmentation(gray: np.ndarray) -> np.ndarray:
    """
    Метод разбиения областей.

    Начинаем со всего изображения как одной области.
    Если область неоднородна, делим ее на четыре подобласти.
    """
    labels = quadtree_split_labels(
        gray,
        min_size=SPLIT_MIN_SIZE,
        std_threshold=SPLIT_STD_THRESHOLD
    )

    result = render_labels_by_mean(gray, labels)
    return result


# =========================
# 6. Слияние/разбиение областей
# =========================

def split_and_merge_segmentation(gray: np.ndarray) -> np.ndarray:
    """
    Комбинированный метод слияния/разбиения.

    1. Сначала выполняется разбиение через дерево квадрантов.
    2. Затем соседние похожие области объединяются.
    """
    split_labels = quadtree_split_labels(
        gray,
        min_size=SPLIT_MIN_SIZE,
        std_threshold=SPLIT_STD_THRESHOLD
    )

    result, _ = merge_regions_from_labels(
        gray,
        split_labels,
        mean_diff_threshold=SPLIT_MERGE_DIFF
    )

    return result


# =========================
# Обработка изображений
# =========================

def process_image(path: Path):
    image = read_image(path)

    if image is None:
        print(f"Не удалось прочитать файл: {path}")
        return

    gray = to_gray(image)

    algorithms = [
        ("01_threshold", "Threshold segmentation", threshold_segmentation),
        ("02_range_threshold", "Range threshold segmentation", range_threshold_segmentation),
        ("03_multi_threshold", "Multi-threshold segmentation", multi_threshold_segmentation),
        ("04_region_merging", "Region merging", region_merging_segmentation),
        ("05_region_splitting", "Region splitting", region_splitting_segmentation),
        ("06_split_and_merge", "Split and merge", split_and_merge_segmentation),
    ]

    image_out_dir = OUT_DIR / path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Сохраняем исходное полутоновое изображение
    save_image(image_out_dir / "00_original_gray.png", gray)

    for file_prefix, title, func in algorithms:
        print(f"Обработка {path.name}: {title}")

        segmented = func(gray)

        # Отдельный результат сегментации
        save_image(image_out_dir / f"{file_prefix}_result.png", segmented)

        # Сравнение исходного и результата
        comparison = make_comparison(gray, segmented, title)
        save_image(image_out_dir / f"{file_prefix}_comparison.png", comparison)

        all_results.append(add_caption(segmented, title))

    # Общая таблица всех результатов
    original_with_caption = add_caption(gray, "Original")
    all_images = [original_with_caption] + all_results

    # Приводим все изображения к одному размеру
    base_h, base_w = all_images[0].shape[:2]
    resized = [
        cv2.resize(img, (base_w, base_h))
        for img in all_images
    ]

    # Формируем сетку 2 x 4
    while len(resized) < 8:
        resized.append(np.full_like(resized[0], 255))

    row1 = cv2.hconcat(resized[:4])
    row2 = cv2.hconcat(resized[4:8])
    summary = cv2.vconcat([row1, row2])

    save_image(image_out_dir / "summary_all_methods.png", summary)


def main():
    if not IN_DIR.exists():
        print("Папка 'in' не найдена. Создай папку 'in' и помести туда изображения.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path for path in IN_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        print("В папке 'in' не найдено изображений.")
        return

    for path in image_paths:
        process_image(path)

    print("Готово. Результаты сохранены в папке 'out'.")


if __name__ == "__main__":
    main()
