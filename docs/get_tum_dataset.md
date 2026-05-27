# Как скачать датасет TUM RGB-D

Сайт датасета: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download

---

## Быстрый старт — одна последовательность

Для первого теста рекомендуется `fr1/desk` — небольшой (~600 МБ), снят в помещении со столом.

```bash
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar xzf rgbd_dataset_freiburg1_desk.tgz
```

Проверь структуру:

```
rgbd_dataset_freiburg1_desk/
├── rgb/                  ← цветные кадры (640×480, ~30 fps)
├── depth/                ← карты глубины (16-bit PNG, мм)
├── rgb.txt               ← метки времени и пути к файлам
├── depth.txt
├── groundtruth.txt       ← эталонная траектория (tx ty tz qx qy qz qw)
└── accelerometer.txt
```

---

## Все последовательности fr1 (те же что использует VGGT-SLAM)

| Последовательность | Размер | Описание |
|---|---|---|
| `freiburg1_desk`   | ~600 МБ | Стол, офис — **рекомендуется для первого теста** |
| `freiburg1_desk2`  | ~590 МБ | Стол, другой ракурс |
| `freiburg1_room`   | ~1.5 ГБ | Целая комната, большая петля |
| `freiburg1_floor`  | ~740 МБ | Камера смотрит вниз |
| `freiburg1_plant`  | ~500 МБ | Растение на столе |
| `freiburg1_teddy`  | ~660 МБ | Мягкая игрушка |
| `freiburg1_xyz`    | ~480 МБ | Чистые поступательные движения |
| `freiburg1_rpy`    | ~490 МБ | Чистые вращения |
| `freiburg1_360`    | ~1.1 ГБ | Поворот на 360° |

Скачать все сразу:

```bash
BASE="https://vision.in.tum.de/rgbd/dataset/freiburg1"
for SEQ in desk desk2 room floor plant teddy xyz rpy 360; do
    wget -c "${BASE}/rgbd_dataset_freiburg1_${SEQ}.tgz"
    tar xzf "rgbd_dataset_freiburg1_${SEQ}.tgz"
done
```

---

## Запуск скрипта оценки

```bash
# Одна последовательность
python scripts/test_on_tum.py --dataset rgbd_dataset_freiburg1_desk

# Результаты сохраняются в ./results/
#   rgbd_dataset_freiburg1_desk_metrics.txt
#   rgbd_dataset_freiburg1_desk_estimated_tum.txt
#   rgbd_dataset_freiburg1_desk_trajectory.png

# Дополнительная проверка через evo (если установлен: pip install evo)
evo_ape tum rgbd_dataset_freiburg1_desk/groundtruth.txt \
    results/rgbd_dataset_freiburg1_desk_estimated_tum.txt -a --plot
```

Все параметры скрипта:

```
--dataset        путь к папке последовательности (обязательно)
--window_size    16       кадров на вызов VGGT
--window_stride  8        новых кадров между вызовами
--min_flow       10.0     минимальный оптический поток для ключевого кадра (px)
--conf_thr       20.0     отфильтровать нижние N% точек по уверенности
--max_frames     0        ограничить число кадров (0 = все)
--out_dir        results/ папка для результатов
--no_plot                 не строить график
```

---

## Требования

```bash
pip install torch torchvision opencv-python numpy matplotlib
pip install -e /path/to/vggt          # VGGT из исходников

# опционально, для дополнительной оценки метрик
pip install evo
```
