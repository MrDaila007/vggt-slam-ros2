# TODO — vggt_slam_ros2 Development Plan

## Этап 1 — Базовая работоспособность (MVP)

- [x] Структура ROS2 пакета (`package.xml`, `setup.py`, `setup.cfg`)
- [x] `VGGTWrapper` — загрузка модели, preprocessing, inference
- [x] `KeyframeSelector` — dual-threshold отбор кадров (optical flow + max interval)
- [x] `SlidingWindow` — скользящее окно с настраиваемым перекрытием
- [x] `MapManager` — инкрементальный накопитель point cloud
- [x] `slam_node.py` — ROS2 Lifecycle Node (полный SLAM)
- [x] `pointcloud_node.py` — упрощённый узел только для облака точек
- [x] `ros_conversions.py` — numpy/torch ↔ PointCloud2, TF2, PoseStamped
- [x] Launch файлы (`vggt_slam.launch.py`, `vggt_pointcloud.launch.py`)
- [x] `config/params.yaml` — все параметры с документацией
- [x] `LICENSE` — Apache-2.0
- [x] `THIRD_PARTY_LICENSES.md` — предупреждения о лицензии VGGT
- [x] `.gitignore` и `requirements.txt`
- [x] `scripts/test_on_tum.py` — тест пайплайна без ROS2, метрики ATE/RPE, Sim(3) выравнивание
- [x] `docs/get_tum_dataset.md` — инструкция по скачиванию TUM RGB-D
- [ ] **Anchor scale** — якорить каждое новое окно к перекрывающимся кадрам предыдущего для устранения дрейфа масштаба (`core/scale_anchor.py`)
- [ ] **Тест на TUM fr1/desk** — запустить `test_on_tum.py`, зафиксировать ATE RMSE как baseline
- [ ] **Тест на остальных 8 последовательностях fr1** — сравнить с результатами VGGT-SLAM

---

## Этап 2 — Loop Closure

- [ ] `core/image_retrieval.py` — DINOv2 embeddings + cosine similarity для поиска петель (без лицензионных ограничений NetVLAD)
- [ ] `core/pose_graph.py` — GTSAM factor graph для глобальной оптимизации траектории
- [ ] Интеграция loop closure в `slam_node.py` — триггер при обнаружении петли
- [ ] Тест loop closure на `freiburg1_room` (длинная петля ~1.5 ГБ)
- [ ] Сравнение ATE до/после loop closure

---

## Этап 3 — Качество и удобство

- [ ] `config/vggt_slam.rviz` — RViz2 конфиг с настроенными отображениями PointCloud2, Path, TF, Depth
- [ ] `Dockerfile` — воспроизводимая среда (CUDA + ROS2 Humble/Jazzy + VGGT)
- [ ] `docker-compose.yaml` — сервисы slam_node + rviz2
- [ ] GitHub Actions CI — flake8 + mypy + `colcon build` check на каждый push
- [ ] `scripts/eval_all_tum.sh` — запустить все 9 последовательностей fr1 и записать в `results/`
- [ ] Видео-демо — запись на офисном/квартирном датасете для README

---

## Этап 4 — Продвинутые фичи

- [ ] **Стерео поддержка** — вторая камера как абсолютная метрическая шкала (устраняет Sim(3) неоднозначность)
- [ ] **Nav2 интеграция** — публикация `OccupancyGrid` из накопленного облака для автономной навигации
- [ ] **Параметрическая оптимизация** — автоматически выбирать `window_size` / `stride` по доступной GPU-памяти
- [ ] **EuRoC датасет** — тест на drone-footage (более сложные условия освещения и движения)
- [ ] **srv/SaveMap.srv** — сохранение карты в PCD/PLY файл по ROS2 сервису
