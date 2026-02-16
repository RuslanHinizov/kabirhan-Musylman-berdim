# Race Vision Backend API

## Обзор

Бэкенд-сервер на базе FastAPI. Управление 25 аналитическими камерами + 3 PTZ камерами, обнаружение YOLOv8 + ColorCNN, потоковая передача WebRTC.

Архитектура:
```
25 Analytics RTSP --> Потоки CameraReader --> MultiCameraManager
3 PTZ RTSP       --> Потоки CameraReader --> MultiCameraManager
                                                    |
    SmartDetectionScheduler --> MultiDetectionLoop --> состояние на камеру
                                                    |
    RankingMerger --> объединенные рейтинги --> WebSocket broadcast
                                                    |
    Потоки WebRTC/MJPEG --> Фронтенд (оператор + публичный дисплей)
```

Сервер по умолчанию работает по адресу `0.0.0.0:8000`. CORS открыт для всех клиентов.

---

## Файлы

### server.py (~940 строк)

Главный сервер FastAPI. Содержит:

- **SharedState**: Потокобезопасное общее состояние. Поля внутри:
  - `annotated_frames` — Неразмеченные (annotated) кадры на камеру (`dict[str, np.ndarray]`)
  - `per_camera_detections` — Результаты обнаружения на камеру (`dict[str, list]`)
  - `combined_rankings` — Объединенный рейтинг со всех камер (`list`)
  - `camera_horse_presence` — На какой камере какие лошади видны (`dict[str, set]`)
  - `race_active` — Статус гонки (`bool`)
  - `detection_fps` — Значение FPS обнаружения (`float`)
  - `detection_count` — Общее количество обнаруженных кадров (`int`)
  - Потокобезопасный доступ обеспечивается через `threading.Lock()`
  - Методы `set_camera_detection()`, `get_annotated_frame()`, `set_combined_rankings()`, `get_combined_rankings()`

- **MultiDetectionLoop(threading.Thread)**: Интеллектуальный цикл обнаружения для нескольких камер
  - В каждом цикле получает от SmartDetectionScheduler камеры, которые нужно обработать
  - Для каждой камеры: получить кадр --> YOLO --> классификация цвета --> 4-уровневый фильтр
  - Обновить CameraDetectionState для каждой камеры
  - Создать объединенный рейтинг со всех камер с помощью RankingMerger
  - Обновить SharedState (для WebSocket broadcast)
  - 4-уровневая система фильтрации:
    - **F1**: Порог уверенности (min 0.75)
    - **F2**: Совместимость CNN + HSV (если CNN conf < 0.92, проверка HSV)
    - **F3**: Ограничение скорости (max 120 м/с — широко для пиксельного шума)
    - **F4**: Временное подтверждение (минимум 2 обнаружения в окне из 5 кадров)
  - В видеорежиме работает с обратной совместимостью с одним источником
  - `DETECTION_INTERVAL = 0.10` (~10 fps)

- **REST API эндпоинты** — CRUD камер, запуск/остановка, проверка здоровья
- **WebSocket broadcast** — ranking_update, camera_detection, race_start/stop
- **MJPEG streaming** — эндпоинты `/stream/cam{N}` и `/stream/{cam_id}`
- **Аргументы CLI**: `--video`, `--gpu`, `--auto-start`

### camera_manager.py (~480 строк)

- **CameraReader(threading.Thread)**: 1 поток на камеру
  - Чтение RTSP, FFmpegReader (subprocess ffmpeg --> raw BGR24 pipe)
  - Поддержка декодирования GPU (NVDEC — h264_cuvid / hevc_cuvid)
  - Потокобезопасный буфер кадров (последний кадр всегда готов)
  - Авто-переподключение: экспоненциальная задержка (1с --> 30с, сбрасывается при успешном подключении)
  - Состояния: `IDLE` --> `CONNECTING` --> `RUNNING` --> `ERROR` --> `STOPPED`
  - Отслеживание FPS (счетчик кадров в секунду)
  - `get_frame()` — Возвращает последний кадр (потокобезопасная копия)
  - `get_frame_dimensions()` — Размеры кадра (width, height)
  - `get_state()` — Текущее состояние подключения
  - `stop()` — Останавливает Reader, очищает процесс FFmpeg

- **VideoFileReader(threading.Thread)**: Для тестовых видеофайлов (режим `--video`)
  - Читает локальные видеофайлы по порядку
  - Назначает отдельный `cam_id` для каждого видеофайла (analytics-1, analytics-2, ...)
  - Воспроизводит файлы в цикле
  - Предоставляет тот же интерфейс, что и CameraReader (`get_frame()`, `get_frame_dimensions()`)

- **MultiCameraManager**: Управляет всеми камерами
  - `start_camera(cam_id, rtsp_url, use_gpu, cam_type)` — Запускает камеру (перезапускает, если уже работает)
  - `stop_camera(cam_id)` — Останавливает определенную камеру
  - `stop_all()` — Останавливает все камеры (включая video reader)
  - `get_frame(cam_id)` — Пробует получить кадр сначала от RTSP reader, затем от video reader
  - `get_frame_dimensions(cam_id)` — Размеры кадра
  - `get_all_frames()` — Возвращает все доступные кадры
  - `get_status()` — Отчет о состоянии всех камер
  - `get_active_cameras()` — Список камер в статусе RUNNING
  - `get_analytics_cameras()` — Активные камеры только типа analytics
  - `is_running(cam_id)` — Проверяет, работает ли камера
  - `start_video_mode(sources)` — Запускает режим видеофайлов
  - `set_gpu(use_gpu)` — Настройка декодирования GPU для новых камер

### smart_detection.py (~300 строк)

- **CameraDetectionState**: Состояние отслеживания на камеру
  - `smooth_x` — Сглаженная позиция X по цвету (EMA, alpha=0.12)
  - `speed` — Скорость по цвету (м/с, EMA, alpha=0.15)
  - `last_pos` — Последняя известная позиция по цвету: `(pos_m, timestamp)`
  - `det_frames` — Номера кадров для временного подтверждения
  - `live_votes` / `current_order` — Система голосования за порядок цветов (окно 15, мин 5 голосов)
  - `filter_stats` — Статистика фильтров F1-F4
  - `priority` — `"high"` | `"low"` | `"idle"`
  - `horses_present` — Цвета, обнаруженные в данный момент (set)
  - `expected_horses` — Цвета, ожидаемые при передаче с соседней камеры
  - `color_confidence` — Значение уверенности по цвету (для ranking merger)
  - `track_start_m` / `track_end_m` — Диапазон расстояния, покрываемый этой камерой
  - Константы фильтров:
    - `CONF_THRESHOLD = 0.75` — F1
    - `HSV_SKIP_CONF = 0.92` — F2
    - `MAX_SPEED_MPS = 120.0` — F3
    - `TEMPORAL_WINDOW = 5`, `TEMPORAL_MIN = 2` — F4
    - `CAMERA_TRACK_M = 100.0` — Каждая камера покрывает 100м

- **SmartDetectionScheduler**: Планирование приоритетов HIGH/LOW/IDLE
  - **HIGH**: 10 fps (`HIGH_PRIORITY_INTERVAL = 0.10`) — Камеры, где обнаружена лошадь
  - **LOW**: 2 fps (`LOW_PRIORITY_INTERVAL = 0.50`) — Соседние камеры
  - **IDLE**: 0.5 fps (`IDLE_SCAN_INTERVAL = 2.0`) — Сканирование
  - `MAX_CAMERAS_PER_CYCLE = 5` — Макс 5 камер обрабатываются в каждом цикле (~100мс бюджет GPU)
  - `HANDOFF_THRESHOLD = 0.85` — Лошадь на %85 кадра --> повысить следующую камеру до HIGH
  - `GRACE_PERIOD = 3.0` секунды — Держать камеру в LOW после потери обнаружения
  - `get_processing_queue()` — Создает список камер для обработки (HIGH > LOW > IDLE round-robin)
  - `update_priorities()` — Обновляет приоритеты камер на основе результатов обнаружения
  - Проверка соседних камер: делает камеры на расстоянии +/-2 в статус LOW
  - Механизм передачи (Handoff): Когда лошадь приближается к правому краю, повышает статус следующей камеры до HIGH

### ranking_merger.py (~200 строк)

- **HorseTrackingInfo**: Отслеживание лошади между камерами
  - `absolute_distance` — Абсолютная позиция в метрах (0-2500)
  - `speed` — Скорость м/с
  - `confidence` — Значение уверенности
  - `last_cam_id` / `last_cam_index` — Последняя замеченная камера
  - `last_seen_time` — Временная метка последнего появления
  - `is_tracked` — Отслеживается ли активно
  - `grace_distance` / `grace_speed` — Последние известные значения, используемые в льготный период (grace period)

- **RankingMerger**: Объединенный рейтинг 0-2500м со всех камер
  - `CAMERA_TRACK_M = 100.0` — Каждая камера покрывает 100м
  - `GRACE_PERIOD = 2.0` секунды — Сохранять последнюю позицию потерянной лошади 2 секунды
  - `TRACK_LENGTH = 2500.0` — Общая длина трассы
  - Алгоритм:
    1. Найти лучшее обнаружение камеры для каждого цвета (самое новое + самое надежное)
    2. Рассчитать абсолютное расстояние: `track_start_m + (smooth_x / frame_width) * 100`
    3. Отсортировать по убыванию расстояния (кто впереди = 1-е место)
    4. Льготный период: Предсказать позицию для потерянной лошади с последней скоростью
    5. Рассчитать разрыв до лидера в секундах
  - Формат вывода (ожидаемый фронтендом):
    ```json
    {
      "id": "horse-1",
      "number": 1,
      "name": "Red Runner",
      "color": "#DC2626",
      "jockeyName": "Jockey 1",
      "silkId": 1,
      "position": 1,
      "distanceCovered": 1234.5,
      "currentLap": 1,
      "timeElapsed": 45.2,
      "speed": 15.5,
      "gapToLeader": 0.0,
      "lastCameraId": "analytics-13"
    }
    ```

### webrtc_server.py (~180 строк)

- **CameraVideoTrack(MediaStreamTrack)**: WebRTC видеотрек на камеру
  - Аналитические камеры: annotated frame (с наложением YOLO)
  - PTZ камеры: raw frame (высокое качество, без обработки)
  - 25 FPS (часы 90000 Hz, `timestamp += 90000/25` в каждом кадре)
  - `recv()` — вызывается aiortc, возвращает следующий кадр
  - Если кадр не найден, возвращает черный заполнитель (480x640)
  - Конвертация numpy BGR --> `av.VideoFrame`

- **Эндпоинты**:
  - `POST /api/webrtc/offer` — WebRTC SDP offer --> answer
  - `GET /api/webrtc/status` — Статус доступности WebRTC + количество активных подключений
  - `POST /api/webrtc/close-all` — Закрыть все подключения WebRTC

- **Флаг WEBRTC_AVAILABLE**: если aiortc не установлен, graceful fallback
  - `/api/webrtc/offer` возвращает 503
  - `/api/webrtc/status` возвращает `{"available": false}`

- **Сет peer_connections**: Активные подключения, автоматическая очистка при изменении статуса подключения

---

## API Эндпоинты

### Управление Камерами

```
PUT    /api/cameras/{camera_id}        — Обновить RTSP URL (body: { rtspUrl: "..." })
POST   /api/cameras/{camera_id}/start  — Запустить камеру
POST   /api/cameras/{camera_id}/stop   — Остановить камеру
GET    /api/streams/status             — Статусы всех камер
POST   /api/cameras/start-all          — Запустить все камеры
POST   /api/cameras/stop-all           — Остановить все камеры
```

Детали **PUT /api/cameras/{camera_id}**:
- Body: `{ "rtspUrl": "rtsp://user:pass@ip:554/stream" }`
- Если камера уже работает, она перезапускается с новым URL
- Сохраняется в словаре `CUSTOM_CAMERA_URLS` (в памяти)

Детали **POST /api/cameras/{camera_id}/start**:
- Сначала должен быть настроен RTSP URL (через PUT)
- Если URL нет, возвращает ошибку 400
- Тип определяется по префиксу camera_id: если `ptz-*`, то PTZ, иначе analytics

Детали **GET /api/streams/status**:
- Возвращает статус всех 25 analytics + 3 PTZ камер
- Ненастроенные камеры видны как `"state": "idle"`
- Для каждой камеры поля: `state`, `fps`, `type`

### WebSocket (ws://localhost:8000/ws)

Сообщения, отправляемые автоматически при установке соединения:
- `horses_detected` — Информация обо всех лошадях (id, number, name, color, jockeyName, silkId)
- `race_start` — Информация о гонке, если она активна

Типы сообщений от клиента:
- `ping` --> Сервер возвращает `pong` (heartbeat)
- `get_state` --> Сервер возвращает сообщение `state` (полная синхронизация состояния)
- `start_race` --> Запускает гонку, транслируется `race_start` всем клиентам
- `stop_race` --> Останавливает гонку, транслируется `race_stop` всем клиентам

Периодическая трансляция от сервера:
- `ranking_update` — Обновление рейтинга (интервал 200мс, `BROADCAST_INTERVAL = 0.20`)
- `camera_detection` — На какой камере какие лошади (интервал 1 секунда)

Формат сообщения `ranking_update`:
```json
{
  "type": "ranking_update",
  "rankings": [
    {
      "id": "horse-1",
      "number": 1,
      "name": "Red Runner",
      "position": 1,
      "distanceCovered": 1234.5,
      "speed": 15.5,
      "gapToLeader": 0.0,
      "lastCameraId": "analytics-13"
    }
  ]
}
```

Формат сообщения `camera_detection`:
```json
{
  "type": "camera_detection",
  "cameras": {
    "analytics-5": ["red", "blue"],
    "analytics-6": ["green"]
  }
}
```

### Потоковая Передача Видео (Video Streaming)

```
GET  /stream/cam{N}           — Поток MJPEG (N=1-25, старый формат)
GET  /stream/{cam_id}         — Поток MJPEG (analytics-1, ptz-1 и т.д.)
POST /api/webrtc/offer        — WebRTC SDP offer --> answer
GET  /api/webrtc/status       — Статус доступности WebRTC
POST /api/webrtc/close-all    — Закрыть все подключения WebRTC
```

**Настройки MJPEG:**
- `MJPEG_QUALITY = 75` (Качество сжатия JPEG)
- `MJPEG_FPS = 25`
- Аналитические камеры: в первую очередь annotated frame (наложение YOLO), иначе raw frame
- PTZ камеры: raw frame
- Если кадра нет, черный заполнитель + текст "cam_id - waiting..."

**Формат тела WebRTC offer:**
```json
{
  "camId": "analytics-1",
  "sdp": "v=0\r\no=...",
  "type": "offer"
}
```

**Ответ WebRTC answer:**
```json
{
  "sdp": "v=0\r\no=...",
  "type": "answer"
}
```

### Система

```
GET  /api/system/health       — Проверка здоровья
```

Формат ответа:
```json
{
  "cameras": {
    "total_configured": 5,
    "running": 3,
    "status": { "analytics-1": { "state": "running", "fps": 25.0, "type": "analytics" } }
  },
  "detection": {
    "fps": 9.8,
    "total_frames": 1234,
    "race_active": true
  },
  "gpu": {
    "cuda_available": true,
    "cuda_device": "NVIDIA GeForce RTX 3090",
    "memory_allocated_gb": 1.23,
    "memory_reserved_gb": 2.00
  }
}
```

---

## CLI Параметры

```bash
python api/server.py [OPTIONS]

--video FILE [FILE ...]    Режим тестирования из видеофайлов
--gpu                      Включить декодирование GPU (NVDEC)
--auto-start               Автоматически запустить гонку
--url URL                  URL RTSP по умолчанию (по умолчанию: rtsp://admin:...@192.168.18.59:554//stream)
--host HOST                Адрес сервера (по умолчанию: 0.0.0.0)
--port PORT                Порт сервера (по умолчанию: 8000)
```

---

## Примеры

```bash
# Режим видео-теста (2 камеры, автоматический запуск гонки)
python api/server.py --video video/exp10_cam1.mp4 video/exp10_cam2.mp4 --auto-start

# Режим реальной RTSP камеры (с декодированием GPU)
python api/server.py --gpu

# Простой запуск (режим RTSP, камеры настраиваются через API)
python api/server.py

# Другой порт и хост
python api/server.py --host 0.0.0.0 --port 9000 --gpu
```

**Поток режима видео-теста:**
1. Видеофайлы воспроизводятся по порядку
2. Каждому файлу назначается `analytics-N` (N = порядок файла)
3. Когда все файлы заканчиваются, цикл начинается заново
4. С `--auto-start` гонка запускается автоматически

**Поток режима RTSP:**
1. Сервер запускается, камеры не настроены
2. Оператор настраивает URL-адреса RTSP через фронтенд (PUT /api/cameras/{id})
3. Камеры запускаются по одной или все вместе (POST .../start или start-all)
4. SmartDetectionScheduler обрабатывает камеры с приоритизацией

---

## Зависимости (requirements.txt)

| Пакет | Описание |
|-------|----------|
| fastapi | Веб-фреймворк (async REST + WebSocket) |
| uvicorn[standard] | ASGI сервер (HTTP/WS) |
| ultralytics | Обнаружение объектов YOLOv8 |
| torch | PyTorch (SimpleColorCNN + YOLO backend) |
| torchvision | Операции с изображениями PyTorch |
| opencv-python | Чтение/запись изображений, операции с видео |
| numpy | Числовые вычисления, буфер кадров |
| aiortc | Реализация WebRTC на Python (опционально) |
| aiohttp | Async HTTP клиент |

Установка:
```bash
pip install -r requirements.txt
```

---

## Соответствие Цвет --> Лошадь

| Цвет | ID | Номер | Имя | Код Цвета Шелка | Жокей |
|------|----|--------|----|----------------|-------|
| red | horse-1 | 1 | Red Runner | #DC2626 | Jockey 1 |
| blue | horse-2 | 2 | Blue Storm | #2563EB | Jockey 2 |
| green | horse-3 | 3 | Green Flash | #16A34A | Jockey 3 |
| yellow | horse-4 | 4 | Yellow Thunder | #FBBF24 | Jockey 4 |
| purple | horse-5 | 5 | Purple Reign | #9333EA | Jockey 5 |

Это соответствие определено в словаре `COLOR_TO_HORSE` внутри `server.py`. Соответствует `SILK_COLORS` на фронтенде.

---

## Константы Конфигурации

| Константа | Значение | Описание |
|-------|-------|----------|
| `SERVER_HOST` | `0.0.0.0` | Адрес прослушивания сервера |
| `SERVER_PORT` | `8000` | Порт сервера |
| `DETECTION_INTERVAL` | `0.10` | Интервал цикла обнаружения (~10 fps) |
| `BROADCAST_INTERVAL` | `0.20` | Интервал трансляции WebSocket (5 Hz) |
| `MJPEG_QUALITY` | `75` | Качество сжатия JPEG |
| `MJPEG_FPS` | `25` | FPS потока MJPEG |
| `TRACK_LENGTH` | `2500` | Общая длина трассы (метры) |
| `NUM_ANALYTICS_CAMERAS` | `25` | Количество аналитических камер |
| `CAMERA_TRACK_M` | `100.0` | Расстояние, покрываемое каждой камерой (метры) |

---

## ID Камер

- **Аналитические камеры**: от `analytics-1` до `analytics-25` (25 штук)
  - Выполняется обнаружение YOLO + ColorCNN
  - Создается annotated frame (с наложением)
  - Приоритизируются с помощью SmartDetectionScheduler

- **PTZ камеры**: от `ptz-1` до `ptz-3` (3 штуки)
  - Только для трансляции, обнаружение не выполняется
  - Raw frame передается в высоком качестве
  - Управление камерой (pan/tilt/zoom) осуществляется через фронтенд

---

## Пайплайн Обнаружения

```
Кадр (BGR numpy array)
    |
    v
YOLOv8s (обнаружение объектов) --> bounding box'ы
    |
    v
SimpleColorCNN (классификация цвета) --> предсказание цвета для каждого бокса
    |
    v
4-Уровневый Фильтр:
    F1: Confidence >= 0.75
    F2: Совместимость CNN + HSV (если CNN conf < 0.92, проверка HSV)
    F3: Ограничение скорости (<= 120 м/с)
    F4: Временное подтверждение (мин 2 обнаружения в 5 кадрах)
    |
    v
Отфильтрованные обнаружения --> Система голосования (окно 15, мин 5 голосов)
    |
    v
Обновление CameraDetectionState (smooth_x, speed, horses_present)
    |
    v
RankingMerger --> Объединенный рейтинг 0-2500м
    |
    v
SharedState --> WebSocket broadcast (интервал 200мс)
```

---

## Обработка Ошибок

- **Ошибка подключения камеры**: CameraReader переподключается автоматически (экспоненциальная задержка 1с --> 30с)
- **Ошибка чтения кадра**: Инициируется попытка переподключения
- **Разрыв WebSocket**: Клиент удаляется из сета `ws_clients`, другие клиенты не затрагиваются
- **Ошибка подключения WebRTC**: Peer connection закрывается и удаляется из сета `peer_connections`
- **aiortc не установлен**: Graceful fallback — используется MJPEG, эндпоинты WebRTC возвращают 503 / `available: false`
- **Льготный период (Grace period)**: Если лощадь не отслеживается 2 секунды, позиция предсказывается по последней скорости, затем помечается как потерянная
