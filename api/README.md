# Race Vision Backend API

## Genel Bakis

FastAPI tabanli backend sunucu. 25 analitik kamera + 3 PTZ kamera yonetimi, YOLOv8 + ColorCNN algilama, WebRTC streaming.

Mimari:
```
25 Analytics RTSP --> CameraReader threads --> MultiCameraManager
3 PTZ RTSP       --> CameraReader threads --> MultiCameraManager
                                                    |
    SmartDetectionScheduler --> MultiDetectionLoop --> per-camera state
                                                    |
    RankingMerger --> combined rankings --> WebSocket broadcast
                                                    |
    WebRTC/MJPEG streams --> Frontend (operator + public display)
```

Sunucu varsayilan olarak `0.0.0.0:8000` adresinde calisir. Tum istemcilere CORS aciktir.

---

## Dosyalar

### server.py (~940 satir)

Ana FastAPI sunucu. Icerir:

- **SharedState**: Thread-safe paylasimli durum. Icerideki alanlar:
  - `annotated_frames` — Kamera basina isaretsiz (annotated) frame'ler (`dict[str, np.ndarray]`)
  - `per_camera_detections` — Kamera basina algilama sonuclari (`dict[str, list]`)
  - `combined_rankings` — Tum kameralardan birlesik siralama (`list`)
  - `camera_horse_presence` — Hangi kamerada hangi atlar gorunuyor (`dict[str, set]`)
  - `race_active` — Yaris durumu (`bool`)
  - `detection_fps` — Algilama FPS degeri (`float`)
  - `detection_count` — Toplam algilanan frame sayisi (`int`)
  - Thread-safe erisim `threading.Lock()` ile saglanir
  - `set_camera_detection()`, `get_annotated_frame()`, `set_combined_rankings()`, `get_combined_rankings()` metodlari

- **MultiDetectionLoop(threading.Thread)**: Akilli cok kamerali algilama dongusu
  - Her dongude SmartDetectionScheduler'dan islenmesi gereken kameralari alir
  - Her kamera icin: frame al --> YOLO --> renk siniflandir --> 4 katmanli filtre
  - Per-camera CameraDetectionState guncelle
  - RankingMerger ile tum kameralardan birlesik siralama olustur
  - SharedState'i guncelle (WebSocket broadcast icin)
  - 4 katmanli filtre sistemi:
    - **F1**: Confidence threshold (min 0.75)
    - **F2**: CNN + HSV uyumu (CNN conf < 0.92 ise HSV kontrolu)
    - **F3**: Hiz kisitlamasi (max 120 m/s — piksel gurultusu icin genis)
    - **F4**: Temporal onaylama (5 frame'lik pencerede min 2 tespit)
  - Video modunda tek kaynak ile geri uyumlu calisir
  - `DETECTION_INTERVAL = 0.10` (~10 fps)

- **REST API endpointleri** — Kamera CRUD, baslat/durdur, saglik kontrolu
- **WebSocket broadcast** — ranking_update, camera_detection, race_start/stop
- **MJPEG streaming** — `/stream/cam{N}` ve `/stream/{cam_id}` endpointleri
- **CLI argumanlari**: `--video`, `--gpu`, `--auto-start`

### camera_manager.py (~480 satir)

- **CameraReader(threading.Thread)**: Kamera basina 1 thread
  - RTSP okuma, FFmpegReader (subprocess ffmpeg --> raw BGR24 pipe)
  - GPU decode destegi (NVDEC — h264_cuvid / hevc_cuvid)
  - Thread-safe frame buffer (en son frame her zaman hazir)
  - Auto-reconnect: exponential backoff (1s --> 30s, basarili baglatinda sifirlanir)
  - Durumlar: `IDLE` --> `CONNECTING` --> `RUNNING` --> `ERROR` --> `STOPPED`
  - FPS izleme (saniye basina frame sayaci)
  - `get_frame()` — En son frame'i dondurur (thread-safe kopya)
  - `get_frame_dimensions()` — Frame boyutlari (width, height)
  - `get_state()` — Mevcut baglanti durumu
  - `stop()` — Reader'i durdurur, FFmpeg process'i temizler

- **VideoFileReader(threading.Thread)**: Test video dosyalari icin (`--video` modu)
  - Yerel video dosyalarini sirayla okur
  - Her video dosyasi icin ayri `cam_id` atar (analytics-1, analytics-2, ...)
  - Dosyalari dongu halinde tekrar oynatir
  - CameraReader ile ayni arayuzu saglar (`get_frame()`, `get_frame_dimensions()`)

- **MultiCameraManager**: Tum kameralari yonetir
  - `start_camera(cam_id, rtsp_url, use_gpu, cam_type)` — Kamera baslatir (zaten calisan varsa yeniden baslatir)
  - `stop_camera(cam_id)` — Belirli bir kamerayi durdurur
  - `stop_all()` — Tum kameralari durdurur (video reader dahil)
  - `get_frame(cam_id)` — Once RTSP reader, sonra video reader'dan frame dener
  - `get_frame_dimensions(cam_id)` — Frame boyutlari
  - `get_all_frames()` — Tum mevcut frame'leri dondurur
  - `get_status()` — Tum kameralarin durum raporu
  - `get_active_cameras()` — RUNNING durumundaki kamera listesi
  - `get_analytics_cameras()` — Yalnizca analytics turundeki aktif kameralar
  - `is_running(cam_id)` — Kameranin calisip calismadigini kontrol eder
  - `start_video_mode(sources)` — Video dosya modunu baslatir
  - `set_gpu(use_gpu)` — Yeni kameralar icin GPU decode ayari

### smart_detection.py (~300 satir)

- **CameraDetectionState**: Kamera basina tracking state
  - `smooth_x` — Renk basina yumusatilmis X pozisyonu (EMA, alpha=0.12)
  - `speed` — Renk basina hiz (m/s, EMA, alpha=0.15)
  - `last_pos` — Renk basina son bilinen pozisyon: `(pos_m, timestamp)`
  - `det_frames` — Temporal onaylama icin frame numaralari
  - `live_votes` / `current_order` — Renk sirasi oylama sistemi (15 pencere, min 5 oy)
  - `filter_stats` — F1-F4 filtre istatistikleri
  - `priority` — `"high"` | `"low"` | `"idle"`
  - `horses_present` — Su anda tespit edilen renkler (set)
  - `expected_horses` — Komsu kameradan handoff ile beklenen renkler
  - `color_confidence` — Renk basina guven degeri (ranking merger icin)
  - `track_start_m` / `track_end_m` — Bu kameranin kapladigi mesafe araligi
  - Filtre sabitleri:
    - `CONF_THRESHOLD = 0.75` — F1
    - `HSV_SKIP_CONF = 0.92` — F2
    - `MAX_SPEED_MPS = 120.0` — F3
    - `TEMPORAL_WINDOW = 5`, `TEMPORAL_MIN = 2` — F4
    - `CAMERA_TRACK_M = 100.0` — Her kamera 100m kapsar

- **SmartDetectionScheduler**: HIGH/LOW/IDLE oncelik planlamasi
  - **HIGH**: 10 fps (`HIGH_PRIORITY_INTERVAL = 0.10`) — At tespit edilen kameralar
  - **LOW**: 2 fps (`LOW_PRIORITY_INTERVAL = 0.50`) — Komsu kameralar
  - **IDLE**: 0.5 fps (`IDLE_SCAN_INTERVAL = 2.0`) — Tarama
  - `MAX_CAMERAS_PER_CYCLE = 5` — Her dongude max 5 kamera islenir (~100ms GPU butcesi)
  - `HANDOFF_THRESHOLD = 0.85` — At frame'in %85'inde --> sonraki kamerayi HIGH yap
  - `GRACE_PERIOD = 3.0` saniye — Algilama kaybolduktan sonra kamerayi LOW tut
  - `get_processing_queue()` — Islenecek kamera listesini olusturur (HIGH > LOW > IDLE round-robin)
  - `update_priorities()` — Algilama sonuclarina gore kamera onceliklerini gunceller
  - Komsu kamera kontrolu: +/-2 mesafedeki kameralari LOW yapar
  - Handoff mekanizmasi: At sag kenara yaklasinca sonraki kamerayi HIGH'a yukseltir

### ranking_merger.py (~200 satir)

- **HorseTrackingInfo**: At basina cross-camera tracking
  - `absolute_distance` — Metre cinsinden mutlak pozisyon (0-2500)
  - `speed` — m/s hiz
  - `confidence` — Guven degeri
  - `last_cam_id` / `last_cam_index` — Son gorulen kamera
  - `last_seen_time` — Son gorulen zaman damgasi
  - `is_tracked` — Aktif olarak izleniyor mu
  - `grace_distance` / `grace_speed` — Grace doneminde kullanilan son bilinen degerler

- **RankingMerger**: Tum kameralardan birlesik 0-2500m siralama
  - `CAMERA_TRACK_M = 100.0` — Her kamera 100m kapsar
  - `GRACE_PERIOD = 2.0` saniye — Kayip atin son pozisyonunu 2 saniye koru
  - `TRACK_LENGTH = 2500.0` — Toplam pist uzunlugu
  - Algoritma:
    1. Her renk icin en iyi kamera tespitini bul (en yeni + en guvenilir)
    2. Mutlak mesafeyi hesapla: `track_start_m + (smooth_x / frame_width) * 100`
    3. Mesafeye gore azalan sirada sirala (en ondeki = 1. sirada)
    4. Grace donemi: Kayip at icin son hiz ile pozisyon tahmin et
    5. Lidere olan farki saniye cinsinden hesapla
  - Cikti formati (frontend'in beklediqi):
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

### webrtc_server.py (~180 satir)

- **CameraVideoTrack(MediaStreamTrack)**: Kamera basina WebRTC video track
  - Analytics kameralar: annotated frame (YOLO overlay ile)
  - PTZ kameralar: raw frame (yuksek kalite, islem yok)
  - 25 FPS (90000 Hz clock, `timestamp += 90000/25` her frame'de)
  - `recv()` — aiortc tarafindan cagrilir, sonraki frame'i dondurur
  - Frame bulunamazsa siyah placeholder (480x640) dondurur
  - numpy BGR --> `av.VideoFrame` donusumu

- **Endpointler**:
  - `POST /api/webrtc/offer` — WebRTC SDP offer --> answer
  - `GET /api/webrtc/status` — WebRTC kullanilabilirlik durumu + aktif baglanti sayisi
  - `POST /api/webrtc/close-all` — Tum WebRTC baglantilarini kapat

- **WEBRTC_AVAILABLE flag**: aiortc yuklu degilse graceful fallback
  - `/api/webrtc/offer` 503 dondurur
  - `/api/webrtc/status` `{"available": false}` dondurur

- **peer_connections** set'i: Aktif baglantilar, baglanti durumu degistiginde otomatik temizleme

---

## API Endpointleri

### Kamera Yonetimi

```
PUT    /api/cameras/{camera_id}        — RTSP URL guncelle (body: { rtspUrl: "..." })
POST   /api/cameras/{camera_id}/start  — Kamerayi baslat
POST   /api/cameras/{camera_id}/stop   — Kamerayi durdur
GET    /api/streams/status             — Tum kamera durumlari
POST   /api/cameras/start-all          — Tum kameralari baslat
POST   /api/cameras/stop-all           — Tum kameralari durdur
```

**PUT /api/cameras/{camera_id}** detaylari:
- Body: `{ "rtspUrl": "rtsp://user:pass@ip:554/stream" }`
- Kamera zaten calisiyorsa yeni URL ile yeniden baslatilir
- `CUSTOM_CAMERA_URLS` dict'ine kaydedilir (bellekte kalici)

**POST /api/cameras/{camera_id}/start** detaylari:
- Once RTSP URL yapilandirilmis olmali (PUT ile)
- URL yoksa 400 hatasi dondurur
- camera_id prefix'ine gore tip belirlenir: `ptz-*` ise PTZ, aksi halde analytics

**GET /api/streams/status** detaylari:
- Tum 25 analytics + 3 PTZ kameranin durumunu dondurur
- Yapilandirilmamis kameralar `"state": "idle"` olarak gorulur
- Her kamera icin: `state`, `fps`, `type` alanlari

### WebSocket (ws://localhost:8000/ws)

Baglanti kuruldugunda otomatik gonderilen mesajlar:
- `horses_detected` — Tum atlarin bilgileri (id, number, name, color, jockeyName, silkId)
- `race_start` — Eger yaris aktifse, yaris bilgileri

Istemciden gelen mesaj tipleri:
- `ping` --> Sunucu `pong` dondurur (heartbeat)
- `get_state` --> Sunucu `state` mesaji dondurur (tam durum senkronizasyonu)
- `start_race` --> Yarisi baslatir, tum istemcilere `race_start` broadcast edilir
- `stop_race` --> Yarisi durdurur, tum istemcilere `race_stop` broadcast edilir

Sunucudan periyodik broadcast:
- `ranking_update` — Siralama guncellemesi (200ms aralik, `BROADCAST_INTERVAL = 0.20`)
- `camera_detection` — Hangi kamerada hangi atlar (1 saniye aralik)

`ranking_update` mesaj formati:
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

`camera_detection` mesaj formati:
```json
{
  "type": "camera_detection",
  "cameras": {
    "analytics-5": ["red", "blue"],
    "analytics-6": ["green"]
  }
}
```

### Video Streaming

```
GET  /stream/cam{N}           — MJPEG stream (N=1-25, eski format)
GET  /stream/{cam_id}         — MJPEG stream (analytics-1, ptz-1 vb.)
POST /api/webrtc/offer        — WebRTC SDP offer --> answer
GET  /api/webrtc/status       — WebRTC kullanilabilirlik durumu
POST /api/webrtc/close-all    — Tum WebRTC baglantilarini kapat
```

**MJPEG ayarlari:**
- `MJPEG_QUALITY = 75` (JPEG sikistrima kalitesi)
- `MJPEG_FPS = 25`
- Analytics kameralar: oncelikle annotated frame (YOLO overlay), yoksa raw frame
- PTZ kameralar: raw frame
- Frame yoksa siyah placeholder + "cam_id - waiting..." metni

**WebRTC offer body formati:**
```json
{
  "camId": "analytics-1",
  "sdp": "v=0\r\no=...",
  "type": "offer"
}
```

**WebRTC answer yaniti:**
```json
{
  "sdp": "v=0\r\no=...",
  "type": "answer"
}
```

### Sistem

```
GET  /api/system/health       — Saglik kontrolu
```

Yanit formati:
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

## CLI Parametreleri

```bash
python api/server.py [OPTIONS]

--video FILE [FILE ...]    Video dosyalarindan test modu
--gpu                      GPU decode etkinlestir (NVDEC)
--auto-start               Yarisi otomatik baslat
--url URL                  Varsayilan RTSP URL (varsayilan: rtsp://admin:...@192.168.18.59:554//stream)
--host HOST                Sunucu adresi (varsayilan: 0.0.0.0)
--port PORT                Sunucu portu (varsayilan: 8000)
```

---

## Ornekler

```bash
# Video test modu (2 kamera, otomatik yaris baslatma)
python api/server.py --video video/exp10_cam1.mp4 video/exp10_cam2.mp4 --auto-start

# RTSP gercek kamera modu (GPU decode ile)
python api/server.py --gpu

# Basit baslatma (RTSP modu, kameralar API ile yapilandirilir)
python api/server.py

# Farkli port ve host
python api/server.py --host 0.0.0.0 --port 9000 --gpu
```

**Video test modu akisi:**
1. Video dosyalari sirayla oynatilir
2. Her dosya `analytics-N` olarak atanir (N = dosya sirasi)
3. Tum dosyalar bitince dongu halinde tekrar baslar
4. `--auto-start` ile yaris otomatik baslatilir

**RTSP modu akisi:**
1. Sunucu baslar, kamera yapilandirilmamis durumda
2. Operator frontend'ten RTSP URL'leri yapilandirir (PUT /api/cameras/{id})
3. Kameralar tek tek veya toplu baslatilir (POST .../start veya start-all)
4. SmartDetectionScheduler kameralari onceliklendirerek isler

---

## Bagimliliklar (requirements.txt)

| Paket | Aciklama |
|-------|----------|
| fastapi | Web framework (async REST + WebSocket) |
| uvicorn[standard] | ASGI sunucu (HTTP/WS) |
| ultralytics | YOLOv8 nesne algilama |
| torch | PyTorch (SimpleColorCNN + YOLO backend) |
| torchvision | PyTorch goruntu islemleri |
| opencv-python | Goruntu okuma/yazma, video islemleri |
| numpy | Sayisal hesaplamalar, frame buffer |
| aiortc | WebRTC Python uygulamasi (opsiyonel) |
| aiohttp | Async HTTP istemci |

Kurulum:
```bash
pip install -r requirements.txt
```

---

## Renk --> At Eslestirme

| Renk | ID | Numara | Ad | Silk Renk Kodu | Jokey |
|------|----|--------|----|----------------|-------|
| red | horse-1 | 1 | Red Runner | #DC2626 | Jockey 1 |
| blue | horse-2 | 2 | Blue Storm | #2563EB | Jockey 2 |
| green | horse-3 | 3 | Green Flash | #16A34A | Jockey 3 |
| yellow | horse-4 | 4 | Yellow Thunder | #FBBF24 | Jockey 4 |
| purple | horse-5 | 5 | Purple Reign | #9333EA | Jockey 5 |

Bu eslestirme `server.py` icindeki `COLOR_TO_HORSE` sozlugunde tanimlidir. Frontend'teki `SILK_COLORS` ile eslesir.

---

## Yapilandirma Sabitleri

| Sabit | Deger | Aciklama |
|-------|-------|----------|
| `SERVER_HOST` | `0.0.0.0` | Sunucu dinleme adresi |
| `SERVER_PORT` | `8000` | Sunucu portu |
| `DETECTION_INTERVAL` | `0.10` | Algilama dongusu araligi (~10 fps) |
| `BROADCAST_INTERVAL` | `0.20` | WebSocket broadcast araligi (5 Hz) |
| `MJPEG_QUALITY` | `75` | JPEG sikistrima kalitesi |
| `MJPEG_FPS` | `25` | MJPEG stream FPS |
| `TRACK_LENGTH` | `2500` | Toplam pist uzunlugu (metre) |
| `NUM_ANALYTICS_CAMERAS` | `25` | Analitik kamera sayisi |
| `CAMERA_TRACK_M` | `100.0` | Her kameranin kapladigi mesafe (metre) |

---

## Kamera Kimlikleri

- **Analytics kameralar**: `analytics-1` ile `analytics-25` arasi (25 adet)
  - YOLO + ColorCNN algilama yapilir
  - Annotated frame (overlay ile) olusturulur
  - SmartDetectionScheduler tarafindan onceliklendirilir

- **PTZ kameralar**: `ptz-1` ile `ptz-3` arasi (3 adet)
  - Yalnizca broadcast icin, algilama yapilmaz
  - Raw frame yuksek kalitede iletilir
  - Kamera kontrol (pan/tilt/zoom) frontend'ten yonetilir

---

## Algilama Pipeline'i

```
Frame (BGR numpy array)
    |
    v
YOLOv8s (nesne algilama) --> bounding box'lar
    |
    v
SimpleColorCNN (renk siniflandirma) --> her kutu icin renk tahmini
    |
    v
4-Katmanli Filtre:
    F1: Confidence >= 0.75
    F2: CNN + HSV uyumu (CNN conf < 0.92 ise HSV kontrolu)
    F3: Hiz kisitlamasi (<= 120 m/s)
    F4: Temporal onaylama (5 frame'de min 2 tespit)
    |
    v
Filtrelenmis tespitler --> Oylama sistemi (15 pencere, min 5 oy)
    |
    v
CameraDetectionState guncelleme (smooth_x, speed, horses_present)
    |
    v
RankingMerger --> Birlesik 0-2500m siralama
    |
    v
SharedState --> WebSocket broadcast (200ms aralik)
```

---

## Hata Islemleri

- **Kamera baglanti hatasi**: CameraReader otomatik yeniden baglanir (exponential backoff 1s --> 30s)
- **Frame okuma hatasi**: Yeniden baglanti denemesi baslatilir
- **WebSocket kopuklugu**: Istemci `ws_clients` set'inden cikarilir, diger istemciler etkilenmez
- **WebRTC baglanti hatasi**: Peer connection kapatilir ve `peer_connections` set'inden cikarilir
- **aiortc yuklu degil**: Graceful fallback — MJPEG kullanilir, WebRTC endpointleri 503 / `available: false` dondurur
- **Grace donemi**: At 2 saniye izlenemezse son hizi ile pozisyon tahmin edilir, sonra kayip olarak isaretlenir
