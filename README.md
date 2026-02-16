# Race Vision — At Yarisi Gercek Zamanli Pozisyon Takip Sistemi

Gercek zamanli at yarisi pozisyon takip ve yayin sistemi. YOLOv8 + CNN ile jokey tespiti ve renk siniflandirma, 28 kamerali profesyonel mimari, WebRTC dusuk gecikmeli video streaming ve 4 dil destegi sunar.

---

## Proje Hakkinda

Race Vision, at yarisi pistlerinde atlarin gercek zamanli konumlarini tespit eden, siralayan ve canli yayin olarak sunan uctan uca bir sistemdir.

- **YOLOv8s** ile jokey/kisi tespiti
- **SimpleColorCNN** (PyTorch) ile jokey renk siniflandirma (5 renk)
- **28 kamerali** profesyonel mimari (25 analitik + 3 PTZ)
- **WebRTC** dusuk gecikmeli video streaming (MJPEG fallback ile)
- **4 dil destegi**: Ingilizce (EN), Turkce (TR), Rusca (RU), Kazakca (KK)
- **GPU hizlandirmali** video decode/encode (NVDEC/NVENC)
- Operator paneli ve herkese acik ekran olmak uzere iki ayri arayuz

---

## Teknolojiler

### Backend

| Teknoloji | Versiyon / Aciklama |
|-----------|---------------------|
| Python | 3.10+ |
| FastAPI + Uvicorn | ASGI web framework |
| YOLOv8s (Ultralytics) | Jokey/kisi tespiti |
| SimpleColorCNN (PyTorch) | Jokey renk siniflandirma (5 renk: kirmizi, mavi, yesil, sari, mor) |
| OpenCV | Video isleme |
| aiortc | WebRTC streaming |
| NVDEC/NVENC | GPU hizlandirmali video decode/encode |

### Frontend

| Teknoloji | Versiyon / Aciklama |
|-----------|---------------------|
| React | 19.2 |
| TypeScript | 5.9 |
| Vite | 7.2 (build tool) |
| Zustand | 5 (state management) |
| Framer Motion | 12 (animasyonlar) |
| Tailwind CSS | 4 (stil) |
| i18next | 4 dil destegi (EN/TR/RU/KK) |
| lucide-react | Ikon kutuphanesi |

---

## Mimari

### Kamera Rolleri

#### 25 Analitik Kamera (analytics-1 — analytics-25)

- Her biri **100m'lik** pist segmentini kapsar (toplam **2500m**)
- YOLO + ColorCNN algilama yapilir
- Izleyiciler bu kameralari **GORMEZ**
- Sadece siralama verisi frontende gonderilir
- Akilli onceliklendirme ile GPU kaynagi verimli kullanilir

#### 3 PTZ Kamera (ptz-1, ptz-2, ptz-3)

- Sadece **yayin kamerasi** — izleyicilerin gordugu ekran
- Algilama **YAPILMAZ** (performans icin)
- Akici yuksek FPS, dusuk gecikme
- Pozisyonlar:
  - `ptz-1`: 0m (baslangic)
  - `ptz-2`: 1250m (orta)
  - `ptz-3`: 2500m (bitis)

### Veri Akisi

```
25 RTSP Kamera --> CameraReader threads --> MultiCameraManager
                                                  |
                                                  v
SmartDetectionScheduler --> MultiDetectionLoop --> per-camera state
                                                  |
                                                  v
RankingMerger --> birlesik siralama --> WebSocket broadcast
                                                  |
                                                  v
WebRTC/MJPEG streams --> Frontend (operator + herkese acik ekran)
```

**Adim adim akis:**

1. 25 analitik kamera RTSP uzerinden `CameraReader` thread'lerine bagli
2. `MultiCameraManager` tum kamera thread'lerini yonetir
3. `SmartDetectionScheduler` hangi kameralarin ne siklikta islenmesi gerektigini belirler
4. `MultiDetectionLoop` her cycle'da secilen kameralarda YOLO + CNN calistirir
5. `RankingMerger` tum kameralardan gelen verileri birlestirip 0-2500m araliginda birlesik siralama olusturur
6. Siralama verisi WebSocket uzerinden tum istemcilere yayinlanir
7. PTZ kameralari WebRTC (veya MJPEG fallback) uzerinden canli goruntu aktarir

---

## Akilli Algilama Onceliklendirme

Sistem, GPU kaynaklarini verimli kullanmak icin uc seviyeli bir onceliklendirme mekanizmasi kullanir:

| Seviye | FPS | Kosul | Aciklama |
|--------|-----|-------|----------|
| **HIGH** | 10 fps | At tespit edilen kameralar | Maksimum islem hizi |
| **LOW** | 2 fps | Komsu kameralarda at var | Hazirlik modu |
| **IDLE** | 0.5 fps | At yok | Tarama modu |

**Kurallar:**
- Cycle basina maksimum **5 kamera** islenir (GPU butcesi)
- At handoff: **%85** frame sinirinda sonraki kameraya gecis tetiklenir
- Onceliklendirme her cycle basinda yeniden hesaplanir

---

## 4 Katmanli Filtreleme

Tespit sonuclarinin guvenilirligi icin dort katmanli bir filtreleme sistemi uygulanir:

1. **Temporal filtre** — Son 5 frame'de en az 3 kez tespit edilmesi gerekir (gurultu azaltma)
2. **EMA smoothing** — Pozisyon yumusatma (alpha=0.3), ani siciramalar onlenir
3. **Hiz filtresi** — Fiziksel olarak mumkun olmayan hizlari reddeder (teleportasyon onleme)
4. **Canli oylama** — Coklu algilamalardan guvenilir sonuc cikarir

Bu katmanlar sirasiyla uygulanir ve sadece tum filtrelerden gecen sonuclar siralama tablosuna yansitilir.

---

## Renk - At Eslestirme Tablosu

Jokey yelekleri renk bazli siniflandirilir ve her renk bir ata karsilik gelir:

| Renk | At ID | Numara | Ad | Silk ID |
|------|-------|--------|----|---------|
| Kirmizi | horse-1 | 1 | Red Runner | 1 |
| Mavi | horse-2 | 2 | Blue Storm | 2 |
| Yesil | horse-3 | 3 | Green Flash | 3 |
| Sari | horse-4 | 4 | Yellow Thunder | 4 |
| Mor | horse-5 | 5 | Purple Reign | 5 |

**Siniflandirma sureci:**
1. YOLO kisi/jokey tespiti yapar ve bounding box cikarir
2. Bounding box icerisindeki goruntu kesilir (crop)
3. SimpleColorCNN bu goruntunun rengini 5 siniftan birine atar
4. Renk-at eslestirme tablosu uzerinden at kimligi belirlenir

---

## Kurulum

### Gereksinimler

- Python 3.10+
- Node.js 18+
- NVIDIA GPU (CUDA destegi, opsiyonel ama onerilen)
- FFmpeg (RTSP decode icin)

### Backend Kurulumu

```bash
cd race_vision
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

> **Not:** GPU hizlandirmasi icin NVIDIA CUDA Toolkit ve uyumlu PyTorch versiyonunun yuklu olmasi gerekir.

### Frontend Kurulumu

```bash
cd Kabirhan-Frontend
npm install
```

---

## Calistirma

### Video Test Modu (gelistirme)

Gercek kamera olmadan, kayitli videolarla test yapmak icin:

```bash
python api/server.py --video video/exp10_cam1.mp4 video/exp10_cam2.mp4 video/exp10_cam3.mp4 --auto-start
```

- `--video`: Test video dosyalarinin yollarini belirtir
- `--auto-start`: Sunucu basladiginda otomatik olarak algilamayi baslatir

### RTSP Gercek Kamera Modu

```bash
python api/server.py --gpu
```

Sunucu basladiktan sonra operator panelinden kamera URL'lerini yapilandirin:

1. `http://localhost:5173/operator` adresine gidin
2. **RTSP Cameras** sekmesine gecin
3. Her kamera icin RTSP URL'sini girin
4. Kameralari tek tek veya toplu olarak baslatin

### Frontend

```bash
cd Kabirhan-Frontend
npm run dev
```

Erisim adresleri:

| Sayfa | URL | Aciklama |
|-------|-----|----------|
| Operator Paneli | `http://localhost:5173/operator` | Kamera yonetimi, yaris kontrolu, 2D pist gorunumu |
| Herkese Acik Ekran | `http://localhost:5173/` | PTZ kamera goruntusu + siralama tablosu |

---

## API Endpointleri

### REST API

| Method | Endpoint | Aciklama |
|--------|----------|----------|
| `PUT` | `/api/cameras/{id}` | Kamera RTSP URL guncelle |
| `POST` | `/api/cameras/{id}/start` | Belirli bir kamerayi baslat |
| `POST` | `/api/cameras/{id}/stop` | Belirli bir kamerayi durdur |
| `GET` | `/api/streams/status` | Tum kamera durumlarini getir |
| `POST` | `/api/cameras/start-all` | Tum kameralari baslat |
| `POST` | `/api/cameras/stop-all` | Tum kameralari durdur |
| `GET` | `/api/system/health` | Sistem saglik kontrolu |

### WebSocket

- **Baglanti:** `ws://localhost:8000/ws`
- **Mesaj tipleri:**

| Mesaj Tipi | Yon | Aciklama |
|------------|-----|----------|
| `ranking_update` | Sunucu -> Istemci | Guncel siralama verisi |
| `camera_detection` | Sunucu -> Istemci | Kamera bazli tespit sonuclari |
| `race_start` | Sunucu -> Istemci | Yaris basladigi bilgisi |
| `race_stop` | Sunucu -> Istemci | Yaris bittigi bilgisi |
| `horses_detected` | Sunucu -> Istemci | Tespit edilen atlar |
| `pong` | Sunucu -> Istemci | Heartbeat cevabi |
| `state` | Sunucu -> Istemci | Tam durum senkronizasyonu |

### Video Streaming

| Endpoint | Aciklama |
|----------|----------|
| `GET /stream/cam{N}` | MJPEG stream (N=1-25 analitik kameralar) |
| `GET /stream/{cam_id}` | MJPEG stream (cam_id formati, ornegin `analytics-1`) |
| `POST /api/webrtc/offer` | WebRTC SDP offer/answer degisimi |
| `GET /api/webrtc/status` | WebRTC baglanti durumu |

**Video streaming stratejisi:**
- Oncelikle WebRTC baglantisi denenir (dusuk gecikme)
- WebRTC basarisiz olursa otomatik olarak MJPEG'e duser (fallback)
- `WebRTCPlayer.tsx` ve `MJPEGPlayer.tsx` bu gecisi seffaf sekilde yonetir

---

## Dosya Yapisi

```
race_vision/
|
|-- api/
|   |-- server.py                # Ana FastAPI sunucu (~940 satir)
|   |-- camera_manager.py        # CameraReader + MultiCameraManager (~480 satir)
|   |-- smart_detection.py       # SmartDetectionScheduler (~300 satir)
|   |-- ranking_merger.py        # RankingMerger (~200 satir)
|   |-- webrtc_server.py         # WebRTC streaming (~180 satir)
|   +-- __init__.py
|
|-- Kabirhan-Frontend/
|   |-- src/
|   |   |-- pages/
|   |   |   |-- OperatorPanel.tsx      # Operator kontrol paneli
|   |   |   +-- PublicDisplay.tsx      # Herkese acik ekran
|   |   |
|   |   |-- components/
|   |   |   |-- WebRTCPlayer.tsx       # WebRTC video player
|   |   |   |-- MJPEGPlayer.tsx        # MJPEG fallback player
|   |   |   |-- LanguageSelector.tsx   # Dil secici
|   |   |   |
|   |   |   |-- operator/
|   |   |   |   |-- CameraGrid.tsx         # 25 kamera grid gorunumu
|   |   |   |   |-- CameraSettings.tsx     # Kamera RTSP ayarlari
|   |   |   |   |-- PTZControlPanel.tsx    # PTZ kamera secimi
|   |   |   |   |-- Track2DView.tsx        # 2D oval pist gorunumu
|   |   |   |   +-- RaceSettings.tsx       # Yaris ayarlari
|   |   |   |
|   |   |   +-- public-display/
|   |   |       |-- PTZCameraDisplay.tsx   # PTZ kamera goruntusu
|   |   |       +-- RankingBoard.tsx       # Siralama tablosu
|   |   |
|   |   |-- store/
|   |   |   |-- cameraStore.ts     # Kamera state (Zustand)
|   |   |   +-- raceStore.ts       # Yaris state (Zustand)
|   |   |
|   |   |-- services/
|   |   |   +-- backendConnection.ts  # WebSocket istemci
|   |   |
|   |   |-- config/
|   |   |   +-- cameras.ts         # Kamera konfigurasyonu
|   |   |
|   |   |-- i18n/
|   |   |   |-- index.ts           # i18next baslatma
|   |   |   +-- locales/
|   |   |       |-- en.ts          # Ingilizce
|   |   |       |-- tr.ts          # Turkce
|   |   |       |-- ru.ts          # Rusca
|   |   |       +-- kk.ts          # Kazakca
|   |   |
|   |   |-- utils/
|   |   |   +-- silkUtils.ts       # Silk/jokey ikon yardimcilari
|   |   |
|   |   +-- types/
|   |       +-- index.ts           # TypeScript tipleri
|   |
|   +-- public/
|       +-- assets/silks/          # 10 jokey silk SVG ikonu
|
|-- models/
|   +-- color_classifier.pt       # Egitilmis renk CNN modeli (2.5MB)
|
|-- tools/
|   |-- test_race_count.py         # RaceTracker + ColorClassifier
|   |-- test_rtsp.py               # FFmpegReader (GPU decode)
|   +-- train_color_classifier.py  # CNN egitim scripti
|
|-- video/                         # Test videolari (3 MP4)
|-- yolov8s.pt                     # YOLOv8s model (22.5MB)
+-- requirements.txt               # Python bagimliliklari
```

---

## Dil Destegi

| Dil | Kod | Dosya |
|-----|-----|-------|
| Ingilizce | EN | `src/i18n/locales/en.ts` |
| Turkce | TR | `src/i18n/locales/tr.ts` |
| Rusca | RU | `src/i18n/locales/ru.ts` |
| Kazakca | KK | `src/i18n/locales/kk.ts` |

**Davranis:**
- Tarayici dili otomatik algilanir ve uygun ceviri yuklenir
- Kullanici tercihi `localStorage`'da kaydedilir
- Sekmeler arasi senkronize edilir
- Varsayilan dil: Ingilizce (EN)

---

## GPU Kaynak Kullanimi

Sistem, NVIDIA GPU'nun farkli donanim motorlarini paralel olarak kullanir:

| Birim | Kullanim | Aciklama |
|-------|----------|----------|
| **NVDEC** | 28 eszamanli video decode | Ayri donanim motoru, CPU'dan bagimsiz |
| **CUDA** | YOLO + CNN inference | Cycle basina 3-5 kamera islenir |
| **NVENC** | WebRTC H.264 encode | Ayri donanim motoru, CUDA'dan bagimsiz |

**Kaynak tuketimi:**
- Toplam yaklasik **600-900MB VRAM**
- 6GB+ VRAM'li GPU'larda rahat calisir
- NVDEC ve NVENC ayri donanim motorlari oldugu icin CUDA inference'i etkilemez

---

## Temel Bilesenler Detayi

### server.py (Ana Sunucu)

FastAPI uygulamasinin giris noktasi. Tum REST endpointlerini, WebSocket yonetimini ve algilama dongulerini icerir. Video test modu ve RTSP modu destekler. Yaklasik 940 satir.

### camera_manager.py

- **CameraReader**: Her kamera icin ayri thread'de RTSP/video okuma
- **MultiCameraManager**: 28 kamera thread'inin yasam dongusunu yonetir
- Frame tamponu ve thread guvenli kuyruk mekanizmasi

### smart_detection.py

- **CameraDetectionState**: Her kameranin algilama durumunu tutar (HIGH/LOW/IDLE)
- **SmartDetectionScheduler**: Hangi kameralarin hangi siklikta islenmesi gerektigini planlar
- At handoff mantigi ve komsu kamera haberlesme protokolu

### ranking_merger.py

- **RankingMerger**: 25 analitik kameradan gelen tespit verilerini birlestir
- 0-2500m araliginda birlesik siralama olusturur
- Cakisan tespitleri cozer ve tutarli siralama uretir

### webrtc_server.py

- **aiortc** tabanli WebRTC sunucu
- SDP offer/answer protokolu
- MJPEG fallback mekanizmasi
- GPU hizlandirmali H.264 encode (NVENC)

### WebRTCPlayer.tsx

- WebRTC baglanti yonetimi
- Otomatik yeniden baglanti
- Basarisiz olursa `MJPEGPlayer.tsx`'e duser (fallback)

### Track2DView.tsx

- 2D oval pist gorunumu
- Atlarin gercek zamanli konumlarini gosterir
- 25 analitik kamera pozisyonlarini isaretler
- Canli animasyonlar (Framer Motion)

### RankingBoard.tsx

- Siralama tablosu
- Jokey silk ikonlari ile gorsel at kimliklendirme
- WebSocket uzerinden gercek zamanli guncelleme
- 4 dil destegi

---

## Operator Paneli Ozellikleri

Operator paneli (`/operator`) su bilesenlerden olusur:

1. **CameraGrid** — 25 analitik kameranin kucuk onizleme grid'i
2. **CameraSettings** — RTSP URL yapilandirma, kamera baslat/durdur
3. **PTZControlPanel** — 3 PTZ kamerasini secme ve izleme
4. **Track2DView** — Oval pist uzerinde atlarin gercek zamanli konumlari
5. **RaceSettings** — Yaris baslat/durdur, siralama sifirlama

---

## Herkese Acik Ekran Ozellikleri

Herkese acik ekran (`/`) su bilesenlerden olusur:

1. **PTZCameraDisplay** — Secili PTZ kamerasinin tam ekran canli goruntusu
2. **RankingBoard** — Gercek zamanli siralama tablosu, silk ikonlari ile
3. **LanguageSelector** — 4 dil arasinda gecis

---

## State Yonetimi

### cameraStore.ts (Zustand)

- Tum kameralarin durumu (bagli/bagli degil, aktif/pasif)
- Stream modu tercihi: `webrtc` veya `mjpeg`
- Kamera ID'leri string formatinda: `analytics-1`, `ptz-1` vb.
- SharedState string `cam_id` anahtarlari kullanir (integer index degil)

### raceStore.ts (Zustand)

- Yaris durumu (baslamadi/devam ediyor/bitti)
- Siralama verileri
- At pozisyonlari

### backendConnection.ts

- WebSocket baglanti yonetimi
- `camera_detection` ve `ranking_update` mesaj isleyicileri
- Otomatik yeniden baglanti mekanizmasi
- Heartbeat (ping/pong) destegi
