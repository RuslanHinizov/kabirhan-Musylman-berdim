# Race Vision Frontend

## Genel Bakis

React 19 + TypeScript + Vite tabanli at yarisi yayin sistemi frontend uygulamasi. Iki ana sayfa: Operator Paneli ve Herkese Acik Ekran.

Sistem, 25 analitik kamera ve 3 PTZ yayin kamerasini yoneten bir backend sunucusuyla WebSocket uzerinden iletisim kurar. Analitik kameralar YOLO + CNN tabanli at tespiti yaparken, PTZ kameralari yalnizca izleyicilere gosterilen yayin goruntusu saglar.

## Teknolojiler

| Teknoloji | Surum | Aciklama |
|-----------|-------|----------|
| React | 19.2.0 | UI kutuphanesi |
| TypeScript | 5.9.3 | Tip guvenli JavaScript |
| Vite | 7.2.4 | Build araci ve gelistirme sunucusu |
| Zustand | 5.0.10 | State management |
| Framer Motion | 12.29.2 | Animasyonlar |
| Tailwind CSS | 4.1.18 | Utility-first CSS framework |
| i18next | 25.8.0 | Coklu dil destegi |
| react-i18next | 16.5.4 | React i18n entegrasyonu |
| lucide-react | 0.563.0 | Ikon kutuphanesi |
| hls.js | 1.6.15 | HLS streaming |
| react-router-dom | 7.13.0 | Sayfa yonlendirme |

## Sayfalar

### Operator Paneli (`/operator`)

Yaris yonetimi icin kontrol paneli. 5 sekme icerir:

1. **PTZ Kontrol** -- Yayin kamerasi secimi (3 PTZ kamera arasinda gecis). Aktif kamera "ON AIR" rozeti ile isaretlenir.
2. **Kamera Grid** -- 25 analitik kameranin canli goruntusu (5x5 grid). Her kamera genisletilebilir.
3. **Pist** -- 2D oval pist uzerinde at pozisyonlari animasyonlu olarak gosterilir.
4. **RTSP Kameralar** -- Kamera URL yapilandirmasi ve baslat/durdur kontrolleri.
5. **Yaris Ayarlari** -- Yaris parametreleri, at ekleme/silme islemleri.

### Herkese Acik Ekran (`/`)

Izleyici ekrani. Tam ekran yayin gorunumu:

- PTZ kamera tam ekran video (WebRTC, MJPEG fallback)
- Sol ust: Yaris suresi ve tur bilgisi
- Alt bar: Jokey ikonlari ile hareket eden siralama (Dubai tarzi yayin gorunumu)
- Hiz gostergesi (km/h)
- Yaris bitis overlay'i

## Bilesen Yapisi

### Sayfa Bilesenleri

```
pages/
+-- OperatorPanel.tsx    -- Sekmeli operator kontrol paneli
+-- PublicDisplay.tsx    -- Tam ekran izleyici gorunumu
```

### Operator Bilesenleri

```
components/operator/
+-- CameraGrid.tsx       -- 25 kamera grid (genisletilebilir)
+-- CameraSettings.tsx   -- RTSP URL yonetimi
+-- PTZControlPanel.tsx  -- PTZ kamera secimi (ON AIR rozeti)
+-- Track2DView.tsx      -- 2D oval pist animasyonu
+-- RaceSettings.tsx     -- Yaris + at konfigurasyonu
```

### Herkese Acik Ekran Bilesenleri

```
components/public-display/
+-- PTZCameraDisplay.tsx -- PTZ kamera goruntusu + overlay bilgileri
+-- RankingBoard.tsx     -- Silk ikonu kartlari ile siralama
```

### Ortak Bilesenler

```
components/
+-- WebRTCPlayer.tsx     -- WebRTC video player (MJPEG fallback)
+-- MJPEGPlayer.tsx      -- MJPEG stream player (WebRTC fallback olarak kullanilir)
+-- RTSPPlayer.tsx       -- RTSP/HLS stream player
+-- LanguageSelector.tsx -- Dil secici dropdown
```

## State Management (Zustand)

Uygulama iki ana store kullanir:

### cameraStore.ts

Kamera durumu ve yapilandirmasi:

- `ptzCameras` -- 3 PTZ kamera durumu (url, online/offline, isim)
- `analyticsCameras` -- 25 analitik kamera durumu
- `activePTZCameraId` -- Aktif yayin kamerasi ID'si
- `streamMode` -- `'webrtc'` veya `'mjpeg'` gecisi
- RTSP URL'leri `localStorage`'da saklanir
- Sekmeler arasi senkronizasyon (`storage` event dinleme)

### raceStore.ts

Yaris durumu ve siralama:

- `race` -- Yaris bilgileri (ad, tur, durum)
- `horses` -- At listesi (isim, renk, silk ikonu)
- `rankings` -- Canli siralama verileri
- `backendConnected` -- Backend baglanti durumu

## Backend Baglantisi

`src/services/backendConnection.ts` dosyasi backend ile WebSocket iletisimini yonetir:

- **Adres:** `ws://localhost:8000/ws`
- **Otomatik yeniden baglanma:** Baglanti kopusunda 3 saniye arayla tekrar dener
- **Heartbeat:** 5 saniye arayla ping/pong kontrolu
- **Mesaj tipleri:**
  - `ranking_update` -- Canli siralama guncellemesi
  - `camera_detection` -- Kamera tespit sonuclari
  - `race_start` -- Yaris basladi bildirimi
  - `race_stop` -- Yaris durdu bildirimi
  - `horses_detected` -- At tespit bildirimi
  - `state` -- Genel durum senkronizasyonu

## Kamera Konfigurasyonu

`src/config/cameras.ts` dosyasinda tanimlanir:

### Analitik Kameralar (25 adet)

- ID: `analytics-1` ile `analytics-25` arasi
- Her kamera 100 metrelik bir pist segmentini kapsar
- Toplam kapsama: 2500 metre
- Yalnizca YOLO + CNN tespiti icin kullanilir, izleyicilere gosterilmez

### PTZ Kameralari (3 adet)

- `ptz-1` -- Baslangic noktasi (0m)
- `ptz-2` -- Orta nokta (1250m)
- `ptz-3` -- Bitis noktasi (2500m)
- Yalnizca yayin icin kullanilir, tespit yapilmaz

### Baslangic Durumu

- Tum kameralar baslangicta offline
- RTSP URL'leri bos
- URL'ler operator panelinden yapilandirilir

## Dil Destegi (i18n)

`src/i18n/` klasorunde yapilandirilir. 4 dil desteklenir:

| Kod | Dil | Dosya |
|-----|-----|-------|
| EN | Ingilizce | `locales/en.ts` |
| TR | Turkce | `locales/tr.ts` |
| RU | Rusca | `locales/ru.ts` |
| KK | Kazakca | `locales/kk.ts` |

Ozellikler:

- Varsayilan dil: Ingilizce (EN)
- Tarayici dili otomatik algilama
- `localStorage`'da tercih kaydi
- Sekmeler arasi dil senkronizasyonu
- Tum UI metinleri cevrilmis

## Stil Sistemi

- Tailwind CSS 4 utility siniflari kullanilir
- Koyu tema tasarimi
- CSS custom properties (design tokens):

| Degisken | Deger | Aciklama |
|----------|-------|----------|
| `--background` | `#09090B` | Ana arka plan |
| `--surface` | `#18181B` | Kart/panel arka plani |
| `--primary` | `#3B82F6` | Ana renk (mavi) |
| `--accent` | `#10B981` | Vurgu rengi (yesil) |

- Ozel bilesenler: `.racing-card`, `.racing-button`, `.live-badge`
- Pulsing LIVE rozeti animasyonu (canli yayin gostergesi)

## Calistirma

### Gereksinimler

- Node.js 18+
- Backend sunucu: `http://localhost:8000`

### Gelistirme

```bash
npm install          # Bagimliliklari yukle
npm run dev          # Gelistirme sunucusu (http://localhost:5173)
```

### Production Build

```bash
npm run build        # dist/ klasorune production build
npm run preview      # Production build onizleme
```

### Lint

```bash
npm run lint         # ESLint ile kod kontrolu
```

## Jokey Silk Ikonlari

`public/assets/silks/` klasorunde 10 adet SVG jokey ikonu bulunur:

- `silk_1.svg` -- `silk_10.svg`
- Her at bir silk ikonu ile eslestirilir
- Renkli jokey yelekleri olarak tasarlanmistir
- Siralama tahtasinda ve pist gorunumunde kullanilir

## Klasor Yapisi

```
Kabirhan-Frontend/
+-- public/
|   +-- assets/
|       +-- silks/              -- Jokey silk SVG ikonlari
+-- src/
|   +-- components/
|   |   +-- operator/           -- Operator paneli bilesenleri
|   |   +-- public-display/     -- Izleyici ekrani bilesenleri
|   |   +-- WebRTCPlayer.tsx
|   |   +-- MJPEGPlayer.tsx
|   |   +-- RTSPPlayer.tsx
|   |   +-- LanguageSelector.tsx
|   +-- config/
|   |   +-- cameras.ts          -- Kamera tanimlari
|   +-- i18n/
|   |   +-- index.ts            -- i18n yapilandirmasi
|   |   +-- locales/            -- Dil dosyalari (en, tr, ru, kk)
|   +-- pages/
|   |   +-- OperatorPanel.tsx
|   |   +-- PublicDisplay.tsx
|   +-- services/
|   |   +-- backendConnection.ts -- WebSocket baglantisi
|   +-- store/
|       +-- cameraStore.ts      -- Kamera state
|       +-- raceStore.ts        -- Yaris state
+-- package.json
+-- vite.config.ts
+-- tsconfig.json
+-- tailwind.config.ts
```
