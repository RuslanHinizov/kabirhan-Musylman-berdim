# 🏇 KABIRHAN - Horse Racing Live Broadcasting System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![React](https://img.shields.io/badge/React-19.0.0-61DAFB.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7.3-3178C6.svg)
![Vite](https://img.shields.io/badge/Vite-7.3.0-646CFF.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Professional Horse Racing Live Tracking and Broadcasting System**

*Built for Kazakhstan State Horse Racing Federation*

[Installation](#-installation) • [Architecture](#-architecture) • [Backend Integration](#-backend-integration) • [API Reference](#-api-reference)

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Technology Stack](#-technology-stack)
4. [Installation](#-installation)
5. [Project Structure](#-project-structure)
6. [Architecture](#-architecture)
7. [Backend Integration](#-backend-integration)
8. [API Reference](#-api-reference)
9. [Component Documentation](#-component-documentation)
10. [State Management](#-state-management)
11. [Internationalization](#-internationalization)
12. [Camera System](#-camera-system)
13. [Configuration](#-configuration)
14. [Deployment](#-deployment)

---

## 🎯 Overview

KABIRHAN is a professional-grade horse racing live tracking and broadcasting system designed for real-time race monitoring. The system provides:

- **Operator Panel**: Complete race control and monitoring interface
- **Public Display**: Minimalist TV broadcast-ready viewer interface
- **Real-time Tracking**: Live horse positions on 2D track visualization
- **Multi-camera Support**: 4 PTZ cameras + 25 fixed cameras
- **AI Backend Integration**: WebSocket-based real-time communication

---

## ✨ Features

### Operator Panel Features
| Feature | Description |
|---------|-------------|
| PTZ Camera Control | Switch between 4 PTZ cameras |
| Camera Grid View | View all cameras simultaneously |
| 2D Track Visualization | Real-time horse positions on stadium track |
| Camera Configuration | RTSP URL management for all cameras |
| Race Settings | Horse management, race configuration |
| Multi-language Support | TR, RU, KK, EN languages |

### Public Display Features
| Feature | Description |
|---------|-------------|
| Clean Broadcast UI | Minimalist design for TV broadcast |
| Live Camera Feed | HLS streaming from active PTZ camera |
| Race Timer | MM:SS.d format with lap counter |
| Speed Indicator | Real-time leader speed (km/h) |
| Top 10 Rankings | Animated jockey silk icons |
| Cross-tab Sync | Auto-sync with operator panel |

---

## 🛠 Technology Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 19.0.0 | UI Framework |
| TypeScript | 5.7.3 | Type Safety |
| Vite | 7.3.0 | Build Tool |
| Zustand | 5.0.3 | State Management |
| Framer Motion | 12.4.7 | Animations |
| i18next | 24.2.3 | Internationalization |
| HLS.js | 1.5.19 | Video Streaming |
| Lucide React | 0.475.0 | Icons |

### Development
| Tool | Purpose |
|------|---------|
| ESLint | Code Quality |
| TypeScript | Static Analysis |
| Vite | Hot Module Replacement |

---

## 📦 Installation

### Prerequisites
- Node.js >= 18.0.0
- npm >= 9.0.0
- (Optional) FFmpeg for HLS streaming

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/kabirhan-pr.git
cd kabirhan-pr

# Install dependencies
npm install

# Start development server
npm run dev
```

### Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server on port 5173 |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

### Access URLs

| URL | Description |
|-----|-------------|
| `http://localhost:5173` | Public Display |
| `http://localhost:5173/operator` | Operator Panel |

---

## 📁 Project Structure

```
kabirhan-pr/
├── src/
│   ├── components/                    # UI Components
│   │   ├── operator/                  # Operator Panel components
│   │   │   ├── CameraGrid.tsx         # 4-camera grid view
│   │   │   ├── CameraSettings.tsx     # RTSP camera configuration
│   │   │   ├── PTZControlPanel.tsx    # PTZ camera selection
│   │   │   ├── RaceSettings.tsx       # Race & horse management
│   │   │   └── Track2DView.tsx        # 2D track visualization
│   │   ├── public-display/            # Public Display components
│   │   │   ├── PTZCameraDisplay.tsx   # Camera video display
│   │   │   └── RankingBoard.tsx       # Horse rankings board
│   │   ├── LanguageSelector.tsx       # Language picker
│   │   └── RTSPPlayer.tsx             # HLS video player
│   │
│   ├── pages/                         # Main Pages
│   │   ├── OperatorPanel.tsx          # Operator control panel
│   │   └── PublicDisplay.tsx          # Public viewer screen
│   │
│   ├── services/                      # Backend Services
│   │   ├── backendConnection.ts       # WebSocket backend connection
│   │   └── mockBackend.ts             # Simulation engine
│   │
│   ├── store/                         # Zustand State Management
│   │   ├── raceStore.ts               # Race state
│   │   └── cameraStore.ts             # Camera state
│   │
│   ├── config/                        # Configuration
│   │   └── cameras.ts                 # Camera definitions
│   │
│   ├── i18n/                          # Internationalization
│   │   ├── index.ts                   # i18next configuration
│   │   └── locales/
│   │       ├── en.ts                  # English
│   │       ├── tr.ts                  # Turkish
│   │       ├── ru.ts                  # Russian
│   │       └── kk.ts                  # Kazakh
│   │
│   ├── types/                         # TypeScript Types
│   │   └── index.ts                   # All type definitions
│   │
│   ├── App.tsx                        # Main app (routing)
│   ├── App.css                        # Global styles
│   ├── index.css                      # Base styles
│   └── main.tsx                       # Entry point
│
├── dist/                              # Production build output
├── package.json                       # Dependencies
├── tsconfig.json                      # TypeScript config
├── vite.config.ts                     # Vite config
└── eslint.config.js                   # ESLint config
```

---

## 🏗 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│  ┌─────────────────────┐       ┌─────────────────────┐          │
│  │   Operator Panel    │◄─────►│   Public Display    │          │
│  │   (localhost:5173)  │       │ (localhost:5173/    │          │
│  │                     │       │     display)        │          │
│  └──────────┬──────────┘       └──────────┬──────────┘          │
│             │                             │                      │
│             └──────────┬──────────────────┘                      │
│                        │                                         │
│              ┌─────────▼─────────┐                               │
│              │   Zustand Store   │                               │
│              │  ┌─────────────┐  │                               │
│              │  │ raceStore   │  │                               │
│              │  │ cameraStore │  │                               │
│              │  └─────────────┘  │                               │
│              └─────────┬─────────┘                               │
│                        │                                         │
│  ┌─────────────────────┴─────────────────────┐                   │
│  │              Services Layer               │                   │
│  │  ┌────────────────┐  ┌─────────────────┐  │                   │
│  │  │ mockBackend.ts │  │backendConnection│  │                   │
│  │  │  (Simulation)  │  │   (WebSocket)   │  │                   │
│  │  └────────────────┘  └────────┬────────┘  │                   │
│  └───────────────────────────────┼───────────┘                   │
└──────────────────────────────────┼───────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      AI BACKEND             │
                    │   (ws://localhost:8081)     │
                    │                             │
                    │  • Horse Detection (YOLO)   │
                    │  • Position Tracking        │
                    │  • Camera Switching         │
                    │  • Race Management          │
                    └─────────────────────────────┘
```

### Data Flow

```
1. Camera Feed → AI Backend → Horse Detection
2. AI Backend → WebSocket → Frontend
3. Frontend → Zustand Store → UI Update
4. Operator Action → Store → localStorage → Public Display Sync
```

---

## 🔌 Backend Integration

### Connection Configuration

The frontend connects to the AI backend via WebSocket. Configuration is in `src/services/backendConnection.ts`:

```typescript
const CONFIG = {
    WS_URL: 'ws://localhost:8081/ws',  // Backend WebSocket URL
    RECONNECT_DELAY: 3000,              // Retry every 3 seconds
    HEARTBEAT_INTERVAL: 5000,           // Ping every 5 seconds
};
```

### Connection Lifecycle

```typescript
import { connectToBackend, disconnectFromBackend, setConnectionStatusCallback } from './services/backendConnection';

// Set status callback
setConnectionStatusCallback((status) => {
    console.log('Connection status:', status);
    // status: 'connecting' | 'connected' | 'disconnected' | 'error'
});

// Connect
connectToBackend();

// Disconnect (on cleanup)
disconnectFromBackend();
```

### Switching Between Mock and Real Backend

In `OperatorPanel.tsx`, you can toggle between mock simulation and real backend:

```typescript
// Start mock simulation (offline demo)
import { startMockSimulation, stopMockSimulation } from '../services/mockBackend';
startMockSimulation();

// Connect to real backend
import { connectToBackend, disconnectFromBackend } from '../services/backendConnection';
connectToBackend();
```

---

## 📡 API Reference

### WebSocket Message Types

#### Messages FROM Backend (Incoming)

| Type | Description | Payload |
|------|-------------|---------|
| `race_start` | Race has started | `{ race: RaceConfig }` |
| `race_stop` | Race has stopped | `{}` |
| `race_finish` | Race has finished | `{ winner: Horse }` |
| `horses_detected` | Horses detected at start | `{ horses: HorseData[] }` |
| `horse_update` | Single horse position update | `{ horse: HorseData, cameraId: string }` |
| `ranking_update` | Full rankings update | `{ rankings: RankingData[] }` |
| `camera_detection` | Horses in camera view | `{ cameraId: string, horseIds: string[] }` |
| `camera_switch` | PTZ camera switched | `{ cameraId: string }` |
| `pong` | Heartbeat response | `{}` |
| `state` | Full state sync | `{ race, rankings }` |
| `error` | Error message | `{ message: string }` |

#### Messages TO Backend (Outgoing)

| Type | Description | Payload |
|------|-------------|---------|
| `ping` | Heartbeat | `{}` |
| `get_state` | Request current state | `{}` |
| `start_race` | Start the race | `{ raceId: string }` |
| `stop_race` | Stop the race | `{}` |
| `switch_camera` | Switch PTZ camera | `{ cameraId: string }` |

### Data Structures

#### HorseData
```typescript
interface HorseData {
    id: string;           // Unique identifier
    number: number;       // Horse number (1-15)
    name?: string;        // Horse name
    color?: string;       // Color hex code
    jockeyName?: string;  // Jockey name
}
```

#### RankingData
```typescript
interface RankingData {
    id: string;
    number: number;
    name?: string;
    color?: string;
    jockeyName?: string;
    position?: number;        // Current rank (1-15)
    distanceCovered?: number; // Distance in meters
    currentLap?: number;      // Current lap number
    timeElapsed?: number;     // Time in seconds
    speed?: number;           // Speed in m/s
    gapToLeader?: number;     // Gap to leader in seconds
    lastCameraId?: string;    // Last camera passed
}
```

#### RaceConfig
```typescript
interface RaceConfig {
    name: string;
    totalLaps: number;
    trackLength: number;
    status?: 'idle' | 'active' | 'finished';
}
```

### Backend Implementation Guide

Your AI backend should implement the following:

```python
# Python WebSocket server example (using websockets library)
import asyncio
import websockets
import json

connected_clients = set()

async def handler(websocket):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))
            
            elif data['type'] == 'get_state':
                state = get_current_state()
                await websocket.send(json.dumps({
                    'type': 'state',
                    'race': state['race'],
                    'rankings': state['rankings']
                }))
            
            elif data['type'] == 'start_race':
                start_race()
                await broadcast({'type': 'race_start', 'race': get_race_config()})
            
    finally:
        connected_clients.remove(websocket)

async def broadcast(message):
    """Send message to all connected clients"""
    if connected_clients:
        await asyncio.gather(
            *[client.send(json.dumps(message)) for client in connected_clients]
        )

# When horse positions update (from AI detection):
async def on_horse_positions_updated(rankings):
    await broadcast({
        'type': 'ranking_update',
        'rankings': rankings
    })

# When horse passes a camera:
async def on_camera_detection(camera_id, horse_ids):
    await broadcast({
        'type': 'camera_detection',
        'cameraId': camera_id,
        'horseIds': horse_ids
    })

# Start server
async def main():
    async with websockets.serve(handler, "localhost", 8081):
        await asyncio.Future()  # run forever

asyncio.run(main())
```

---

## 🧩 Component Documentation

### Pages

#### OperatorPanel.tsx
Main control panel for race operators.

```typescript
// Features:
// - 5 tab navigation (PTZ, Grid, Track, Cameras, Settings)
// - Backend connection toggle
// - Language selector
// - Keyboard shortcuts (1-5 for tabs)

// Key functions:
const toggleBackend = () => {
    if (useMockBackend) {
        stopMockSimulation();
        connectToBackend();
    } else {
        disconnectFromBackend();
        startMockSimulation();
    }
};
```

#### PublicDisplay.tsx
Minimalist broadcast display for viewers.

```typescript
// Features:
// - Full-screen camera view
// - Race timer (MM:SS.d format)
// - Speed indicator (km/h)
// - Top 10 horse rankings with jockey silks
// - Auto-sync with operator panel

// Time formatting:
const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toFixed(1).padStart(4, '0')}`;
};
```

### Operator Components

#### Track2DView.tsx
2D stadium track visualization with real-time horse positions.

```typescript
// Track geometry:
const TRACK_LENGTH = 2500;      // meters
const straightLen = 450;         // pixels
const curveRadius = 160;         // pixels

// Position calculation:
const getPos = (pct: number, laneOffset: number = 0) => {
    // Returns {x, y} coordinates on the track
    // pct: 0-1 (percentage around track)
    // laneOffset: negative=inside, positive=outside
};

// Smooth animation hook:
const useSmooth = (target: number, speed: number = 0.2) => {
    // Uses requestAnimationFrame for smooth interpolation
};
```

#### CameraSettings.tsx
RTSP camera URL configuration and stream management.

```typescript
// Features:
// - Edit RTSP URLs for all cameras
// - Start/stop individual streams
// - Stream status indicators
// - Auto-refresh every 5 seconds
```

#### PTZControlPanel.tsx
PTZ camera selection with preview.

```typescript
// Features:
// - 4 PTZ camera buttons
// - Active camera highlight
// - Stream preview
// - localStorage sync for cross-tab updates
```

### Public Display Components

#### RankingBoard.tsx
Animated horse rankings with position change indicators.

```typescript
// Features:
// - Top horses display
// - Position change arrows (↑↓)
// - Jockey silk colors
// - Gap to leader display
// - Framer Motion animations
```

### Shared Components

#### RTSPPlayer.tsx
HLS video player with connection status.

```typescript
// Features:
// - Native HLS (Safari) / HLS.js fallback
// - Connection status indicators
// - Auto-reconnect on errors
// - Muted autoplay

interface RTSPPlayerProps {
    camera: CameraConfig;
    className?: string;
    showControls?: boolean;
    muted?: boolean;
    autoPlay?: boolean;
}
```

#### LanguageSelector.tsx
Multi-language dropdown with localStorage sync.

```typescript
// Supported languages:
const languages = [
    { code: 'tr', name: 'Türkçe', flag: '🇹🇷' },
    { code: 'ru', name: 'Русский', flag: '🇷🇺' },
    { code: 'kk', name: 'Қазақша', flag: '🇰🇿' },
    { code: 'en', name: 'English', flag: '🇬🇧' },
];
```

---

## 📊 State Management

### raceStore.ts

```typescript
interface RaceState {
    race: {
        id: string;
        name: string;
        status: 'idle' | 'active' | 'finished';
        totalLaps: number;
        currentLap: number;
        startTime: number | null;
        trackLength: number;
        startFinishPosition: number;
    };
    horses: Horse[];
    rankings: Horse[];
}

// Actions:
startRace()              // Start the race
stopRace()               // Stop the race
resetRace()              // Reset to initial state
addHorse(horse)          // Add a horse
removeHorse(id)          // Remove a horse
updateRankings(horses)   // Update rankings from backend
setRaceConfig(config)    // Set race configuration
clearHorses()            // Remove all horses
```

### cameraStore.ts

```typescript
interface CameraState {
    ptzCameras: PTZCamera[];
    fixedCameras: FixedCamera[];
    activePTZCameraId: string;
}

// Actions:
setActivePTZCamera(id)           // Set active PTZ camera + localStorage
setActivePTZ(id)                 // Set active PTZ (from backend)
syncFromStorage()                // Sync from localStorage
updateFixedCameraHorses(id, ids) // Update horses in camera view
initializeFromConfig(cameras)    // Initialize camera list
```

### Cross-Tab Synchronization

```typescript
// Language sync (i18n/index.ts)
window.addEventListener('storage', (event) => {
    if (event.key === 'language' && event.newValue) {
        i18n.changeLanguage(event.newValue);
    }
});

// Camera sync (cameraStore.ts)
window.addEventListener('storage', (event) => {
    if (event.key === 'activePTZCamera' && event.newValue) {
        useCameraStore.getState().syncFromStorage();
    }
});
```

---

## 🌍 Internationalization

### Supported Languages

| Code | Language | Flag |
|------|----------|------|
| `tr` | Turkish | 🇹🇷 |
| `ru` | Russian | 🇷🇺 |
| `kk` | Kazakh | 🇰🇿 |
| `en` | English | 🇬🇧 |

### Usage

```typescript
import { useTranslation } from 'react-i18next';

const MyComponent = () => {
    const { t } = useTranslation();
    
    return <h1>{t('header.raceControl')}</h1>;
};
```

### Translation Keys

```typescript
// Example translation structure (locales/en.ts)
export default {
    header: {
        raceControl: 'Race Control Panel',
        lap: 'Lap',
        speed: 'Speed',
    },
    track: {
        liveRaceTrack: 'Live Race Track',
        stadium: 'Stadium',
        leader: 'Leader',
    },
    race: {
        status: 'Race Status',
        idle: 'Idle',
        active: 'Active',
        finished: 'Finished',
    },
    // ... more keys
};
```

---

## 📹 Camera System

### Camera Types

| Type | Count | Positions | Purpose |
|------|-------|-----------|---------|
| PTZ | 4 | 0m, 625m, 1250m, 1875m | Main broadcast cameras |
| Fixed | 25 | Every 100m | Horse detection |

### Camera Configuration

```typescript
// config/cameras.ts

export const PTZ_CAMERAS: CameraConfig[] = [
    {
        id: 'ptz-1',
        name: 'PTZ Camera 1',
        type: 'ptz' as const,
        position: 0,
        rtspUrl: 'rtsp://192.168.1.101:554/stream1',
        hlsUrl: '/streams/ptz-1/index.m3u8',
        status: 'online' as const,
    },
    // ... 3 more PTZ cameras
];

export const FIXED_CAMERAS: CameraConfig[] = [
    {
        id: 'fixed-0',
        name: 'Fixed Camera 0m',
        type: 'fixed' as const,
        position: 0,
        rtspUrl: 'rtsp://192.168.1.201:554/stream1',
        status: 'online' as const,
    },
    // ... 24 more fixed cameras
];
```

### HLS Streaming

The frontend expects HLS streams at:
```
http://localhost:8080/streams/{camera-id}/index.m3u8
```

To convert RTSP to HLS, you need a streaming server (e.g., using FFmpeg):

```bash
# Example FFmpeg command for RTSP to HLS
ffmpeg -i rtsp://192.168.1.101:554/stream1 \
    -c:v copy \
    -c:a aac \
    -f hls \
    -hls_time 2 \
    -hls_list_size 3 \
    -hls_flags delete_segments \
    /path/to/streams/ptz-1/index.m3u8
```

---

## ⚙️ Configuration

### Constants (types/index.ts)

```typescript
export const TRACK_LENGTH = 2500;      // Track length in meters
export const TOTAL_LAPS = 3;           // Default lap count
export const MAX_HORSES = 15;          // Maximum horses per race
export const TOP_HORSES_DISPLAY = 10;  // Top horses to show
export const FIXED_CAMERA_COUNT = 25;  // Number of fixed cameras
export const CAMERA_SPACING = 100;     // Distance between cameras (m)
```

### Environment Variables

Create a `.env` file for environment-specific settings:

```env
VITE_WS_URL=ws://localhost:8081/ws
VITE_HLS_BASE_URL=http://localhost:8080
```

---

## 🚀 Deployment

### Production Build

```bash
# Build for production
npm run build

# Output in dist/ folder:
# dist/
# ├── index.html          (0.46 kB)
# └── assets/
#     ├── index.css       (35.49 kB)
#     └── index.js        (977.88 kB, gzip: 308.66 kB)
```

### Static Hosting

The built files can be served by any static file server:

```bash
# Using serve
npx serve dist

# Using nginx
server {
    listen 80;
    root /path/to/dist;
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### Docker Deployment

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 🧪 Testing

### Test Commands

```bash
# TypeScript type checking
npx tsc --noEmit

# ESLint
npx eslint src --ext .ts,.tsx

# ESLint with no warnings allowed
npx eslint src --ext .ts,.tsx --max-warnings 0
```

### Current Test Status

| Test | Status |
|------|--------|
| TypeScript Compile | ✅ 0 errors |
| ESLint | ✅ 0 errors, 0 warnings |
| Production Build | ✅ Success |

---

## 📄 License

MIT License

---

## 👥 Contributors

Built for the **Kazakhstan State Horse Racing Federation** 🇰🇿

---

---

# 🇷🇺 ДОКУМЕНТАЦИЯ НА РУССКОМ ЯЗЫКЕ

## 🎨 Система Silk (Жокейские формы)

### Принцип работы

Система автоматически сопоставляет цвета жокейских форм с иконками:

1. **Backend** определяет доминирующий цвет формы жокея (например `#DC2626`)
2. **Frontend** использует алгоритм **Euclidean RGB distance** для поиска ближайшего Silk
3. Отображается соответствующая **SVG иконка** жокея

### Таблица цветов (10 Silk)

| ID | Цвет | HEX | Файл |
|----|------|-----|------|
| 1 | Red (Красный) | #DC2626 | silk_1.svg |
| 2 | Blue (Синий) | #2563EB | silk_2.svg |
| 3 | Green (Зелёный) | #16A34A | silk_3.svg |
| 4 | Yellow (Жёлтый) | #FBBF24 | silk_4.svg |
| 5 | Purple (Фиолетовый) | #9333EA | silk_5.svg |
| 6 | Orange (Оранжевый) | #EA580C | silk_6.svg |
| 7 | Pink (Розовый) | #EC4899 | silk_7.svg |
| 8 | Cyan (Голубой) | #06B6D4 | silk_8.svg |
| 9 | Lime (Лайм) | #84CC16 | silk_9.svg |
| 10 | Orange Alt | #F97316 | silk_10.svg |

### Расположение файлов

```
public/assets/silks/
├── silk_1.svg   (Красный жокей)
├── silk_2.svg   (Синий жокей)
├── silk_3.svg   (Зелёный жокей)
├── silk_4.svg   (Жёлтый жокей)
├── silk_5.svg   (Оранжевый жокей)
├── silk_6.svg   (Фиолетовый жокей)
├── silk_7.svg   (Розовый жокей)
├── silk_8.svg   (Голубой жокей)
├── silk_9.svg   (Лаймовый жокей)
└── silk_10.svg  (Оранжевый вариант)
```

### Функции silkUtils.ts

```typescript
// Найти ближайший Silk по цвету
findClosestSilkId('#DC2626');  // → 1 (Red)

// Получить путь к SVG иконке
getSilkImagePath(1);  // → '/assets/silks/silk_1.svg'

// Получить HEX цвет Silk
getSilkColor(1);  // → '#DC2626'

// Получить Silk по умолчанию для лошади (циклично)
getDefaultSilkId(15);  // → 5 (15 % 10 + 1)

// Получить название цвета
getSilkName(1);  // → 'Red'
```

### Алгоритм сопоставления цветов

```typescript
// silkUtils.ts - Euclidean RGB distance
export const colorDistance = (color1: string, color2: string): number => {
    const rgb1 = hexToRgb(color1);
    const rgb2 = hexToRgb(color2);
    return Math.sqrt(
        Math.pow(rgb1.r - rgb2.r, 2) +
        Math.pow(rgb1.g - rgb2.g, 2) +
        Math.pow(rgb1.b - rgb2.b, 2)
    );
};

export const findClosestSilkId = (detectedColor: string): number => {
    let closestId = 1;
    let minDistance = Infinity;

    for (let silkId = 1; silkId <= TOTAL_SILKS; silkId++) {
        const silkColor = SILK_COLORS[silkId];
        const distance = colorDistance(detectedColor, silkColor);
        if (distance < minDistance) {
            minDistance = distance;
            closestId = silkId;
        }
    }
    return closestId;
};
```

---

## 📺 Публичный дисплей (PublicDisplay)

### Дизайн в стиле LONGINES

```
┌────────────────────────────────────────────────────────────────┐
│ ┌──────────┐                                                   │
│ │  1:16.2  │                      VIDEO                        │
│ │  Lap 1/3 │                                                   │
│ └──────────┘                                                   │
│                                                                │
│ ┌──────────────────────────────────────────────────┬─────────┐ │
│ │ ⭕ │ 🎽 │ 🎽 │ 🎽 │ 🎽 │ 🎽 │ 🎽 │ 🎽 │ 🎽 │ 🎽 │ 55.4   │ │
│ │    │  5 │ 10 │ 13 │  7 │  6 │  9 │ 11 │  2 │  4 │ km/h   │ │
│ └──────────────────────────────────────────────────┴─────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### Элементы интерфейса

| Элемент | Расположение | Описание |
|---------|--------------|----------|
| Таймер | Верх-лево | Время гонки (MM:SS.d) + номер круга |
| Видео | Центр | MJPEG поток с камеры |
| Маркер | Низ-лево | Красный круг (позиция старта) |
| Жокеи | Низ-центр | SVG иконки + номера лошадей |
| Спидометр | Низ-право | Скорость лидера (км/ч) |

### Стилизация нижней панели

```typescript
// Фон: тёмно-зелёный градиент
background: 'linear-gradient(to right, #0a1a0a, #0d1f0d, #0a1a0a)'

// Высота панели
height: 120px

// Размер иконок жокеев
height: 70px

// Анимация смены позиций: 2 секунды
transition: { type: 'spring', stiffness: 80, damping: 20, duration: 2 }
```

---

## 🔌 Интеграция с Backend

### Конфигурация подключения

```typescript
// services/backendConnection.ts
const CONFIG = {
    WS_URL: 'ws://localhost:8000/ws',  // WebSocket URL
    RECONNECT_DELAY: 3000,              // Переподключение через 3 сек
    HEARTBEAT_INTERVAL: 5000,           // Ping каждые 5 сек
};
```

### Формат сообщений от Backend

#### ranking_update - Обновление рейтинга

```json
{
    "type": "ranking_update",
    "rankings": [
        {
            "id": "horse_1",
            "number": 5,
            "name": "Thunder Bolt",
            "color": "#16A34A",
            "jockeyName": "John Smith",
            "position": 1,
            "distanceCovered": 1250.5,
            "currentLap": 2,
            "timeElapsed": 76.2,
            "speed": 15.3,
            "gapToLeader": 0
        }
    ]
}
```

**Важно:** Поле `color` используется для автоматического сопоставления с Silk!

#### horses_detected - Обнаружены лошади

```json
{
    "type": "horses_detected",
    "horses": [
        {
            "id": "horse_1",
            "number": 1,
            "color": "#DC2626",
            "name": "Lightning",
            "jockeyName": "Mike"
        }
    ]
}
```

### Обработка данных на Frontend

```typescript
// backendConnection.ts - обработка ranking_update
case 'ranking_update':
    if (message.rankings) {
        const formattedRankings = message.rankings.map((r, index) => {
            // Автоматическое сопоставление цвета с Silk
            const detectedColor = r.color || getDefaultColor(r.number);
            const silkId = r.silkId || findClosestSilkId(detectedColor);
            const silkColor = r.silkColor || getSilkColor(silkId);

            return {
                id: r.id,
                number: r.number,
                name: r.name,
                color: detectedColor,
                jockeyName: r.jockeyName,
                currentPosition: r.position || index + 1,
                silkId: silkId,        // ← Silk ID
                silkColor: silkColor,  // ← Цвет Silk
                // ... остальные поля
            };
        });
        updateRankings(formattedRankings);
    }
    break;
```

---

## 📁 Структура ключевых файлов

### types/index.ts - Типы и константы

```typescript
// Константы
export const TRACK_LENGTH = 2500;      // Длина трассы (м)
export const MAX_HORSES = 50;          // Максимум лошадей
export const TOTAL_SILKS = 10;         // Количество Silk
export const TOP_HORSES_DISPLAY = 10;  // Топ в рейтинге

// Цвета Silk
export const SILK_COLORS: Record<number, string> = {
    1: '#DC2626',  // Red
    2: '#2563EB',  // Blue
    3: '#16A34A',  // Green
    4: '#FBBF24',  // Yellow
    5: '#9333EA',  // Purple
    6: '#EA580C',  // Orange
    7: '#EC4899',  // Pink
    8: '#06B6D4',  // Cyan
    9: '#84CC16',  // Lime
    10: '#F97316', // Orange variant
};

// Интерфейс лошади
export interface Horse {
    id: string;
    name: string;
    number: number;
    color: string;
    jockeyName: string;
    silkId: number;      // ID формы жокея
    silkColor: string;   // Цвет формы
    currentPosition: number;
    distanceCovered: number;
    currentLap: number;
    timeElapsed: number;
    speed: number;
    gapToLeader: number;
}
```

### store/raceStore.ts - Состояние гонки

```typescript
interface RaceStore {
    race: Race;           // Информация о гонке
    horses: Horse[];      // Список лошадей
    rankings: Horse[];    // Текущий рейтинг

    // Действия
    addHorse(horse): void;
    updateRankings(rankings): void;
    startRace(): void;
    stopRace(): void;
    resetRace(): void;
}
```

### pages/PublicDisplay.tsx - Публичный дисплей

Основные элементы:
- `AnimatedTime` - анимированный таймер
- `AnimatedNumber` - анимированный спидометр
- `topHorses` - топ 10 лошадей для отображения
- `getSilkImagePath()` - путь к SVG иконке жокея

---

## 🚀 Запуск системы

### Режим разработки

```bash
cd Kabirhan-Frontend
npm install
npm run dev
```

Откроется `http://localhost:5173/operator`

### Сборка для продакшена

```bash
npm run build
```

Файлы в папке `dist/`

### URL адреса

| URL | Описание |
|-----|----------|
| `http://localhost:5173/` | Публичный дисплей |
| `http://localhost:5173/operator` | Панель оператора |

---

## 🛠 Добавление нового Silk

1. Создайте SVG файл `public/assets/silks/silk_11.svg`

2. Добавьте цвет в `src/types/index.ts`:
```typescript
export const SILK_COLORS = {
    // ...
    11: '#НОВЫЙ_ЦВЕТ',
};
```

3. Обновите константу:
```typescript
export const TOTAL_SILKS = 11;
```

4. (Опционально) Добавьте название в `silkUtils.ts`:
```typescript
const names = {
    // ...
    11: 'New Color',
};
```

---

## 📝 Важные замечания

1. **Backend должен отправлять поле `color`** в формате HEX (например `#DC2626`)

2. **Silk сопоставляется автоматически** - не нужно отправлять `silkId` с backend

3. **Анимации занимают 2 секунды** - это создаёт плавный эффект смены позиций

4. **SVG иконки имеют прозрачный фон** - они хорошо смотрятся на любом фоне

5. **Поддерживается до 50 лошадей** - Silk назначается циклично (лошадь 11 = Silk 1)

---

<div align="center">

**🏇 KABIRHAN - Professional Horse Racing System**

*Fast. Accurate. Beautiful.*

</div>
