// Backend Connection Service
// This service handles WebSocket connection to the AI backend

import { useRaceStore } from '../store/raceStore';
import { useCameraStore } from '../store/cameraStore';
import { findClosestSilkId, getSilkColor } from '../utils/silkUtils';
import { BACKEND_WS_URL } from '../config/backend';

// Configuration
const CONFIG = {
    WS_URL: BACKEND_WS_URL,             // Backend WebSocket URL (race_vision backend)
    RECONNECT_DELAY: 3000,              // Retry connection every 3 seconds
    HEARTBEAT_INTERVAL: 5000,           // Send ping every 5 seconds
};

// Connection state
let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
let isConnected = false;
let manualDisconnect = false;
let connectionVersion = 0;

// Connection status callback
let onStatusChange: ((status: 'connecting' | 'connected' | 'disconnected' | 'error') => void) | null = null;

// Set status change callback
export const setConnectionStatusCallback = (callback: typeof onStatusChange) => {
    onStatusChange = callback;
};

const clearHeartbeat = () => {
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
};

// Connect to backend
export const connectToBackend = () => {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        console.log('Already connected to backend');
        return;
    }

    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }

    manualDisconnect = false;
    console.log(`Connecting to backend: ${CONFIG.WS_URL}`);
    onStatusChange?.('connecting');

    try {
        const socket = new WebSocket(CONFIG.WS_URL);
        ws = socket;

        const myVersion = ++connectionVersion;
        const isStale = () => ws !== socket || myVersion !== connectionVersion;

        socket.onopen = () => {
            if (isStale()) return;

            console.log('Connected to AI backend');
            isConnected = true;
            onStatusChange?.('connected');

            clearHeartbeat();
            heartbeatTimer = setInterval(() => {
                if (!isStale() && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ type: 'ping' }));
                }
            }, CONFIG.HEARTBEAT_INTERVAL);

            if (!isStale() && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: 'get_state' }));
            }
        };

        socket.onmessage = (event) => {
            if (isStale()) return;
            try {
                const message = JSON.parse(event.data);
                handleBackendMessage(message);
            } catch (error) {
                console.error('Failed to parse message:', error);
            }
        };

        socket.onerror = (error) => {
            if (isStale()) return;
            console.error('WebSocket error:', error);
            onStatusChange?.('error');
        };

        socket.onclose = () => {
            if (isStale()) return;

            ws = null;
            isConnected = false;
            onStatusChange?.('disconnected');
            clearHeartbeat();
            console.log('Disconnected from backend');

            if (!manualDisconnect) {
                reconnectTimer = setTimeout(() => {
                    if (manualDisconnect) return;
                    console.log('Reconnecting to backend...');
                    connectToBackend();
                }, CONFIG.RECONNECT_DELAY);
            }
        };
    } catch (error) {
        console.error('Failed to connect:', error);
        onStatusChange?.('error');
    }
};

// Disconnect from backend
export const disconnectFromBackend = () => {
    manualDisconnect = true;
    connectionVersion += 1; // invalidate pending callbacks from previous sockets

    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }

    clearHeartbeat();

    if (ws) {
        const socket = ws;
        ws = null;
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
            socket.close();
        }
    }

    isConnected = false;
    console.log('Disconnected from backend');
};

// Send message to backend
export const sendToBackend = (message: object) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    } else {
        console.warn('Cannot send message: not connected to backend');
    }
};

const toFrontendHorse = (horse: HorseData, fallbackPosition: number) => {
    const detectedColor = horse.color || getDefaultColor(horse.number);
    const silkId = horse.silkId || findClosestSilkId(detectedColor);
    return {
        id: horse.id,
        name: horse.name || `Horse ${horse.number}`,
        number: horse.number,
        color: detectedColor,
        jockeyName: horse.jockeyName || `Jockey ${horse.number}`,
        currentPosition: fallbackPosition,
        currentLap: 1,
        timeElapsed: 0,
        gapToLeader: 0,
        distanceCovered: 0,
        speed: 0,
        lastCameraId: '',
        silkId,
        silkColor: getSilkColor(silkId),
    };
};

const applyHorseRoster = (horses: HorseData[] | undefined) => {
    if (!horses || horses.length === 0) return;
    const { setHorses } = useRaceStore.getState();
    const mapped = horses.map((horse, index) => toFrontendHorse(horse, index + 1));
    setHorses(mapped);
};

// Handle incoming messages from backend
const handleBackendMessage = (message: BackendMessage) => {
    const { updateRankings, startRace, stopRace, setRaceConfig } = useRaceStore.getState();
    const { updateAnalyticsCameraHorses, batchUpdateCameraHorses, batchUpdateCameraStatuses, setActivePTZ } = useCameraStore.getState();

    switch (message.type) {
        // ============ RACE EVENTS ============
        case 'race_start':
            console.log('рџЏЃ Race started from backend');
            if (message.race) {
                setRaceConfig({
                    name: message.race.name,
                    totalLaps: message.race.totalLaps,
                    trackLength: message.race.trackLength,
                    startFinishPosition: message.race.startFinishPosition,
                    status: message.race.status,
                });
            }
            startRace();
            break;

        case 'race_stop':
        case 'race_finish':
            console.log('рџЏЃ Race finished');
            stopRace();
            break;

        // ============ HORSE EVENTS ============
        case 'horses_detected':
            // Backend detected horses at race start
            console.log(`рџЏ‡ ${message.horses?.length} horses detected`);
            applyHorseRoster(message.horses);
            break;

        case 'horse_update':
            // Single horse position update
            // This is called when a horse passes a camera
            console.log(`рџ“· Horse #${message.horse?.number} at camera ${message.cameraId}`);
            break;

        // ============ RANKING EVENTS ============
        case 'ranking_update':
            // Main ranking update - called when horses pass cameras
            console.log('рџ“Љ Ranking update received');
            if (message.rankings) {
                const formattedRankings = message.rankings.map((r: RankingData, index: number) => {
                    // Backend'den gelen renk ile en yakД±n silk'i eЕџleЕџtir
                    const detectedColor = r.color || getDefaultColor(r.number);
                    const silkId = r.silkId || findClosestSilkId(detectedColor);
                    const silkColor = r.silkColor || getSilkColor(silkId);

                    return {
                        id: r.id,
                        name: r.name || `Horse ${r.number}`,
                        number: r.number,
                        color: detectedColor,
                        jockeyName: r.jockeyName || `Jockey ${r.number}`,
                        currentPosition: r.position || index + 1,
                        distanceCovered: r.distanceCovered || 0,
                        currentLap: r.currentLap || 1,
                        timeElapsed: r.timeElapsed || 0,
                        speed: r.speed || 0,
                        gapToLeader: r.gapToLeader || 0,
                        lastCameraId: r.lastCameraId || '',
                        silkId: silkId,
                        silkColor: silkColor,
                    };
                });
                updateRankings(formattedRankings);
            }
            break;

        // ============ CAMERA EVENTS ============
        case 'camera_detection':
            // Backward-compatible payload: { cameraId, horseIds }
            if (message.cameraId && message.horseIds) {
                updateAnalyticsCameraHorses(message.cameraId, message.horseIds);
            }
            // New payload: { cameras: { "analytics-1": ["red", ...], ... } }
            if (message.cameras) {
                const horseUpdates: Record<string, string[]> = {};
                Object.entries(message.cameras).forEach(([cameraId, horseIds]) => {
                    if (Array.isArray(horseIds)) {
                        horseUpdates[cameraId] = horseIds;
                    }
                });
                if (Object.keys(horseUpdates).length > 0) {
                    batchUpdateCameraHorses(horseUpdates);
                }
            }
            break;

        case 'camera_switch':
            // PTZ camera switched
            if (message.cameraId) {
                setActivePTZ(message.cameraId);
            }
            break;

        case 'camera_status':
            // Camera health status broadcast — single batch update (avoids N re-renders)
            if (message.cameras) {
                const statusUpdates: Record<string, 'online' | 'offline'> = {};
                Object.entries(message.cameras).forEach(([cameraId, status]) => {
                    const s = status as { online?: boolean; fps?: number; latency_ms?: number };
                    statusUpdates[cameraId] = s.online ? 'online' : 'offline';
                });
                if (Object.keys(statusUpdates).length > 0) {
                    batchUpdateCameraStatuses(statusUpdates);
                }
            }
            break;

        case 'alert':
            // System alert from backend
            console.warn(`[Alert] ${message.alert_type}: ${message.message}`);
            break;

        // ============ SYSTEM EVENTS ============
        case 'pong':
            // Heartbeat response
            break;

        case 'state':
            // Full state sync
            console.log('рџ“¦ Received full state from backend');
            if (message.race) {
                setRaceConfig(message.race);
            }
            applyHorseRoster(message.horses);
            if (message.rankings) {
                handleBackendMessage({ type: 'ranking_update', rankings: message.rankings });
            }
            break;

        case 'error':
            console.error('вќЊ Backend error:', message.message);
            break;

        default:
            console.log('Unknown message type:', message.type);
    }
};

// Helper function for default horse colors
const getDefaultColor = (number: number): string => {
    const colors = [
        '#EF4444', '#F97316', '#EAB308', '#22C55E', '#14B8A6',
        '#3B82F6', '#8B5CF6', '#EC4899', '#6366F1', '#06B6D4'
    ];
    return colors[(number - 1) % colors.length];
};

// Get connection status
export const isBackendConnected = () => isConnected;

// ============ MESSAGE TYPES ============

interface BackendMessage {
    type: string;
    race?: {
        name: string;
        totalLaps: number;
        trackLength: number;
        startFinishPosition?: number;
        status?: 'pending' | 'active' | 'finished';
    };
    horses?: HorseData[];
    horse?: HorseData;
    rankings?: RankingData[];
    cameraId?: string;
    horseIds?: string[];
    cameras?: Record<string, string[]>;
    message?: string;
}

interface HorseData {
    id: string;
    number: number;
    name?: string;
    color?: string;
    jockeyName?: string;
    silkId?: number;
    silkColor?: string;
}

interface RankingData {
    id: string;
    number: number;
    name?: string;
    color?: string;
    jockeyName?: string;
    position?: number;
    distanceCovered?: number;
    currentLap?: number;
    timeElapsed?: number;
    speed?: number;
    gapToLeader?: number;
    lastCameraId?: string;
    silkId?: number;
    silkColor?: string;
}