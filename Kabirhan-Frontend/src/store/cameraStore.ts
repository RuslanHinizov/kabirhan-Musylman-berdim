import { create } from 'zustand';
import { ANALYTICS_CAMERAS, PTZ_CAMERAS } from '../config/cameras';
import type { AnalyticsCameraConfig, PTZCameraConfig } from '../config/cameras';

// Store camera types (extend config with runtime state)
export interface AnalyticsCameraState extends AnalyticsCameraConfig {
    horsesInView: string[];
}

export interface PTZCameraState extends PTZCameraConfig {
    isActive: boolean;
}

interface CameraState {
    ptzCameras: PTZCameraState[];
    analyticsCameras: AnalyticsCameraState[];
    activePTZCameraId: string;
    streamMode: 'webrtc' | 'mjpeg';

    // Actions
    setActivePTZCamera: (cameraId: string) => void;
    setActivePTZ: (cameraId: string) => void;
    setStreamMode: (mode: 'webrtc' | 'mjpeg') => void;
    updateCameraStatus: (cameraId: string, status: 'online' | 'offline') => void;
    batchUpdateCameraStatuses: (updates: Record<string, 'online' | 'offline'>) => void;
    updateAnalyticsCameraHorses: (cameraId: string, horseIds: string[]) => void;
    batchUpdateCameraHorses: (updates: Record<string, string[]>) => void;
    initializeCameras: () => void;
    syncFromStorage: () => void;
}

// Get saved RTSP URLs from localStorage
const getSavedUrls = (): Record<string, string> => {
    try {
        return JSON.parse(localStorage.getItem('race-vision-camera-urls') || '{}');
    } catch {
        return {};
    }
};

// Build PTZ cameras from config + saved URLs
const createPTZCameras = (): PTZCameraState[] => {
    const savedCamera = getSavedCamera();
    const savedUrls = getSavedUrls();
    return PTZ_CAMERAS.map(cam => ({
        ...cam,
        rtspUrl: savedUrls[cam.id] || cam.rtspUrl,
        isActive: cam.id === savedCamera,
    }));
};

// Build analytics cameras from config + saved URLs
const createAnalyticsCameras = (): AnalyticsCameraState[] => {
    const savedUrls = getSavedUrls();
    return ANALYTICS_CAMERAS.map(cam => ({
        ...cam,
        rtspUrl: savedUrls[cam.id] || cam.rtspUrl,
        horsesInView: [],
    }));
};

// Get saved camera from localStorage
const getSavedCamera = (): string => {
    return localStorage.getItem('activePTZCamera') || PTZ_CAMERAS[0]?.id || 'ptz-1';
};

export const useCameraStore = create<CameraState>((set, get) => ({
    ptzCameras: createPTZCameras(),
    analyticsCameras: createAnalyticsCameras(),
    activePTZCameraId: getSavedCamera(),
    streamMode: 'webrtc',

    setStreamMode: (mode) => set({ streamMode: mode }),

    setActivePTZCamera: (cameraId) => {
        localStorage.setItem('activePTZCamera', cameraId);
        set((state) => ({
            ptzCameras: state.ptzCameras.map(cam => ({
                ...cam,
                isActive: cam.id === cameraId
            })),
            activePTZCameraId: cameraId
        }));
        // Notify backend so /stream/ptz-live switches instantly
        const backendPort = 8000;
        const host = window.location.hostname || '127.0.0.1';
        const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
        fetch(`${protocol}//${host}:${backendPort}/api/ptz/active`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cameraId }),
        }).catch(() => { /* ignore network errors */ });
    },

    setActivePTZ: (cameraId) => {
        localStorage.setItem('activePTZCamera', cameraId);
        set((state) => ({
            ptzCameras: state.ptzCameras.map(cam => ({
                ...cam,
                isActive: cam.id === cameraId
            })),
            activePTZCameraId: cameraId
        }));
        // Notify backend so /stream/ptz-live switches instantly
        const backendPort = 8000;
        const host = window.location.hostname || '127.0.0.1';
        const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
        fetch(`${protocol}//${host}:${backendPort}/api/ptz/active`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cameraId }),
        }).catch(() => { /* ignore network errors */ });
    },

    updateCameraStatus: (cameraId, status) => set((state) => ({
        ptzCameras: state.ptzCameras.map(cam =>
            cam.id === cameraId ? { ...cam, status } : cam
        ),
        analyticsCameras: state.analyticsCameras.map(cam =>
            cam.id === cameraId ? { ...cam, status } : cam
        )
    })),

    batchUpdateCameraStatuses: (updates) => set((state) => {
        // Single set() call for all cameras — avoids 29 separate re-renders
        let ptzChanged = false;
        let analyticsChanged = false;
        const ptzCameras = state.ptzCameras.map(cam => {
            const newStatus = updates[cam.id];
            if (newStatus !== undefined && cam.status !== newStatus) {
                ptzChanged = true;
                return { ...cam, status: newStatus };
            }
            return cam;
        });
        const analyticsCameras = state.analyticsCameras.map(cam => {
            const newStatus = updates[cam.id];
            if (newStatus !== undefined && cam.status !== newStatus) {
                analyticsChanged = true;
                return { ...cam, status: newStatus };
            }
            return cam;
        });
        // Only return new arrays if something actually changed
        return {
            ...(ptzChanged ? { ptzCameras } : {}),
            ...(analyticsChanged ? { analyticsCameras } : {}),
        };
    }),

    updateAnalyticsCameraHorses: (cameraId, horseIds) => set((state) => ({
        analyticsCameras: state.analyticsCameras.map(cam =>
            cam.id === cameraId ? { ...cam, horsesInView: horseIds } : cam
        )
    })),

    batchUpdateCameraHorses: (updates) => set((state) => {
        let changed = false;
        const analyticsCameras = state.analyticsCameras.map(cam => {
            const newHorses = updates[cam.id];
            if (newHorses !== undefined) {
                const prev = cam.horsesInView;
                if (prev.length !== newHorses.length || prev.some((h, i) => h !== newHorses[i])) {
                    changed = true;
                    return { ...cam, horsesInView: newHorses };
                }
            }
            return cam;
        });
        return changed ? { analyticsCameras } : {};
    }),

    initializeCameras: () => set({
        ptzCameras: createPTZCameras(),
        analyticsCameras: createAnalyticsCameras(),
        activePTZCameraId: getSavedCamera()
    }),

    syncFromStorage: () => {
        const savedCamera = getSavedCamera();
        const currentCamera = get().activePTZCameraId;

        if (savedCamera !== currentCamera) {
            set((state) => ({
                ptzCameras: state.ptzCameras.map(cam => ({
                    ...cam,
                    isActive: cam.id === savedCamera
                })),
                activePTZCameraId: savedCamera
            }));
        }
    }
}));

// Listen for camera changes from other tabs
if (typeof window !== 'undefined') {
    window.addEventListener('storage', (event) => {
        if (event.key === 'activePTZCamera' && event.newValue) {
            useCameraStore.getState().syncFromStorage();
        }
    });
}
