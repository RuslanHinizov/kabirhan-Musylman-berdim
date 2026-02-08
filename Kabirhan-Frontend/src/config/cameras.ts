// Camera Configuration for Race Vision
// Two separate configs: Analytics (backend detection) and PTZ (public display)

// ============================================================
// TYPES
// ============================================================

export type AnalyticsCameraConfig = {
    id: string;
    name: string;
    rtspUrl: string;
    trackStart: number;   // meters — start of track segment this camera covers
    trackEnd: number;     // meters — end of track segment
    status: 'online' | 'offline';
};

export type PTZCameraConfig = {
    id: string;
    name: string;
    rtspUrl: string;
    mjpegUrl: string;     // MJPEG stream URL for public display
    position: number;     // meters — position on track
    status: 'online' | 'offline';
};

// ============================================================
// BACKEND URL
// ============================================================

const BACKEND_URL = 'http://localhost:8000';

// ============================================================
// ANALYTICS CAMERAS — processed by backend (YOLO + CNN detection)
// Viewers NEVER see these streams. Only the detection results
// (jockey positions, rankings) are sent to the frontend.
// ============================================================

export const ANALYTICS_CAMERAS: AnalyticsCameraConfig[] = [
    {
        id: 'analytics-1',
        name: 'Analytics Camera 1',
        rtspUrl: 'rtsp://admin:password@192.168.1.101:554/stream',
        trackStart: 0,
        trackEnd: 100,
        status: 'online',
    },
    {
        id: 'analytics-2',
        name: 'Analytics Camera 2',
        rtspUrl: 'rtsp://admin:password@192.168.1.102:554/stream',
        trackStart: 100,
        trackEnd: 200,
        status: 'online',
    },
    {
        id: 'analytics-3',
        name: 'Analytics Camera 3',
        rtspUrl: 'rtsp://admin:password@192.168.1.103:554/stream',
        trackStart: 200,
        trackEnd: 300,
        status: 'online',
    },
];

// ============================================================
// PTZ CAMERAS — shown to viewers on public display
// These provide the broadcast-quality live feed.
// Operator can switch between them.
// ============================================================

export const PTZ_CAMERAS: PTZCameraConfig[] = [
    {
        id: 'ptz-1',
        name: 'PTZ Camera 1',
        rtspUrl: 'rtsp://admin:password@192.168.1.201:554/stream',
        mjpegUrl: '',   // Set real PTZ stream URL when camera is connected
        position: 0,
        status: 'offline',
    },
    {
        id: 'ptz-2',
        name: 'PTZ Camera 2',
        rtspUrl: 'rtsp://admin:password@192.168.1.202:554/stream',
        mjpegUrl: '',
        position: 100,
        status: 'offline',
    },
    {
        id: 'ptz-3',
        name: 'PTZ Camera 3',
        rtspUrl: 'rtsp://admin:password@192.168.1.203:554/stream',
        mjpegUrl: '',
        position: 200,
        status: 'offline',
    },
];
