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
    position: number;     // meters — position on track
    status: 'online' | 'offline';
};

// ============================================================
// ANALYTICS CAMERAS — 25 cameras for 2500m track (100m each)
// IP range: 10.223.70.20 - 10.223.70.44
// Viewers NEVER see these streams. Only the detection results
// (jockey positions, rankings) are sent to the frontend.
// ============================================================

// Generate 25 analytics cameras dynamically
const generateAnalyticsCameras = (): AnalyticsCameraConfig[] => {
    const cameras: AnalyticsCameraConfig[] = [];

    for (let i = 0; i < 25; i++) {
        cameras.push({
            id: `analytics-${i + 1}`,
            name: `Camera ${i + 1}`,
            rtspUrl: '',
            trackStart: i * 100,
            trackEnd: (i + 1) * 100,
            status: 'offline',
        });
    }

    return cameras;
};

export const ANALYTICS_CAMERAS: AnalyticsCameraConfig[] = generateAnalyticsCameras();

// ============================================================
// PTZ CAMERAS — shown to viewers on public display
// These provide the broadcast-quality live feed.
// Operator can switch between them.
// ============================================================

export const PTZ_CAMERAS: PTZCameraConfig[] = [
    {
        id: 'ptz-1',
        name: 'PTZ Camera 1',
        rtspUrl: '',
        position: 0,
        status: 'offline',
    },
    {
        id: 'ptz-2',
        name: 'PTZ Camera 2',
        rtspUrl: '',
        position: 833,  // ~1/3 of track
        status: 'offline',
    },
    {
        id: 'ptz-3',
        name: 'PTZ Camera 3',
        rtspUrl: '',
        position: 1667,  // ~2/3 of track
        status: 'offline',
    },
    {
        id: 'ptz-4',
        name: 'PTZ Camera 4',
        rtspUrl: '',
        position: 2500,  // End of track
        status: 'offline',
    },
];
