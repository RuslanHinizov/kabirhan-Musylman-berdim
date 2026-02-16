import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Camera, Play, Square, Save, RefreshCw } from 'lucide-react';
import { PTZ_CAMERAS, ANALYTICS_CAMERAS } from '../../config/cameras';
import { BACKEND_HTTP_URL } from '../../config/backend';

// Local type for camera row data
type CameraData = {
    id: string;
    name: string;
    type: 'ptz' | 'analytics';
    rtspUrl: string;
    position?: number;      // PTZ cameras
    trackStart?: number;    // Analytics cameras
    trackEnd?: number;      // Analytics cameras
    status: 'online' | 'offline' | 'connecting';
};

// Separate component to prevent re-render issues with input state
const CameraRow = ({
    camera,
    isActive,
    isLoading,
    onStartStream,
    onStopStream,
    onSave
}: {
    camera: CameraData;
    isActive: boolean;
    isLoading: boolean;
    onStartStream: (id: string) => void;
    onStopStream: (id: string) => void;
    onSave: (id: string, url: string, type: 'ptz' | 'analytics') => void;
}) => {
    const { t } = useTranslation();
    const [localUrl, setLocalUrl] = useState(camera.rtspUrl);
    const [isEditing, setIsEditing] = useState(false);

    const handleSave = () => {
        onSave(camera.id, localUrl, camera.type);
        setIsEditing(false);
    };

    const posLabel = camera.type === 'analytics'
        ? `${camera.trackStart}–${camera.trackEnd}m`
        : `${camera.position}m`;

    return (
        <div className="flex items-center gap-3 p-3 rounded-lg bg-[var(--background)] border border-[var(--border)] hover:border-[var(--text-muted)] transition-colors">
            <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${isActive ? 'bg-[var(--accent)]' : 'bg-[var(--danger)]'}`} />

            <div className="w-28 flex-shrink-0">
                <span className="text-sm font-medium text-[var(--text-primary)]">{camera.name}</span>
            </div>

            <div className="w-20 flex-shrink-0 text-xs text-[var(--text-muted)]">
                {posLabel}
            </div>

            <div className="flex-1 min-w-0">
                <input
                    type="text"
                    value={localUrl}
                    onChange={(e) => {
                        setLocalUrl(e.target.value);
                        setIsEditing(true);
                    }}
                    className="w-full text-xs py-1.5 font-mono bg-[#1a1a2e] border border-[#333] rounded px-2 text-white focus:border-blue-500 focus:outline-none"
                    placeholder={t('cameraSettings.rtspPlaceholder')}
                />
            </div>

            <div className="flex items-center gap-1 flex-shrink-0">
                {isEditing && (
                    <button
                        onClick={handleSave}
                        className="p-2 text-[var(--accent)] hover:bg-[var(--accent)]/10 rounded transition-colors cursor-pointer"
                        title={t('cameraSettings.save')}
                    >
                        <Save className="w-4 h-4" />
                    </button>
                )}

                {isActive ? (
                    <button
                        onClick={() => onStopStream(camera.id)}
                        disabled={isLoading}
                        className="p-2 text-[var(--danger)] hover:bg-[var(--danger)]/10 rounded transition-colors cursor-pointer disabled:opacity-50"
                        title={t('cameraSettings.stopStream')}
                    >
                        {isLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Square className="w-4 h-4" />}
                    </button>
                ) : (
                    <button
                        onClick={() => onStartStream(camera.id)}
                        disabled={isLoading}
                        className="p-2 text-[var(--accent)] hover:bg-[var(--accent)]/10 rounded transition-colors cursor-pointer disabled:opacity-50"
                        title={t('cameraSettings.startStream')}
                    >
                        {isLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                    </button>
                )}
            </div>
        </div>
    );
};

const STORAGE_KEY = 'race-vision-camera-urls';
const SYNC_ONCE_KEY = 'race-vision-camera-sync-done-v1';

const getSavedUrls = (): Record<string, string> => {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    } catch {
        return {};
    }
};

export const CameraSettings = () => {
    const { t } = useTranslation();
    const savedUrls = getSavedUrls();
    const [ptzCameras, setPtzCameras] = useState<CameraData[]>(
        PTZ_CAMERAS.map(c => ({ ...c, type: 'ptz' as const, status: c.status, rtspUrl: savedUrls[c.id] || c.rtspUrl }))
    );
    const [analyticsCameras, setAnalyticsCameras] = useState<CameraData[]>(
        ANALYTICS_CAMERAS.map(c => ({ ...c, type: 'analytics' as const, status: c.status, rtspUrl: savedUrls[c.id] || c.rtspUrl }))
    );
    const [streamStatus, setStreamStatus] = useState<Record<string, boolean>>({});
    const [loading, setLoading] = useState<Record<string, boolean>>({});

    // Fetch raw stream states from backend: { camId: "running" | "connecting" | ... }
    const fetchStreamStates = useCallback(async (): Promise<Record<string, string>> => {
        try {
            const response = await fetch(`${BACKEND_HTTP_URL}/api/streams/status`);
            const status = await response.json() as Record<string, { state?: string }>;
            const states: Record<string, string> = {};
            Object.keys(status).forEach(id => {
                states[id] = status[id]?.state || 'idle';
            });
            return states;
        } catch {
            return {};
        }
    }, []);

    // Fetch stream status
    const fetchStreamStatus = useCallback(async () => {
        const states = await fetchStreamStates();
        const activeIds: Record<string, boolean> = {};
        Object.entries(states).forEach(([id, state]) => {
            // Treat connecting as active to avoid duplicate start clicks/restarts.
            if (state === 'running' || state === 'connecting') {
                activeIds[id] = true;
            }
        });
        setStreamStatus(activeIds);
    }, [fetchStreamStates]);

    // On mount: always sync saved URLs to backend.
    // Sync once per tab session to avoid repetitive PUT storms in logs.
    useEffect(() => {
        const syncUrlsToBackend = async () => {
            const saved = getSavedUrls();
            const promises = Object.entries(saved).map(([camId, url]) => {
                if (!url) return Promise.resolve();
                return fetch(`${BACKEND_HTTP_URL}/api/cameras/${camId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rtspUrl: url })
                }).catch(() => {}); // silent fail
            });
            await Promise.all(promises);
        };

        const initialize = async () => {
            const syncDone = sessionStorage.getItem(SYNC_ONCE_KEY) === '1';
            if (!syncDone) {
                await syncUrlsToBackend();
                sessionStorage.setItem(SYNC_ONCE_KEY, '1');
            }

            await fetchStreamStatus();
        };

        initialize();
        const interval = setInterval(fetchStreamStatus, 10000);
        return () => clearInterval(interval);
    }, [fetchStreamStates, fetchStreamStatus]);

    // Start stream — ensures URL is sent to backend first, then starts
    const startStream = useCallback(async (cameraId: string) => {
        setLoading(prev => ({ ...prev, [cameraId]: true }));
        try {
            // Ensure backend has the URL (may have been lost on restart)
            const saved = getSavedUrls();
            const url = saved[cameraId];
            if (url) {
                await fetch(`${BACKEND_HTTP_URL}/api/cameras/${cameraId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rtspUrl: url })
                });
            }
            await fetch(`${BACKEND_HTTP_URL}/api/cameras/${cameraId}/start`, { method: 'POST' });
            await fetchStreamStatus();
        } catch (error) {
            console.error('Failed to start stream:', error);
        }
        setLoading(prev => ({ ...prev, [cameraId]: false }));
    }, [fetchStreamStatus]);

    // Stop stream
    const stopStream = useCallback(async (cameraId: string) => {
        setLoading(prev => ({ ...prev, [cameraId]: true }));
        try {
            await fetch(`${BACKEND_HTTP_URL}/api/cameras/${cameraId}/stop`, { method: 'POST' });
            await fetchStreamStatus();
        } catch (error) {
            console.error('Failed to stop stream:', error);
        }
        setLoading(prev => ({ ...prev, [cameraId]: false }));
    }, [fetchStreamStatus]);

    // Save camera URL
    const saveCamera = useCallback(async (cameraId: string, url: string, type: 'ptz' | 'analytics') => {
        // Persist to localStorage
        try {
            const saved = getSavedUrls();
            saved[cameraId] = url;
            localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
        } catch {
            // localStorage may fail in private browsing
        }

        // Send to backend
        try {
            await fetch(`${BACKEND_HTTP_URL}/api/cameras/${cameraId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rtspUrl: url })
            });
        } catch {
            // Continue even if server not available - saved locally
        }

        if (type === 'ptz') {
            setPtzCameras(prev => prev.map(c => c.id === cameraId ? { ...c, rtspUrl: url } : c));
        } else {
            setAnalyticsCameras(prev => prev.map(c => c.id === cameraId ? { ...c, rtspUrl: url } : c));
        }
    }, []);

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="mb-6">
                <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-1">{t('cameraSettings.title')}</h2>
                <p className="text-sm text-[var(--text-muted)]">{t('cameraSettings.description')}</p>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="racing-card p-5">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-medium text-[var(--text-primary)] flex items-center gap-2">
                            <Camera className="w-4 h-4 text-[var(--primary)]" />
                            {t('cameraSettings.ptzBroadcast')}
                        </h3>
                        <span className="text-xs text-[var(--text-muted)]">
                            {Object.keys(streamStatus).filter(id => id.startsWith('ptz')).length} / {ptzCameras.length} {t('cameraSettings.active')}
                        </span>
                    </div>

                    <div className="space-y-2">
                        {ptzCameras.map(camera => (
                            <CameraRow
                                key={camera.id}
                                camera={camera}
                                isActive={!!streamStatus[camera.id]}
                                isLoading={!!loading[camera.id]}
                                onStartStream={startStream}
                                onStopStream={stopStream}
                                onSave={saveCamera}
                            />
                        ))}
                    </div>
                </div>

                <div className="racing-card p-5">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-medium text-[var(--text-primary)] flex items-center gap-2">
                            <Camera className="w-4 h-4 text-[var(--warning)]" />
                            {t('cameraSettings.analyticsDetection')}
                        </h3>
                        <span className="text-xs text-[var(--text-muted)]">
                            {Object.keys(streamStatus).filter(id => id.startsWith('analytics')).length} / {analyticsCameras.length} {t('cameraSettings.active')}
                        </span>
                    </div>

                    <div className="space-y-2">
                        {analyticsCameras.map(camera => (
                            <CameraRow
                                key={camera.id}
                                camera={camera}
                                isActive={!!streamStatus[camera.id]}
                                isLoading={!!loading[camera.id]}
                                onStartStream={startStream}
                                onStopStream={stopStream}
                                onSave={saveCamera}
                            />
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-6 racing-card p-4">
                <h4 className="text-sm font-medium text-[var(--text-primary)] mb-2">{t('cameraSettings.backendServer')}</h4>
                <div className="text-xs text-[var(--text-muted)] space-y-1">
                    <p>Backend: <code className="text-[var(--text-secondary)]">{BACKEND_HTTP_URL}</code></p>
                    <p>Run: <code className="text-[var(--text-secondary)]">python api/server.py --video data/videos/exp10_cam1.mp4 --auto-start</code></p>
                </div>
            </div>
        </div>
    );
};
