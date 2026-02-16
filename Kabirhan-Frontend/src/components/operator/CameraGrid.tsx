import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Video, VideoOff, Maximize2, Minimize2, X } from 'lucide-react';
import { useCameraStore } from '../../store/cameraStore';
import { useRaceStore } from '../../store/raceStore';
import { SnapshotPlayer } from '../SnapshotPlayer';
import { MJPEGPlayer } from '../MJPEGPlayer';
import { TRACK_LENGTH } from '../../types';
import { BACKEND_HTTP_URL } from '../../config/backend';

export const CameraGrid = () => {
    const { t } = useTranslation();
    const { analyticsCameras } = useCameraStore();
    const { rankings } = useRaceStore();
    const [expandedCamera, setExpandedCamera] = useState<string | null>(null);
    const [streamStates, setStreamStates] = useState<Record<string, string>>({});

    const fetchStreamStates = useCallback(async () => {
        try {
            const res = await fetch(`${BACKEND_HTTP_URL}/api/streams/status`);
            const status = await res.json() as Record<string, { state?: string }>;
            const next: Record<string, string> = {};
            Object.entries(status).forEach(([camId, info]) => {
                next[camId] = info?.state || 'idle';
            });
            setStreamStates(next);
        } catch {
            // keep previous states on transient network issues
        }
    }, []);

    useEffect(() => {
        const initial = setTimeout(() => {
            void fetchStreamStates();
        }, 0);
        const interval = setInterval(() => {
            void fetchStreamStates();
        }, 5000);
        return () => {
            clearTimeout(initial);
            clearInterval(interval);
        };
    }, [fetchStreamStates]);

    const toggleExpand = (cameraId: string) => {
        setExpandedCamera(prev => prev === cameraId ? null : cameraId);
    };

    const visibleGridCameras = analyticsCameras;
    // Keep snapshot cadence stable to avoid remount/reconnect storms when
    // toggling expanded camera view.
    const gridIntervalMs = 1200;
    const runningAnalyticsCount = Object.entries(streamStates).filter(
        ([camId, s]) => camId.startsWith('analytics-') && s === 'running'
    ).length;
    const livePreviewMode = runningAnalyticsCount > 0 && runningAnalyticsCount <= 4;

    return (
        <div className="p-6 h-full flex flex-col overflow-hidden">
            {/* Header */}
            <div className="mb-4 flex-shrink-0">
                <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-1">{t('cameraGrid.analyticsCameras')}</h2>
                <p className="text-sm text-[var(--text-muted)]">
                    {analyticsCameras.length} {t('cameraGrid.cameraCount')}
                </p>
            </div>

            {/* Expanded Camera View */}
            {expandedCamera && (() => {
                const index = analyticsCameras.findIndex(c => c.id === expandedCamera);
                if (index === -1) return null;
                const camera = analyticsCameras[index];
                const horsesNear = rankings.filter(horse => {
                    const horsePos = horse.distanceCovered % TRACK_LENGTH;
                    return horsePos >= camera.trackStart && horsePos < camera.trackEnd;
                });

                return (
                    <div className="mb-4 flex-shrink-0 rounded-xl overflow-hidden border-2 border-blue-500 shadow-lg shadow-blue-500/20">
                        <div className="relative bg-black" style={{ height: '50vh' }}>
                            <MJPEGPlayer
                                url={`${BACKEND_HTTP_URL}/stream/${camera.id}`}
                                cameraName={camera.name}
                                className="w-full h-full"
                                objectFit="contain"
                            />
                            {/* Top bar overlay */}
                            <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-2 bg-gradient-to-b from-black/70 to-transparent">
                                <div className="flex items-center gap-3">
                                    <span className="text-white font-bold text-lg">{camera.name}</span>
                                    <span className="text-blue-300 text-sm">{camera.trackStart}-{camera.trackEnd}m</span>
                                    {horsesNear.length > 0 && (
                                        <span className="text-xs bg-amber-500/30 text-amber-300 px-2 py-0.5 rounded-full font-medium">
                                            {horsesNear.length} {horsesNear.length > 1 ? t('cameraGrid.horsesDetected') : t('cameraGrid.horseDetected')}
                                        </span>
                                    )}
                                </div>
                                <button
                                    onClick={() => setExpandedCamera(null)}
                                    className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>
                        </div>
                    </div>
                );
            })()}

            {/* Camera Grid - 5 cameras side by side for better density with 25 cameras */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3 flex-1 overflow-y-auto p-1">
                {visibleGridCameras.map((camera) => {
                    const state = streamStates[camera.id] || 'idle';
                    const isOnline = state === 'running' || state === 'connecting';
                    const isExpanded = expandedCamera === camera.id;
                    const useLivePreview = livePreviewMode && state === 'running' && !isExpanded;

                    // Find horses within this camera's track segment
                    const horsesNear = rankings.filter(horse => {
                        const horsePos = horse.distanceCovered % TRACK_LENGTH;
                        return horsePos >= camera.trackStart && horsePos < camera.trackEnd;
                    });
                    const hasHorses = horsesNear.length > 0;

                    return (
                        <div
                            key={camera.id}
                            className={`
                                relative rounded-xl overflow-hidden transition-all duration-300 cursor-pointer group
                                ${isExpanded
                                    ? 'ring-2 ring-blue-500 shadow-lg shadow-blue-500/20'
                                    : hasHorses
                                        ? 'ring-2 ring-amber-500 shadow-lg shadow-amber-500/20'
                                        : 'border border-[var(--border)]'}
                            `}
                            onClick={() => toggleExpand(camera.id)}
                        >
                            {/* Detection preview:
                               - Live MJPEG when only a few analytics cameras are active
                               - Snapshot polling fallback for high camera counts */}
                            <div className="aspect-video bg-black relative">
                                {useLivePreview ? (
                                    <MJPEGPlayer
                                        url={`${BACKEND_HTTP_URL}/stream/${camera.id}`}
                                        cameraName={camera.name}
                                        className="w-full h-full"
                                    />
                                ) : (
                                    <SnapshotPlayer
                                        url={`${BACKEND_HTTP_URL}/snapshot/${camera.id}`}
                                        cameraName={camera.name}
                                        className="w-full h-full"
                                        intervalMs={gridIntervalMs}
                                    />
                                )}
                                {/* Expand/Collapse icon on hover */}
                                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <div className="p-1 rounded bg-black/60 text-white">
                                        {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                                    </div>
                                </div>
                            </div>

                            {/* Bottom Info Bar */}
                            <div className="p-3 bg-[var(--surface)]">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        {isOnline ? (
                                            <Video className={`w-4 h-4 ${hasHorses ? 'text-amber-400' : 'text-[var(--text-muted)]'}`} />
                                        ) : (
                                            <VideoOff className="w-4 h-4 text-[var(--danger)]" />
                                        )}
                                        <span className={`text-sm font-semibold ${hasHorses ? 'text-amber-400' : 'text-[var(--text-primary)]'}`}>
                                            {camera.name}
                                        </span>
                                    </div>

                                    <div className="flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-red-500'}`} />
                                        <span className="text-xs text-[var(--text-muted)]">
                                            {camera.trackStart}-{camera.trackEnd}m
                                        </span>
                                    </div>
                                </div>

                                {/* Horse detection badge */}
                                {hasHorses && (
                                    <div className="mt-2 flex items-center gap-1">
                                        <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full font-medium">
                                            {horsesNear.length} {horsesNear.length > 1 ? t('cameraGrid.horsesDetected') : t('cameraGrid.horseDetected')}
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Legend */}
            <div className="mt-4 flex items-center justify-center gap-8 text-xs text-[var(--text-muted)] flex-shrink-0 pt-3 border-t border-[var(--border)]">
                <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                    <span>{t('cameraGrid.online')}</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                    <span>{t('cameraGrid.offline')}</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded border-2 border-amber-500" />
                    <span>{t('cameraGrid.horseDetectedLegend')}</span>
                </div>
            </div>
        </div>
    );
};

