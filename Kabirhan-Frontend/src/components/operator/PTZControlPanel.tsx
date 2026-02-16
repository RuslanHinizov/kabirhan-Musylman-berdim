import { Check, Wifi } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useCameraStore } from '../../store/cameraStore';
import { MJPEGPlayer } from '../MJPEGPlayer';
import { BACKEND_HTTP_URL } from '../../config/backend';

export const PTZControlPanel = () => {
    const { t } = useTranslation();
    const { ptzCameras, activePTZCameraId, setActivePTZCamera } = useCameraStore();

    return (
        <div className="p-6 h-full">
            <div className="mb-6">
                <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-1">{t('ptzPanel.title')}</h2>
                <p className="text-sm text-[var(--text-muted)]">{t('ptzPanel.description')}</p>
            </div>

            {/* 2x2 layout for 4 PTZ cameras */}
            <div className="grid grid-cols-2 gap-4 h-[calc(100%-100px)] overflow-y-auto pr-1">
                {ptzCameras.map((camera) => {
                    const isActive = camera.id === activePTZCameraId;

                    return (
                        <button
                            key={camera.id}
                            onClick={() => setActivePTZCamera(camera.id)}
                            className={`
                relative rounded-lg cursor-pointer transition-all overflow-hidden aspect-video
                ${isActive
                                    ? 'ring-2 ring-inset ring-[var(--primary)] border border-[var(--primary)]'
                                    : 'bg-[var(--surface)] border border-[var(--border)] hover:border-[var(--text-muted)]'}
              `}
                        >
                            {/* Video Preview */}
                            <div className="absolute inset-0">
                                <MJPEGPlayer
                                    url={`${BACKEND_HTTP_URL}/stream/${camera.id}`}
                                    cameraName={camera.name}
                                    className="w-full h-full"
                                    objectFit="contain"
                                    showStatusOverlay={false}
                                    showInfoBadge={false}
                                />
                            </div>

                            {/* Label */}
                            <div className="absolute top-4 left-4 flex items-center gap-2 z-10">
                                <span className={`
                  px-3 py-1.5 rounded text-sm font-medium
                  ${isActive ? 'bg-[var(--primary)] text-white' : 'bg-black/70 text-white'}
                `}>
                                    {camera.name}
                                </span>

                                {isActive && <span className="live-badge">{t('ptzPanel.onAir')}</span>}
                            </div>

                            {/* Status */}
                            <div className="absolute top-4 right-4 flex items-center gap-1.5 z-10">
                                <div className="bg-black/70 rounded px-2 py-1 flex items-center gap-1.5">
                                    <Wifi className={`w-4 h-4 ${camera.status === 'online' ? 'text-[var(--accent)]' : 'text-[var(--danger)]'}`} />
                                    <span className="text-xs text-white">
                                        {camera.status === 'online' ? t('ptzPanel.online') : t('ptzPanel.offline')}
                                    </span>
                                </div>
                            </div>

                            {/* Active Check */}
                            {isActive && (
                                <div className="absolute bottom-4 right-4 w-8 h-8 bg-[var(--primary)] rounded-full flex items-center justify-center z-10">
                                    <Check className="w-5 h-5 text-white" strokeWidth={2.5} />
                                </div>
                            )}

                            {/* Camera position info */}
                            <div className="absolute bottom-4 left-4 text-xs bg-black/70 text-white px-2 py-1 rounded z-10">
                                {t('ptzPanel.position')}: {camera.position}m
                            </div>
                        </button>
                    );
                })}
            </div>

            <p className="mt-4 text-sm text-[var(--text-muted)]">
                {t('ptzPanel.keyboardTip', { count: ptzCameras.length })}
            </p>
        </div>
    );
};
