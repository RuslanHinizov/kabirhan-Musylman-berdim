import { Camera, Wifi, Maximize } from 'lucide-react';
import { useCameraStore } from '../../store/cameraStore';
import { useRaceStore } from '../../store/raceStore';
import { MJPEGPlayer } from '../MJPEGPlayer';

export const PTZCameraDisplay = () => {
    const { ptzCameras, activePTZCameraId } = useCameraStore();
    const { race } = useRaceStore();
    const activeCamera = ptzCameras.find(cam => cam.id === activePTZCameraId);

    return (
        <div className="relative w-full h-full bg-[var(--background)]">
            {/* Camera Feed — use PTZ camera's mjpegUrl */}
            {activeCamera?.mjpegUrl ? (
                <MJPEGPlayer
                    url={activeCamera.mjpegUrl}
                    cameraName={activeCamera.name}
                    className="absolute inset-0 w-full h-full"
                />
            ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                        <Camera className="w-16 h-16 text-[var(--border)] mx-auto mb-4" strokeWidth={1} />
                        <p className="text-sm text-[var(--text-muted)]">No Camera Feed</p>
                    </div>
                </div>
            )}

            {/* Top Left */}
            <div className="absolute top-4 left-4 flex items-center gap-3 z-10">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--surface)] rounded-lg border border-[var(--border)]">
                    <Camera className="w-4 h-4 text-[var(--text-muted)]" />
                    <span className="text-sm font-medium text-[var(--text-primary)]">
                        {activeCamera?.name || 'PTZ 1'}
                    </span>
                </div>

                <div className="live-badge">LIVE</div>
            </div>

            {/* Top Right */}
            <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
                <div className="flex items-center gap-1.5 px-2.5 py-1.5 bg-[var(--surface)] rounded-lg border border-[var(--border)]">
                    <Wifi className="w-3.5 h-3.5 text-[var(--accent)]" />
                    <span className="text-xs text-[var(--text-secondary)]">HD</span>
                </div>

                <button className="p-1.5 bg-[var(--surface)] rounded-lg border border-[var(--border)] hover:border-[var(--text-muted)] transition-colors cursor-pointer">
                    <Maximize className="w-4 h-4 text-[var(--text-muted)]" />
                </button>
            </div>

            {/* Bottom Left */}
            <div className="absolute bottom-4 left-4 z-10">
                <div className="px-4 py-2.5 bg-[var(--surface)] rounded-lg border border-[var(--border)]">
                    <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-0.5">Race</p>
                    <p className="text-sm font-medium text-[var(--text-primary)]">
                        {race.name || 'Grand Championship'}
                    </p>
                </div>
            </div>

            {/* Bottom Right - Camera Selector */}
            <div className="absolute bottom-4 right-4 flex gap-1 z-10">
                {ptzCameras.map((cam, i) => (
                    <div
                        key={cam.id}
                        className={`
              w-8 h-6 rounded flex items-center justify-center text-xs font-medium cursor-pointer transition-colors
              ${cam.id === activePTZCameraId
                                ? 'bg-[var(--primary)] text-white'
                                : 'bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:text-[var(--text-secondary)]'}
            `}
                    >
                        {i + 1}
                    </div>
                ))}
            </div>
        </div>
    );
};
