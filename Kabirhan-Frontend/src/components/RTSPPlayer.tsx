import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';

interface CameraInput {
    name: string;
    hlsUrl?: string;
    position?: number;
}

interface RTSPPlayerProps {
    camera: CameraInput;
    className?: string;
    showControls?: boolean;
    muted?: boolean;
    autoPlay?: boolean;
}

type StreamStatus = 'connecting' | 'online' | 'offline' | 'error';

// Helper to get initial status
const getInitialStatus = (hlsUrl: string | undefined): StreamStatus =>
    hlsUrl ? 'connecting' : 'error';

const getInitialError = (hlsUrl: string | undefined): string =>
    hlsUrl ? '' : 'No HLS stream URL configured';

export const RTSPPlayer = ({
    camera,
    className = '',
    showControls = false,
    muted = true,
    autoPlay = true
}: RTSPPlayerProps) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const hlsRef = useRef<Hls | null>(null);

    // Use key to force re-mount when camera changes
    const [key, setKey] = useState(0);
    const [status, setStatus] = useState<StreamStatus>(() => getInitialStatus(camera.hlsUrl));
    const [errorMessage, setErrorMessage] = useState<string>(() => getInitialError(camera.hlsUrl));

    // Track camera URL changes
    const prevUrlRef = useRef(camera.hlsUrl);

    if (prevUrlRef.current !== camera.hlsUrl) {
        prevUrlRef.current = camera.hlsUrl;
        setKey(k => k + 1);
        setStatus(getInitialStatus(camera.hlsUrl));
        setErrorMessage(getInitialError(camera.hlsUrl));
    }

    useEffect(() => {
        const video = videoRef.current;

        // Clean up previous HLS instance
        if (hlsRef.current) {
            hlsRef.current.destroy();
            hlsRef.current = null;
        }

        if (!video || !camera.hlsUrl) return;

        // Check if browser supports HLS natively (Safari)
        if (video.canPlayType('application/vnd.apple.mpegurl')) {
            video.src = camera.hlsUrl;

            const handleLoad = () => {
                setStatus('online');
                if (autoPlay) video.play();
            };

            const handleError = () => {
                setStatus('offline');
                setErrorMessage('Failed to load stream');
            };

            video.addEventListener('loadedmetadata', handleLoad);
            video.addEventListener('error', handleError);

            return () => {
                video.removeEventListener('loadedmetadata', handleLoad);
                video.removeEventListener('error', handleError);
            };
        }

        // Use HLS.js for other browsers
        if (Hls.isSupported()) {
            const hls = new Hls({
                enableWorker: true,
                lowLatencyMode: true,
                backBufferLength: 90,
                liveSyncDurationCount: 3,
                liveMaxLatencyDurationCount: 10,
                liveDurationInfinity: true,
                highBufferWatchdogPeriod: 1,
            });

            hlsRef.current = hls;

            hls.loadSource(camera.hlsUrl);
            hls.attachMedia(video);

            hls.on(Hls.Events.MANIFEST_PARSED, () => {
                setStatus('online');
                if (autoPlay) {
                    video.play().catch(err => {
                        console.log('Autoplay prevented:', err);
                    });
                }
            });

            hls.on(Hls.Events.ERROR, (_event, data) => {
                if (data.fatal) {
                    switch (data.type) {
                        case Hls.ErrorTypes.NETWORK_ERROR:
                            setStatus('offline');
                            setErrorMessage('Network error - retrying...');
                            setTimeout(() => hls.startLoad(), 3000);
                            break;
                        case Hls.ErrorTypes.MEDIA_ERROR:
                            setErrorMessage('Media error - recovering...');
                            hls.recoverMediaError();
                            break;
                        default:
                            setStatus('error');
                            setErrorMessage('Fatal error occurred');
                            hls.destroy();
                            break;
                    }
                }
            });

            return () => {
                hls.destroy();
            };
        }

        setStatus('error');
        setErrorMessage('HLS not supported in this browser');
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [key, autoPlay]);

    return (
        <div className={`relative bg-black overflow-hidden ${className}`}>
            {/* Video Element */}
            <video
                ref={videoRef}
                className="w-full h-full object-cover"
                muted={muted}
                playsInline
                controls={showControls}
            />

            {/* Status Overlay */}
            {status !== 'online' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80">
                    {status === 'connecting' && (
                        <>
                            <Loader2 className="w-12 h-12 text-blue-400 animate-spin mb-4" />
                            <p className="text-white text-sm">Connecting to stream...</p>
                            <p className="text-gray-500 text-xs mt-1">{camera.name}</p>
                        </>
                    )}

                    {status === 'offline' && (
                        <>
                            <WifiOff className="w-12 h-12 text-red-400 mb-4" />
                            <p className="text-white text-sm">Stream Offline</p>
                            <p className="text-gray-500 text-xs mt-1">{camera.name}</p>
                            <p className="text-gray-600 text-xs mt-2">{errorMessage}</p>
                        </>
                    )}

                    {status === 'error' && (
                        <>
                            <WifiOff className="w-12 h-12 text-yellow-400 mb-4" />
                            <p className="text-white text-sm">Connection Error</p>
                            <p className="text-gray-500 text-xs mt-1">{camera.name}</p>
                            <p className="text-red-400 text-xs mt-2">{errorMessage}</p>
                        </>
                    )}
                </div>
            )}

            {/* Camera Info Badge */}
            <div className="absolute top-2 left-2 flex items-center gap-2">
                <div className={`
                    flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium
                    ${status === 'online' ? 'bg-green-600' :
                        status === 'connecting' ? 'bg-blue-600' : 'bg-red-600'}
                    text-white
                `}>
                    <Wifi className="w-3 h-3" />
                    {camera.name}
                </div>
            </div>

            {/* Position Badge */}
            <div className="absolute top-2 right-2 px-2 py-1 bg-black/60 rounded text-xs text-white">
                {camera.position}m
            </div>
        </div>
    );
};
