import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Loader2, Wifi, WifiOff } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface SnapshotPlayerProps {
    url: string;
    cameraName?: string;
    className?: string;
    objectFit?: 'cover' | 'contain';
    intervalMs?: number;
}

type StreamStatus = 'connecting' | 'online' | 'offline';

const withCacheBust = (url: string): string => {
    const sep = url.includes('?') ? '&' : '?';
    return `${url}${sep}_rv=${Date.now()}`;
};

const SnapshotPlayerContent = ({
    url,
    cameraName = 'Camera',
    className = '',
    objectFit = 'cover',
    intervalMs = 1200,
}: SnapshotPlayerProps) => {
    const { t } = useTranslation();
    const [status, setStatus] = useState<StreamStatus>('connecting');
    const [src, setSrc] = useState(() => withCacheBust(url));
    const inFlightRef = useRef(true);
    const watchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const hasEverLoadedRef = useRef(false);
    const consecutiveErrorsRef = useRef(0);

    const clearWatchdog = useCallback(() => {
        if (watchdogRef.current) {
            clearTimeout(watchdogRef.current);
            watchdogRef.current = null;
        }
    }, []);

    const armWatchdog = useCallback(() => {
        clearWatchdog();
        // Only use watchdog for initial connection — once loaded, skip it
        if (hasEverLoadedRef.current) return;
        const timeoutMs = Math.max(intervalMs * 3, 8000);
        watchdogRef.current = setTimeout(() => {
            inFlightRef.current = false;
            setStatus('offline');
        }, timeoutMs);
    }, [clearWatchdog, intervalMs]);

    const requestNextSnapshot = useCallback(() => {
        if (typeof document !== 'undefined' && document.hidden) {
            return;
        }
        if (inFlightRef.current) {
            return;
        }
        inFlightRef.current = true;
        armWatchdog();
        setSrc(withCacheBust(url));
    }, [armWatchdog, url]);

    useEffect(() => {
        const timer = setInterval(requestNextSnapshot, intervalMs);
        return () => clearInterval(timer);
    }, [intervalMs, requestNextSnapshot]);

    useEffect(() => {
        armWatchdog(); // guard the initial request
        return () => clearWatchdog();
    }, [armWatchdog, clearWatchdog]);

    const badgeClass = useMemo(() => {
        if (status === 'online') return 'bg-green-600';
        if (status === 'connecting') return 'bg-blue-600';
        return 'bg-red-600';
    }, [status]);

    return (
        <div className={`relative bg-black overflow-hidden ${className}`}>
            <img
                src={src}
                alt={cameraName}
                className={`w-full h-full ${objectFit === 'contain' ? 'object-contain' : 'object-cover'}`}
                onLoad={() => {
                    clearWatchdog();
                    inFlightRef.current = false;
                    hasEverLoadedRef.current = true;
                    consecutiveErrorsRef.current = 0;
                    setStatus('online');
                }}
                onError={() => {
                    clearWatchdog();
                    inFlightRef.current = false;
                    consecutiveErrorsRef.current += 1;
                    // Only show offline if never loaded, or 10+ consecutive errors
                    if (!hasEverLoadedRef.current || consecutiveErrorsRef.current >= 10) {
                        setStatus('offline');
                    }
                }}
            />

            {status !== 'online' && !hasEverLoadedRef.current && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70">
                    {status === 'connecting' ? (
                        <>
                            <Loader2 className="w-9 h-9 text-blue-400 animate-spin mb-3" />
                            <p className="text-white text-xs">{t('stream.connectingStream')}</p>
                        </>
                    ) : (
                        <>
                            <WifiOff className="w-9 h-9 text-red-400 mb-3" />
                            <p className="text-white text-xs">{t('stream.streamOffline')}</p>
                        </>
                    )}
                </div>
            )}

            <div className="absolute top-2 left-2">
                <div className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium text-white ${badgeClass}`}>
                    <Wifi className="w-3 h-3" />
                    {cameraName}
                </div>
            </div>
        </div>
    );
};


export const SnapshotPlayer = (props: SnapshotPlayerProps) => {
    const {
        url,
        cameraName = 'Camera',
        className = '',
        objectFit = 'cover',
        intervalMs = 1200,
    } = props;

    return (
        <SnapshotPlayerContent
            key={`${url}-${intervalMs}`}
            url={url}
            cameraName={cameraName}
            className={className}
            objectFit={objectFit}
            intervalMs={intervalMs}
        />
    );
};
