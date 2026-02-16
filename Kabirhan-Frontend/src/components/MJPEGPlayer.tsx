import { useCallback, useEffect, useRef, useState } from 'react';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface MJPEGPlayerProps {
    url: string;
    cameraName?: string;
    className?: string;
    objectFit?: 'cover' | 'contain';
    showStatusOverlay?: boolean;
    showInfoBadge?: boolean;
}

type StreamStatus = 'connecting' | 'online' | 'offline';

const MJPEGPlayerContent = ({
    url,
    cameraName,
    className,
    objectFit,
    showStatusOverlay = true,
    showInfoBadge = true,
}: MJPEGPlayerProps) => {
    const { t } = useTranslation();
    const [status, setStatus] = useState<StreamStatus>('connecting');
    const [imgKey, setImgKey] = useState(0);
    const mountedRef = useRef(true);
    const loadedRef = useRef(false);
    const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const clearRetryTimer = useCallback(() => {
        if (retryTimerRef.current) {
            clearTimeout(retryTimerRef.current);
            retryTimerRef.current = null;
        }
    }, []);

    const reconnect = useCallback((delayMs: number = 1000) => {
        if (!mountedRef.current) return;
        clearRetryTimer();
        retryTimerRef.current = setTimeout(() => {
            if (!mountedRef.current) return;
            loadedRef.current = false;
            setStatus('connecting');
            setImgKey(k => k + 1);
        }, delayMs);
    }, [clearRetryTimer]);

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            clearRetryTimer();
        };
    }, [clearRetryTimer]);

    // Connection watchdog: if first frame doesn't arrive quickly, force reconnect.
    useEffect(() => {
        loadedRef.current = false;
        const watchdog = setTimeout(() => {
            if (!mountedRef.current) return;
            if (!loadedRef.current) {
                setStatus('offline');
                reconnect(1200);
            }
        }, 8000);
        return () => clearTimeout(watchdog);
    }, [imgKey, url, reconnect]);

    const handleLoad = () => {
        loadedRef.current = true;
        clearRetryTimer();
        setStatus('online');
    };

    const handleError = () => {
        loadedRef.current = false;
        setStatus('offline');
        reconnect(1800);
    };

    const src = `${url}${url.includes('?') ? '&' : '?'}_rv=${imgKey}`;

    return (
        <div className={`relative bg-black overflow-hidden ${className}`}>
            {/* MJPEG Stream as img */}
            <img
                key={imgKey}
                src={src}
                alt={cameraName}
                className={`w-full h-full ${objectFit === 'contain' ? 'object-contain' : 'object-cover'}`}
                onLoad={handleLoad}
                onError={handleError}
            />

            {/* Status Overlay */}
            {showStatusOverlay && status !== 'online' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80">
                    {status === 'connecting' && (
                        <>
                            <Loader2 className="w-12 h-12 text-blue-400 animate-spin mb-4" />
                            <p className="text-white text-sm">{t('stream.connectingStream')}</p>
                            <p className="text-gray-500 text-xs mt-1">{cameraName}</p>
                        </>
                    )}

                    {status === 'offline' && (
                        <>
                            <WifiOff className="w-12 h-12 text-red-400 mb-4" />
                            <p className="text-white text-sm">{t('stream.streamOffline')}</p>
                            <p className="text-gray-500 text-xs mt-1">{cameraName}</p>
                        </>
                    )}
                </div>
            )}

            {/* Camera Info Badge */}
            {showInfoBadge && (
                <div className="absolute top-2 left-2 flex items-center gap-2">
                    <div className={`
                        flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium
                        ${status === 'online' ? 'bg-green-600' :
                            status === 'connecting' ? 'bg-blue-600' : 'bg-red-600'}
                        text-white
                    `}>
                        <Wifi className="w-3 h-3" />
                        {cameraName}
                    </div>
                </div>
            )}
        </div>
    );
};

export const MJPEGPlayer = (props: MJPEGPlayerProps) => {
    const {
        url,
        cameraName = 'Camera',
        className = '',
        objectFit = 'cover',
        showStatusOverlay = true,
        showInfoBadge = true,
    } = props;

    // Remount when URL changes so connection state resets without effect-driven setState.
    return (
        <MJPEGPlayerContent
            key={url}
            url={url}
            cameraName={cameraName}
            className={className}
            objectFit={objectFit}
            showStatusOverlay={showStatusOverlay}
            showInfoBadge={showInfoBadge}
        />
    );
};
