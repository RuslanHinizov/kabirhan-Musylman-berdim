import { useState, useEffect } from 'react';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';

interface MJPEGPlayerProps {
    url: string;
    cameraName?: string;
    className?: string;
}

export const MJPEGPlayer = ({
    url,
    cameraName = 'Camera',
    className = ''
}: MJPEGPlayerProps) => {
    const [status, setStatus] = useState<'connecting' | 'online' | 'offline'>('connecting');
    const [imgKey, setImgKey] = useState(0);

    useEffect(() => {
        setStatus('connecting');
        setImgKey(k => k + 1);
    }, [url]);

    const handleLoad = () => {
        setStatus('online');
    };

    const handleError = () => {
        setStatus('offline');
        // Retry after 3 seconds
        setTimeout(() => {
            setImgKey(k => k + 1);
            setStatus('connecting');
        }, 3000);
    };

    return (
        <div className={`relative bg-black overflow-hidden ${className}`}>
            {/* MJPEG Stream as img */}
            <img
                key={imgKey}
                src={url}
                alt={cameraName}
                className="w-full h-full object-cover"
                onLoad={handleLoad}
                onError={handleError}
            />

            {/* Status Overlay */}
            {status !== 'online' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80">
                    {status === 'connecting' && (
                        <>
                            <Loader2 className="w-12 h-12 text-blue-400 animate-spin mb-4" />
                            <p className="text-white text-sm">Connecting to stream...</p>
                            <p className="text-gray-500 text-xs mt-1">{cameraName}</p>
                        </>
                    )}

                    {status === 'offline' && (
                        <>
                            <WifiOff className="w-12 h-12 text-red-400 mb-4" />
                            <p className="text-white text-sm">Stream Offline</p>
                            <p className="text-gray-500 text-xs mt-1">{cameraName}</p>
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
                    {cameraName}
                </div>
            </div>
        </div>
    );
};
