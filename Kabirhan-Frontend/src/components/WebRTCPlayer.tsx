import { useState, useEffect, useRef, useCallback } from 'react';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';
import { MJPEGPlayer } from './MJPEGPlayer';
import { useTranslation } from 'react-i18next';
import { BACKEND_HTTP_URL } from '../config/backend';

interface WebRTCPlayerProps {
    camId: string;           // e.g. "analytics-1" or "ptz-2"
    cameraName?: string;
    className?: string;
    objectFit?: 'cover' | 'contain';
    fallbackToMjpeg?: boolean;  // Fall back to MJPEG if WebRTC fails
    enabled?: boolean;          // If false, don't open a WebRTC connection
}

export const WebRTCPlayer = ({
    camId,
    cameraName = 'Camera',
    className = '',
    objectFit = 'cover',
    fallbackToMjpeg = true,
    enabled = true,
}: WebRTCPlayerProps) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const pcRef = useRef<RTCPeerConnection | null>(null);
    const [status, setStatus] = useState<'connecting' | 'online' | 'offline'>('connecting');
    const [useMjpegFallback, setUseMjpegFallback] = useState(false);
    const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const mountedRef = useRef(true);
    const { t } = useTranslation();

    const cleanup = useCallback(() => {
        if (retryTimeoutRef.current) {
            clearTimeout(retryTimeoutRef.current);
            retryTimeoutRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        if (pcRef.current) {
            pcRef.current.close();
            pcRef.current = null;
        }
    }, []);

    const connect = useCallback(async () => {
        if (!mountedRef.current || !enabled) return;
        cleanup();
        setStatus('connecting');

        try {
            // Check if WebRTC is available on the server
            const statusRes = await fetch(`${BACKEND_HTTP_URL}/api/webrtc/status`);
            const statusData = await statusRes.json();

            if (!statusData.available) {
                if (fallbackToMjpeg) {
                    setUseMjpegFallback(true);
                    return;
                }
                setStatus('offline');
                return;
            }

            // Create peer connection
            const isLocalBackend = BACKEND_HTTP_URL.includes('localhost') || BACKEND_HTTP_URL.includes('127.0.0.1');
            const pc = new RTCPeerConnection({
                // Local backend does not need public STUN and produces cleaner ICE behavior.
                iceServers: isLocalBackend ? [] : [{ urls: 'stun:stun.l.google.com:19302' }]
            });
            pcRef.current = pc;

            // We want to receive video only
            pc.addTransceiver('video', { direction: 'recvonly' });

            // Handle incoming track
            pc.ontrack = (event) => {
                if (!mountedRef.current) return;
                if (videoRef.current && event.streams[0]) {
                    videoRef.current.srcObject = event.streams[0];
                    setStatus('online');
                }
            };

            // Handle connection state changes
            pc.onconnectionstatechange = () => {
                if (!mountedRef.current) return;
                const s = pc.connectionState;
                if (s === 'connected') {
                    setStatus('online');
                } else if (s === 'failed' || s === 'disconnected' || s === 'closed') {
                    setStatus('offline');
                    if (fallbackToMjpeg) {
                        setUseMjpegFallback(true);
                    } else {
                        // Retry after 3 seconds
                        retryTimeoutRef.current = setTimeout(() => {
                            if (mountedRef.current) connect();
                        }, 3000);
                    }
                }
            };

            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // Wait briefly for ICE gathering; keep local switching snappy.
            const iceGatherTimeoutMs = isLocalBackend ? 600 : 1800;
            await new Promise<void>((resolve) => {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    const timer = setTimeout(() => {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }, iceGatherTimeoutMs);

                    const checkState = () => {
                        if (pc.iceGatheringState === 'complete') {
                            clearTimeout(timer);
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    };
                    pc.addEventListener('icegatheringstatechange', checkState);
                }
            });

            if (!mountedRef.current || !pc.localDescription) return;

            // Send offer to server
            const response = await fetch(`${BACKEND_HTTP_URL}/api/webrtc/offer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    camId,
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type,
                }),
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            const answer = await response.json();

            if (!mountedRef.current) return;

            // Set remote description (server's answer)
            await pc.setRemoteDescription(
                new RTCSessionDescription({
                    sdp: answer.sdp,
                    type: answer.type,
                })
            );

        } catch (err) {
            console.error(`[WebRTC] Connection failed for ${camId}:`, err);
            if (!mountedRef.current) return;

            if (fallbackToMjpeg) {
                setUseMjpegFallback(true);
            } else {
                setStatus('offline');
                retryTimeoutRef.current = setTimeout(() => {
                    if (mountedRef.current) connect();
                }, 3000);
            }
        }
    }, [camId, fallbackToMjpeg, cleanup, enabled]);

    useEffect(() => {
        mountedRef.current = true;
        setUseMjpegFallback(false);
        if (enabled) {
            connect();
        } else {
            cleanup();
            setStatus('offline');
        }

        return () => {
            mountedRef.current = false;
            cleanup();
        };
    }, [camId, connect, cleanup, enabled]);

    // MJPEG fallback
    if (useMjpegFallback) {
        // Use full cam_id for stream URL (e.g. /stream/ptz-1, /stream/analytics-3)
        // This matches the backend's /stream/{cam_id_full} endpoint
        return (
                <MJPEGPlayer
                url={`${BACKEND_HTTP_URL}/stream/${camId}`}
                cameraName={cameraName}
                className={className}
                objectFit={objectFit}
            />
        );
    }

    return (
        <div className={`relative bg-black overflow-hidden ${className}`}>
            {/* WebRTC Video */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`w-full h-full ${objectFit === 'contain' ? 'object-contain' : 'object-cover'}`}
            />

            {/* Status Overlay */}
            {status !== 'online' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80">
                    {status === 'connecting' && (
                        <>
                            <Loader2 className="w-12 h-12 text-blue-400 animate-spin mb-4" />
                            <p className="text-white text-sm">{t('stream.connectingWebRTC')}</p>
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
            <div className="absolute top-2 left-2 flex items-center gap-2">
                <div className={`
                    flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium
                    ${status === 'online' ? 'bg-green-600' :
                        status === 'connecting' ? 'bg-blue-600' : 'bg-red-600'}
                    text-white
                `}>
                    <Wifi className="w-3 h-3" />
                    {cameraName}
                    {status === 'online' && (
                        <span className="text-green-200 text-[10px] ml-1">{t('stream.webrtcLabel')}</span>
                    )}
                </div>
            </div>
        </div>
    );
};
