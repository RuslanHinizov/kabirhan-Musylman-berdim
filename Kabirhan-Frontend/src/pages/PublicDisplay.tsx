import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence, useSpring, useTransform } from 'framer-motion';
import { Trophy } from 'lucide-react';
import { connectToBackend, disconnectFromBackend } from '../services/backendConnection';
import { useRaceStore } from '../store/raceStore';
import { useCameraStore } from '../store/cameraStore';
import { WebRTCPlayer } from '../components/WebRTCPlayer';
import { getSilkImagePath } from '../utils/silkUtils';
import { BACKEND_HTTP_URL } from '../config/backend';

// Format time as MM:SS.d
const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toFixed(1).padStart(4, '0')}`;
};

// Animated time component
const AnimatedTime = ({ value }: { value: number }) => {
    const spring = useSpring(value, { stiffness: 100, damping: 30 });
    const [displayValue, setDisplayValue] = useState(formatTime(value));

    useEffect(() => { spring.set(value); }, [spring, value]);
    useEffect(() => {
        const unsubscribe = spring.on('change', v => setDisplayValue(formatTime(v)));
        return unsubscribe;
    }, [spring]);

    return <span>{displayValue}</span>;
};

// Animated number component (for speed)
const AnimatedNumber = ({ value, decimals = 1 }: { value: number; decimals?: number }) => {
    const spring = useSpring(value, { stiffness: 100, damping: 30 });
    const display = useTransform(spring, v => v.toFixed(decimals));
    const [displayValue, setDisplayValue] = useState(value.toFixed(decimals));

    useEffect(() => { spring.set(value); }, [spring, value]);
    useEffect(() => {
        const unsubscribe = display.on('change', v => setDisplayValue(v));
        return unsubscribe;
    }, [display]);

    return <span>{displayValue}</span>;
};

export const PublicDisplay = () => {
    const { t } = useTranslation();
    const { race, rankings } = useRaceStore();
    const { activePTZCameraId, ptzCameras, syncFromStorage } = useCameraStore();
    const [ptzStates, setPtzStates] = useState<Record<string, string>>({});

    // Connect to backend WebSocket
    useEffect(() => {
        connectToBackend();
        return () => disconnectFromBackend();
    }, []);

    // Listen for camera changes from operator panel (other tabs)
    useEffect(() => {
        const handleStorageChange = () => syncFromStorage();
        window.addEventListener('storage', handleStorageChange);
        return () => window.removeEventListener('storage', handleStorageChange);
    }, [syncFromStorage]);

    // Keep runtime PTZ states synced to avoid retry storms on offline cameras.
    useEffect(() => {
        const fetchPtzStates = async () => {
            try {
                const response = await fetch(`${BACKEND_HTTP_URL}/api/streams/status`);
                if (!response.ok) return;
                const all = await response.json() as Record<string, { state?: string }>;
                const next: Record<string, string> = {};
                ptzCameras.forEach((cam) => {
                    next[cam.id] = all[cam.id]?.state || 'idle';
                });
                setPtzStates(next);
            } catch {
                // Keep last known states on transient errors.
            }
        };

        void fetchPtzStates();
        const interval = setInterval(() => {
            void fetchPtzStates();
        }, 3000);
        return () => clearInterval(interval);
    }, [ptzCameras]);

    const leader = rankings[0];
    const speed = leader ? leader.speed * 3.6 : 0;
    const time = leader?.timeElapsed || 0;
    const winner = rankings[0];

    // Top horses for ranking display
    const topHorses = rankings.slice(0, 10);

    return (
        <div className="h-screen w-screen bg-black relative overflow-hidden">
            {/* PTZ Video Background - keep PTZ WebRTC streams warm for instant camera switching */}
            {ptzCameras.map((cam) => (
                <div
                    key={cam.id}
                    className={`absolute inset-0 transition-opacity duration-150 ${
                        cam.id === activePTZCameraId ? 'opacity-100 z-0' : 'opacity-0 pointer-events-none'
                    }`}
                >
                    {(() => {
                        const state = ptzStates[cam.id] || 'idle';
                        const isActive = cam.id === activePTZCameraId;
                        const shouldEnable = isActive;

                        return (
                            <WebRTCPlayer
                                camId={cam.id}
                                cameraName={cam.name}
                                className="w-full h-full"
                                objectFit="cover"
                                fallbackToMjpeg={isActive && state !== 'offline'}
                                enabled={shouldEnable}
                            />
                        );
                    })()}
                </div>
            ))}

            {/* TOP LEFT - Time & Lap */}
            <div className="absolute top-6 left-6 z-10">
                <div className="bg-black/80 backdrop-blur-sm rounded-lg px-4 py-3 border border-white/10">
                    <div className="text-3xl font-bold text-white font-mono tabular-nums tracking-tight">
                        <AnimatedTime value={time} />
                    </div>
                    <div className="text-sm text-white/70 mt-1 font-medium">
                        {t('header.lap')} {leader?.currentLap || 1}/{race.totalLaps}
                    </div>
                </div>
            </div>

            {/* BOTTOM - Professional TV Racing Bar */}
            <div className="absolute bottom-0 left-0 right-0 z-20">
                <div
                    className="border-t border-white/20"
                    style={{
                        background: 'rgba(0, 0, 0, 0.85)',
                        backdropFilter: 'blur(8px)',
                    }}
                >
                    <div className="flex items-center h-[120px]">

                        {/* LEFT - Speedometer */}
                        <div className="w-36 h-full flex items-center justify-center border-r border-white/10 bg-black/50">
                            <div className="text-center">
                                <div className="text-4xl font-bold text-white font-mono tabular-nums">
                                    <AnimatedNumber value={speed} decimals={1} />
                                </div>
                                <div className="text-sm text-gray-400 mt-1">km/h</div>
                            </div>
                        </div>

                        {/* CENTER - Jockeys with Numbers (1st place on right, last on left) */}
                        <div className="flex-1 relative overflow-visible" style={{ minHeight: 120 }}>
                            {topHorses.map((horse, index) => {
                                const slotWidth = 80;
                                const containerCenter = 50;
                                // Reverse: index 0 (1st place) goes to right, last goes to left
                                const reversedOffset = ((topHorses.length - 1 - index) - (topHorses.length - 1) / 2) * slotWidth;

                                return (
                                    <motion.div
                                        key={horse.id}
                                        layout
                                        animate={{ x: reversedOffset }}
                                        transition={{
                                            x: {
                                                type: 'spring',
                                                stiffness: 40,
                                                damping: 30,
                                            },
                                            layout: {
                                                type: 'spring',
                                                stiffness: 40,
                                                damping: 30,
                                            },
                                        }}
                                        className="absolute bottom-2 flex flex-col items-center w-[72px]"
                                        style={{
                                            left: `${containerCenter}%`,
                                            marginLeft: -36,
                                        }}
                                    >
                                        {/* Jockey Icon */}
                                        <img
                                            src={getSilkImagePath(horse.silkId)}
                                            alt={`#${horse.number}`}
                                            className="h-[60px] w-auto object-contain"
                                            style={{
                                                filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.6))',
                                            }}
                                        />

                                        {/* Horse Number */}
                                        <div className="font-extrabold text-2xl text-white leading-none mt-1">
                                            {horse.number}
                                        </div>
                                    </motion.div>
                                );
                            })}
                        </div>

                        {/* RIGHT - Finish Marker (next to 1st place) */}
                        <div className="flex items-center justify-center w-20 h-full border-l border-white/10">
                            <div className="relative flex flex-col items-center">
                                <div className="w-10 h-10 rounded-full bg-red-600 border-[3px] border-white shadow-lg z-10"></div>
                                <div className="w-1.5 h-12 bg-gradient-to-b from-gray-300 to-gray-500 -mt-1"></div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>

            {/* Race Finished Overlay */}
            <AnimatePresence>
                {race.status === 'finished' && (
                    <motion.div
                        className="absolute inset-0 bg-black/90 flex items-center justify-center z-50"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <motion.div
                            className="text-center"
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: 0.2 }}
                        >
                            <Trophy className="w-20 h-20 text-amber-400 mx-auto mb-6" strokeWidth={1.5} />
                            <h2 className="text-4xl font-bold text-white mb-8">{t('display.raceFinished')}</h2>

                            {winner && (
                                <div className="flex items-center justify-center gap-6">
                                    <div className="w-28 h-36 flex items-center justify-center">
                                        <img
                                            src={getSilkImagePath(winner.silkId)}
                                            alt={`Winner silk`}
                                            className="w-24 h-32 object-contain drop-shadow-[0_8px_16px_rgba(0,0,0,0.5)]"
                                        />
                                    </div>
                                    <div className="text-left">
                                        <div className="text-xs text-amber-400 uppercase tracking-wider mb-1">{t('display.winner')}</div>
                                        <p className="text-2xl font-bold text-white">{winner.name}</p>
                                        <p className="text-gray-400">{winner.jockeyName}</p>
                                        <p className="text-amber-400 font-mono text-xl mt-2">#{winner.number}</p>
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
