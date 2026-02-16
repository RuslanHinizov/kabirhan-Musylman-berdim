import { useMemo, useEffect, useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useRaceStore } from '../../store/raceStore';
import { useCameraStore } from '../../store/cameraStore';
import { TRACK_LENGTH } from '../../types';

// Smooth interpolation
const useSmooth = (target: number, speed: number = 0.2) => {
    const [value, setValue] = useState(target);
    const targetRef = useRef(target);
    const frameRef = useRef<number | undefined>(undefined);

    useEffect(() => { targetRef.current = target; }, [target]);

    useEffect(() => {
        const animate = () => {
            setValue(prev => {
                const diff = targetRef.current - prev;
                if (Math.abs(diff) < 0.001) return targetRef.current;
                return prev + diff * speed;
            });
            frameRef.current = requestAnimationFrame(animate);
        };
        frameRef.current = requestAnimationFrame(animate);
        return () => { if (frameRef.current) cancelAnimationFrame(frameRef.current); };
    }, [speed]);

    return value;
};

// Horse with smooth movement
interface HorseProps {
    percent: number;
    laneOffset: number;
    horse: { color: string; number: number };
    isLeader: boolean;
    getPos: (pct: number, offset: number) => { x: number; y: number };
}

const Horse = ({ percent, laneOffset, horse, isLeader, getPos }: HorseProps) => {
    const smoothPercent = useSmooth(percent, 0.25);
    const pos = getPos(smoothPercent, laneOffset);

    return (
        <g>
            <ellipse cx={pos.x + 2} cy={pos.y + 3} rx={isLeader ? 12 : 10} ry={5} fill="rgba(0,0,0,0.3)" />
            {isLeader && <circle cx={pos.x} cy={pos.y} r={18} fill="url(#leaderGlow)" opacity="0.5" />}
            <circle cx={pos.x} cy={pos.y} r={isLeader ? 14 : 11} fill={horse.color}
                stroke={isLeader ? '#FFD700' : '#000'} strokeWidth={isLeader ? 3 : 1} />
            <text x={pos.x} y={pos.y + 1} fontSize={isLeader ? "12" : "10"} fontWeight="bold"
                fill="white" textAnchor="middle" dominantBaseline="middle">{horse.number}</text>
        </g>
    );
};

export const Track2DView = () => {
    const { t } = useTranslation();
    const { rankings, race } = useRaceStore();
    const { analyticsCameras } = useCameraStore();

    const svgWidth = 1200;
    const svgHeight = 550;

    // Stadium dimensions
    const straightLen = 450;
    const curveRadius = 160;
    const trackWidth = 70;

    const cx = svgWidth / 2;
    const cy = svgHeight / 2;

    // Track perimeter
    const perimeter = (straightLen * 2) + (2 * Math.PI * curveRadius);

    // Get position on track
    const getPos = (pct: number, laneOffset: number = 0) => {
        const p = ((pct % 1) + 1) % 1;
        const dist = p * perimeter;

        if (dist < straightLen) {
            return {
                x: cx - straightLen / 2 + dist,
                y: cy - curveRadius - laneOffset
            };
        } else if (dist < straightLen + Math.PI * curveRadius) {
            const angle = -Math.PI / 2 + (dist - straightLen) / curveRadius;
            const r = curveRadius + laneOffset;
            return {
                x: cx + straightLen / 2 + Math.cos(angle) * r,
                y: cy + Math.sin(angle) * r
            };
        } else if (dist < 2 * straightLen + Math.PI * curveRadius) {
            const d = dist - straightLen - Math.PI * curveRadius;
            return {
                x: cx + straightLen / 2 - d,
                y: cy + curveRadius + laneOffset
            };
        } else {
            const angle = Math.PI / 2 + (dist - 2 * straightLen - Math.PI * curveRadius) / curveRadius;
            const r = curveRadius + laneOffset;
            return {
                x: cx - straightLen / 2 + Math.cos(angle) * r,
                y: cy + Math.sin(angle) * r
            };
        }
    };

    // Inner edge (for cameras) - place cameras FURTHER INSIDE the track (green area)
    const getInnerPos = (pct: number) => getPos(pct, -trackWidth / 2 - 25);

    // Outer edge (for distance markers)
    const getOuterPos = (pct: number) => getPos(pct, trackWidth / 2 + 30);

    // Track path
    const trackPath = (r: number) => {
        const l = cx - straightLen / 2, ri = cx + straightLen / 2;
        return `M ${l} ${cy - r} L ${ri} ${cy - r} A ${r} ${r} 0 0 1 ${ri} ${cy + r} L ${l} ${cy + r} A ${r} ${r} 0 0 1 ${l} ${cy - r} Z`;
    };

    // Analytics camera positions on track — each camera at its position (100m, 200m, etc.)
    const cameras = useMemo(() => {
        const detectionThreshold = 60;

        return analyticsCameras.map(cam => {
            // Position camera at trackEnd (100, 200, 300...)
            const position = cam.trackEnd;
            const pct = position / TRACK_LENGTH;

            // Check if any horse is near this camera's segment
            const isActive = rankings.some(horse => {
                const horsePos = horse.distanceCovered % TRACK_LENGTH;
                return horsePos >= cam.trackStart - detectionThreshold && horsePos < cam.trackEnd + detectionThreshold;
            });

            return {
                ...getInnerPos(pct),
                active: isActive,
                name: cam.name,
                label: `${position}m`,  // Just show the position: 100m, 200m, etc.
            };
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [rankings, analyticsCameras]);

    // Distance markers (outside track)
    const markers = useMemo(() => {
        const arr: Array<{ x: number, y: number, label: string }> = [];
        for (let m = 0; m < TRACK_LENGTH; m += 200) {
            arr.push({ ...getOuterPos(m / TRACK_LENGTH), label: `${m}m` });
        }
        return arr;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Horse data
    const horses = useMemo(() => {
        return rankings.map((h, i) => ({
            horse: h,
            percent: (h.distanceCovered % TRACK_LENGTH) / TRACK_LENGTH,
            laneOffset: (i - (rankings.length - 1) / 2) * 4,
            isLeader: i === 0
        }));
    }, [rankings]);

    const startPct = (race.startFinishPosition || 0) / TRACK_LENGTH;

    return (
        <div className="h-full flex flex-col overflow-hidden bg-[#0a1628]">
            <div className="flex-shrink-0 px-6 py-3 flex items-center justify-between border-b border-white/10">
                <div>
                    <h2 className="text-lg font-semibold text-white">{t('track.liveRaceTrack')}</h2>
                    <p className="text-sm text-gray-400">{TRACK_LENGTH}m {t('track.stadium')} • {t('header.lap')} {rankings[0]?.currentLap || 1}/{race.totalLaps}</p>
                </div>
                <div className="flex items-center gap-1.5">
                    {rankings.slice(0, 10).map((h, i) => (
                        <div key={h.id} className={`w-7 h-7 rounded-full text-xs font-bold text-white flex items-center justify-center
              ${i === 0 ? 'ring-2 ring-yellow-400' : ''}`} style={{ backgroundColor: h.color }}>{h.number}</div>
                    ))}
                </div>
            </div>

            <div className="flex-1 min-h-0 p-4 flex items-center justify-center">
                <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-full">
                    <defs>
                        <linearGradient id="trackGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="#1e3a5f" />
                            <stop offset="50%" stopColor="#0f2847" />
                            <stop offset="100%" stopColor="#1e3a5f" />
                        </linearGradient>
                        <radialGradient id="leaderGlow">
                            <stop offset="0%" stopColor="#FFD700" stopOpacity="0.8" />
                            <stop offset="100%" stopColor="#FFD700" stopOpacity="0" />
                        </radialGradient>
                    </defs>

                    {/* Grass infield */}
                    <path d={trackPath(curveRadius - trackWidth / 2 - 5)} fill="#0d3a28" />

                    {/* Track */}
                    <path d={trackPath(curveRadius + trackWidth / 2)} fill="url(#trackGrad)" stroke="#2d4a6f" strokeWidth="2" />
                    <path d={trackPath(curveRadius - trackWidth / 2)} fill="#0a1628" stroke="#2d4a6f" strokeWidth="1" />

                    {/* Lane lines */}
                    {[-20, 0, 20].map(o => (
                        <path key={o} d={trackPath(curveRadius + o)} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" strokeDasharray="15 15" />
                    ))}

                    {/* Start/Finish */}
                    {(() => {
                        const p1 = getPos(startPct, -trackWidth / 2);
                        const p2 = getPos(startPct, trackWidth / 2);
                        return (
                            <g>
                                <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke="white" strokeWidth="6" />
                                <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke="#FFD700" strokeWidth="3" />
                            </g>
                        );
                    })()}

                    {/* Distance markers (OUTSIDE - every 200m) */}
                    {markers.map((m, i) => (
                        <text key={i} x={m.x} y={m.y} fontSize="12" fill="#94a3b8" textAnchor="middle" dominantBaseline="middle" fontWeight="600">
                            {m.label}
                        </text>
                    ))}

                    {/* Analytics Cameras (ON track inner edge) */}
                    {cameras.map((c, i) => (
                        <g key={i}>
                            <circle cx={c.x} cy={c.y} r={c.active ? 6 : 4} fill={c.active ? '#F59E0B' : '#64748b'}
                                stroke={c.active ? '#FCD34D' : '#94a3b8'} strokeWidth="1.5" />
                            {c.active && (
                                <circle cx={c.x} cy={c.y} r={10} fill="none" stroke="#F59E0B" strokeWidth="1" opacity="0.4">
                                    <animate attributeName="r" values="6;12;6" dur="1s" repeatCount="indefinite" />
                                    <animate attributeName="opacity" values="0.4;0;0.4" dur="1s" repeatCount="indefinite" />
                                </circle>
                            )}
                            {/* Camera label - placed below/near the camera marker */}
                            <text x={c.x} y={c.y + 12} fontSize="9" fill={c.active ? '#FCD34D' : '#64748b'}
                                textAnchor="middle" dominantBaseline="hanging" fontWeight="600" style={{ pointerEvents: 'none' }}>
                                {c.label}
                            </text>
                        </g>
                    ))}

                    {/* Horses */}
                    {horses.map(({ horse, percent, laneOffset, isLeader }) => (
                        <Horse key={horse.id} percent={percent} laneOffset={laneOffset} horse={horse} isLeader={isLeader} getPos={getPos} />
                    ))}
                </svg>
            </div>

            <div className="flex-shrink-0 px-6 py-3 flex items-center justify-center gap-8 text-xs text-gray-400 border-t border-white/10">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-[#475569]" />
                    <span>{t('track.camera')} ({analyticsCameras.length})</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-[#F59E0B]" />
                    <span>{t('track.horseDetected')}</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-5 h-1 bg-yellow-400 rounded" />
                    <span>{t('track.startFinish')}</span>
                </div>
            </div>
        </div>
    );
};
