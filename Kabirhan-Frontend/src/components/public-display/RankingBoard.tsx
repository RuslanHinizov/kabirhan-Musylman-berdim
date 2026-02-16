import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import { Timer, Flag } from 'lucide-react';
import { useRaceStore } from '../../store/raceStore';
import { useState, useEffect, useRef } from 'react';
import { getSilkImagePath } from '../../utils/silkUtils';
import { useTranslation } from 'react-i18next';

export const RankingBoard = () => {
    const { t } = useTranslation();
    const { rankings, race } = useRaceStore();
    const [previousPositions, setPreviousPositions] = useState<Record<string, number>>({});
    const prevRankingsRef = useRef(rankings);

    // Top 10 horses
    const topHorses = rankings.slice(0, 10);

    /* eslint-disable react-hooks/set-state-in-effect */
    useEffect(() => {
        const newPrevPositions: Record<string, number> = {};
        prevRankingsRef.current.forEach(horse => {
            newPrevPositions[horse.id] = horse.currentPosition;
        });
        setPreviousPositions(newPrevPositions);
        prevRankingsRef.current = rankings;
    }, [rankings]);
    /* eslint-enable react-hooks/set-state-in-effect */

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    const formatGap = (gap: number) => {
        if (gap === 0) return '—';
        return `+${gap.toFixed(1)}s`;
    };

    const getPositionChange = (horseId: string, currentPos: number) => {
        const prevPos = previousPositions[horseId];
        if (!prevPos || prevPos === currentPos) return 0;
        return prevPos - currentPos;
    };

    const leader = topHorses[0];

    return (
        <div className="h-full flex flex-col bg-[var(--background)]">
            {/* Header */}
            <div className="flex-shrink-0 px-6 py-3 flex items-center justify-between border-b border-[var(--border)]">
                <div className="flex items-center gap-6">
                    <span className="text-sm font-medium text-[var(--text-secondary)]">{t('ranking.liveStandings')}</span>

                    <div className="flex items-center gap-2 text-sm">
                        <Flag className="w-4 h-4 text-[var(--text-muted)]" />
                        <span className="text-[var(--text-primary)] font-medium">{leader?.currentLap || 1}</span>
                        <span className="text-[var(--text-muted)]">/ {race.totalLaps}</span>
                    </div>

                    <div className="flex items-center gap-2 text-sm">
                        <Timer className="w-4 h-4 text-[var(--text-muted)]" />
                        <span className="text-[var(--text-primary)] font-medium stat-value">
                            {formatTime(leader?.timeElapsed || 0)}
                        </span>
                    </div>
                </div>

                {leader && (
                    <div className="flex items-center gap-2 text-sm">
                        <span className="text-[var(--text-muted)]">{t('ranking.leader')}</span>
                        <img
                            src={getSilkImagePath(leader.silkId)}
                            alt={leader.name}
                            className="w-8 h-10 object-contain"
                        />
                        <span className="text-[var(--text-primary)] font-medium">{leader.name}</span>
                    </div>
                )}
            </div>

            {/* Leaderboard with Silk Images */}
            <div className="flex-1 px-4 py-3 overflow-x-auto">
                <LayoutGroup>
                    <motion.div className="flex gap-3 h-full min-w-max" layout>
                        <AnimatePresence mode="popLayout">
                            {topHorses.map((horse, index) => {
                                const positionChange = getPositionChange(horse.id, horse.currentPosition);
                                const isLeader = index === 0;
                                const isTop3 = index < 3;

                                return (
                                    <motion.div
                                        key={horse.id}
                                        layout
                                        layoutId={horse.id}
                                        initial={{ opacity: 0, scale: 0.95 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.95 }}
                                        transition={{
                                            layout: { type: "spring", stiffness: 200, damping: 25 }
                                        }}
                                        className={`
                                            ${isLeader ? 'w-48' : isTop3 ? 'w-40' : 'w-32'}
                                            min-w-0 rounded-xl p-3 flex flex-col relative overflow-hidden flex-shrink-0
                                            ${isLeader
                                                ? 'bg-gradient-to-br from-amber-500/20 to-orange-600/10 border-2 border-amber-500'
                                                : isTop3
                                                    ? 'bg-[var(--surface)] border border-[var(--border)]'
                                                    : 'bg-[var(--surface)]/50 border border-[var(--border)]/50'}
                                        `}
                                    >
                                        {/* Position Badge */}
                                        <div className="flex items-center justify-between mb-2">
                                            <div className={`
                                                ${isLeader ? 'w-10 h-10 text-lg' : 'w-8 h-8 text-sm'}
                                                rounded-full flex items-center justify-center font-bold
                                                ${index === 0 ? 'bg-amber-500 text-black' :
                                                    index === 1 ? 'bg-gray-400 text-black' :
                                                        index === 2 ? 'bg-orange-600 text-white' :
                                                            'bg-gray-600 text-white'}
                                            `}>
                                                {horse.currentPosition}
                                            </div>

                                            {positionChange !== 0 && (
                                                <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${positionChange > 0
                                                    ? 'bg-green-500/20 text-green-400'
                                                    : 'bg-red-500/20 text-red-400'
                                                    }`}>
                                                    {positionChange > 0 ? `↑${positionChange}` : `↓${Math.abs(positionChange)}`}
                                                </span>
                                            )}
                                        </div>

                                        {/* Silk Image */}
                                        <div className="flex-1 flex items-center justify-center mb-2">
                                            <div className={`
                                                relative ${isLeader ? 'w-20 h-24' : isTop3 ? 'w-16 h-20' : 'w-12 h-16'}
                                                rounded-lg bg-white shadow-inner flex items-center justify-center overflow-hidden border-2 border-white/50
                                            `}>
                                                <img
                                                    src={getSilkImagePath(horse.silkId)}
                                                    alt={`${horse.name} silk`}
                                                    className="w-full h-full object-contain p-1"
                                                />
                                            </div>
                                        </div>

                                        {/* Horse Info */}
                                        <div className="text-center">
                                            <p className={`${isLeader ? 'text-sm' : 'text-xs'} font-bold text-[var(--text-primary)] truncate`}>
                                                {horse.name}
                                            </p>
                                            {isTop3 && (
                                                <p className="text-[10px] text-[var(--text-muted)] truncate">
                                                    {horse.jockeyName}
                                                </p>
                                            )}
                                        </div>

                                        {/* Gap - only for top 5 */}
                                        {index < 5 && (
                                            <div className="mt-2 text-center">
                                                <span className={`text-xs font-medium stat-value ${horse.gapToLeader === 0
                                                    ? 'text-amber-400'
                                                    : 'text-[var(--text-muted)]'
                                                    }`}>
                                                    {formatGap(horse.gapToLeader)}
                                                </span>
                                            </div>
                                        )}

                                        {/* Leader Crown */}
                                        {isLeader && (
                                            <div className="absolute top-1 right-1 text-xl">
                                                👑
                                            </div>
                                        )}
                                    </motion.div>
                                );
                            })}
                        </AnimatePresence>
                    </motion.div>
                </LayoutGroup>
            </div>
        </div>
    );
};
