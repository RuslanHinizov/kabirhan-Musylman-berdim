import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Plus, Trash2, Play, Square, RotateCcw, Save } from 'lucide-react';
import { useRaceStore } from '../../store/raceStore';
import { HORSE_COLORS, HORSE_NAMES, JOCKEY_NAMES, MAX_HORSES, TRACK_LENGTH, TOTAL_SILKS } from '../../types';
import { getSilkImagePath, getSilkColor, getDefaultSilkId } from '../../utils/silkUtils';
import { sendToBackend } from '../../services/backendConnection';

export const RaceSettings = () => {
    const { t } = useTranslation();
    const {
        race, horses, setRace, addHorse, removeHorse, updateHorse,
        startRace, stopRace, resetRace, backendConnected
    } = useRaceStore();
    const [raceName, setRaceName] = useState(race.name);
    const [totalLaps, setTotalLaps] = useState(race.totalLaps);
    const [startFinishPosition, setStartFinishPosition] = useState(race.startFinishPosition);

    useEffect(() => {
        setRaceName(race.name);
        setTotalLaps(race.totalLaps);
        setStartFinishPosition(race.startFinishPosition);
    }, [race]);

    const generateId = () => Math.random().toString(36).substr(2, 9);

    const handleAddHorse = () => {
        if (horses.length >= MAX_HORSES) return;
        const newNumber = horses.length + 1;
        const silkId = getDefaultSilkId(newNumber);

        addHorse({
            id: generateId(),
            name: HORSE_NAMES[(newNumber - 1) % HORSE_NAMES.length],
            number: newNumber,
            color: HORSE_COLORS[(newNumber - 1) % HORSE_COLORS.length],
            jockeyName: JOCKEY_NAMES[(newNumber - 1) % JOCKEY_NAMES.length],
            currentPosition: newNumber,
            currentLap: 1,
            timeElapsed: 0,
            gapToLeader: 0,
            distanceCovered: 0,
            speed: 0,
            lastCameraId: 'cam-0',
            silkId: silkId,
            silkColor: getSilkColor(silkId),
        });
    };

    const handleSaveSettings = () => {
        setRace({ name: raceName, totalLaps, startFinishPosition });

        if (!backendConnected) return;

        sendToBackend({
            type: 'set_race_config',
            race: {
                name: raceName,
                totalLaps,
                trackLength: race.trackLength,
                startFinishPosition,
            },
        });
        sendToBackend({
            type: 'set_horses',
            horses: horses.map((horse) => ({
                id: horse.id,
                number: horse.number,
                name: horse.name,
                color: horse.color,
                jockeyName: horse.jockeyName,
                silkId: horse.silkId,
            })),
        });
    };

    const handleStartFinishChange = (value: number) => {
        setStartFinishPosition(value);
        setRace({ startFinishPosition: value });
    };

    const handleSilkChange = (horseId: string, newSilkId: number) => {
        updateHorse(horseId, {
            silkId: newSilkId,
            silkColor: getSilkColor(newSilkId)
        });
    };

    const handleStartRace = () => {
        startRace();
        if (backendConnected) {
            sendToBackend({
                type: 'set_race_config',
                race: {
                    name: raceName,
                    totalLaps,
                    trackLength: race.trackLength,
                    startFinishPosition,
                },
            });
            sendToBackend({
                type: 'set_horses',
                horses: horses.map((horse) => ({
                    id: horse.id,
                    number: horse.number,
                    name: horse.name,
                    color: horse.color,
                    jockeyName: horse.jockeyName,
                    silkId: horse.silkId,
                })),
            });
            sendToBackend({ type: 'start_race' });
        }
    };

    const handleStopRace = () => {
        stopRace();
        if (backendConnected) {
            sendToBackend({ type: 'stop_race' });
        }
    };

    const handleResetRace = () => {
        resetRace();
        if (!backendConnected) return;

        const { race: resetRaceState, horses: resetHorses } = useRaceStore.getState();
        sendToBackend({ type: 'reset_race' });
        sendToBackend({
            type: 'set_race_config',
            race: {
                name: resetRaceState.name,
                totalLaps: resetRaceState.totalLaps,
                trackLength: resetRaceState.trackLength,
                startFinishPosition: resetRaceState.startFinishPosition,
            },
        });
        sendToBackend({
            type: 'set_horses',
            horses: resetHorses.map((horse) => ({
                id: horse.id,
                number: horse.number,
                name: horse.name,
                color: horse.color,
                jockeyName: horse.jockeyName,
                silkId: horse.silkId,
            })),
        });
    };

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="mb-6">
                <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-1">{t('raceSettings.title')}</h2>
                <p className="text-sm text-[var(--text-muted)]">{t('raceSettings.description')}</p>
            </div>

            {/* Two Column Layout */}
            <div className="grid grid-cols-2 gap-6">
                {/* Left Column - Race Config */}
                <div className="space-y-5">
                    <div className="racing-card p-5">
                        <h3 className="text-sm font-medium text-[var(--text-primary)] mb-4">{t('raceSettings.raceConfiguration')}</h3>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-xs text-[var(--text-muted)] mb-1.5">{t('raceSettings.raceName')}</label>
                                <input type="text" value={raceName} onChange={(e) => setRaceName(e.target.value)} />
                            </div>

                            <div>
                                <label className="block text-xs text-[var(--text-muted)] mb-1.5">{t('raceSettings.totalLaps')}</label>
                                <input type="number" min="1" max="10" value={totalLaps} onChange={(e) => setTotalLaps(Number(e.target.value))} />
                            </div>

                            <div className="pt-4 border-t border-[var(--border)]">
                                <label className="block text-xs text-[var(--text-muted)] mb-2">{t('raceSettings.startFinishPosition')}</label>
                                <input
                                    type="range" min="0" max={TRACK_LENGTH} step="100"
                                    value={startFinishPosition}
                                    onChange={(e) => handleStartFinishChange(Number(e.target.value))}
                                    className="w-full"
                                />
                                <div className="flex justify-between mt-1 text-[10px] text-[var(--text-muted)]">
                                    <span>0m</span>
                                    <span className="font-medium text-[var(--text-secondary)]">{startFinishPosition}m</span>
                                    <span>{TRACK_LENGTH}m</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Race Controls */}
                    <div className="racing-card p-5">
                        <h3 className="text-sm font-medium text-[var(--text-primary)] mb-4">{t('raceSettings.raceControls')}</h3>

                        <div className="flex flex-wrap items-center gap-3">
                            <button
                                onClick={handleStartRace}
                                disabled={race.status === 'active'}
                                className="racing-button flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <Play className="w-4 h-4" />
                                {t('raceSettings.startRace')}
                            </button>

                            <button
                                onClick={handleStopRace}
                                disabled={race.status !== 'active'}
                                className="racing-button flex items-center gap-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <Square className="w-4 h-4" />
                                {t('raceSettings.stopRace')}
                            </button>

                            <button
                                onClick={handleResetRace}
                                className="racing-button flex items-center gap-2 bg-gray-600 hover:bg-gray-700"
                            >
                                <RotateCcw className="w-4 h-4" />
                                {t('raceSettings.reset')}
                            </button>
                        </div>

                        <div className="mt-4 pt-4 border-t border-[var(--border)] flex items-center justify-between">
                            <span className="text-sm text-[var(--text-muted)]">{t('raceSettings.status')}</span>
                            <span className={`
                                px-3 py-1.5 rounded text-xs font-medium
                                ${race.status === 'pending' ? 'bg-[var(--surface-light)] text-[var(--text-muted)]' :
                                    race.status === 'active' ? 'bg-green-500 text-white' :
                                        'bg-amber-500 text-black'}
                            `}>
                                {race.status === 'pending' ? t('raceSettings.ready') : race.status === 'active' ? t('raceSettings.racing') : t('raceSettings.finished')}
                            </span>
                        </div>
                    </div>

                    {/* Save Settings */}
                    <div className="racing-card p-5">
                        <h3 className="text-sm font-medium text-[var(--text-primary)] mb-4">{t('raceSettings.settings')}</h3>
                        <button
                            onClick={handleSaveSettings}
                            className="racing-button flex items-center gap-2 w-full justify-center"
                        >
                            <Save className="w-4 h-4" />
                            {t('raceSettings.saveSettings')}
                        </button>
                    </div>
                </div>

                {/* Right Column - Horses */}
                <div className="racing-card p-5 flex flex-col h-fit max-h-[calc(100vh-200px)]">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-medium text-[var(--text-primary)]">
                            {t('raceSettings.horses')} ({horses.length}/{MAX_HORSES})
                        </h3>
                        <button
                            onClick={handleAddHorse}
                            disabled={horses.length >= MAX_HORSES}
                            className="racing-button text-xs py-2 px-3 flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Plus className="w-3.5 h-3.5" />
                            {t('raceSettings.addHorse')}
                        </button>
                    </div>

                    <div className="space-y-2 overflow-y-auto flex-1">
                        {horses.map((horse, index) => (
                            <div key={horse.id} className="flex items-center gap-2 p-3 rounded-lg bg-[var(--background)] border border-[var(--border)]">
                                {/* Position Number */}
                                <span className="text-xs text-[var(--text-muted)] w-5 flex-shrink-0">#{index + 1}</span>

                                {/* Silk Preview */}
                                <div className="w-10 h-12 rounded bg-white flex items-center justify-center flex-shrink-0 border border-[var(--border)]">
                                    <img
                                        src={getSilkImagePath(horse.silkId)}
                                        alt={`Silk ${horse.silkId}`}
                                        className="w-8 h-10 object-contain"
                                    />
                                </div>

                                {/* Silk Selector */}
                                <select
                                    value={horse.silkId || 1}
                                    onChange={(e) => handleSilkChange(horse.id, Number(e.target.value))}
                                    className="w-16 text-xs py-1.5 px-1 flex-shrink-0 bg-[var(--surface)] border border-[var(--border)] rounded"
                                >
                                    {Array.from({ length: TOTAL_SILKS }, (_, i) => i + 1).map(id => (
                                        <option key={id} value={id}>#{id}</option>
                                    ))}
                                </select>

                                {/* Horse Name */}
                                <div className="flex-1 min-w-0">
                                    <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">{t('raceSettings.horseName')}</label>
                                    <input
                                        type="text" value={horse.name}
                                        onChange={(e) => updateHorse(horse.id, { name: e.target.value })}
                                        className="w-full text-sm py-1 px-2"
                                        placeholder={t('raceSettings.horseName')}
                                    />
                                </div>

                                {/* Jockey Name */}
                                <div className="flex-1 min-w-0">
                                    <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">{t('raceSettings.jockeyName')}</label>
                                    <input
                                        type="text" value={horse.jockeyName}
                                        onChange={(e) => updateHorse(horse.id, { jockeyName: e.target.value })}
                                        className="w-full text-sm py-1 px-2"
                                        placeholder={t('raceSettings.jockeyName')}
                                    />
                                </div>

                                {/* Color Override */}
                                <input
                                    type="color" value={horse.color}
                                    onChange={(e) => updateHorse(horse.id, { color: e.target.value })}
                                    className="w-7 h-7 rounded cursor-pointer flex-shrink-0"
                                    title={t('raceSettings.colorOverride')}
                                />

                                {/* Delete Button */}
                                <button
                                    onClick={() => removeHorse(horse.id)}
                                    className="p-1.5 text-[var(--text-muted)] hover:text-[var(--danger)] transition-colors cursor-pointer flex-shrink-0"
                                    title={t('raceSettings.removeHorse')}
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </div>
                        ))}

                        {horses.length === 0 && (
                            <p className="text-center py-12 text-sm text-[var(--text-muted)]">
                                {t('raceSettings.noHorses')}
                            </p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
