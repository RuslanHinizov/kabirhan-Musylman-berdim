import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Camera, Grid3X3, Map, Settings, ExternalLink, Video, Wifi } from 'lucide-react';
import { PTZControlPanel } from '../components/operator/PTZControlPanel';
import { CameraGrid } from '../components/operator/CameraGrid';
import { Track2DView } from '../components/operator/Track2DView';
import { RaceSettings } from '../components/operator/RaceSettings';
import { CameraSettings } from '../components/operator/CameraSettings';
import { LanguageSelector } from '../components/LanguageSelector';
import { useCameraStore } from '../store/cameraStore';
import { useRaceStore } from '../store/raceStore';
import {
    connectToBackend,
    disconnectFromBackend,
    setConnectionStatusCallback
} from '../services/backendConnection';

type TabId = 'ptz' | 'grid' | 'track' | 'cameras' | 'settings';
type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export const OperatorPanel = () => {
    const { t } = useTranslation();
    const [activeTab, setActiveTab] = useState<TabId>('ptz');
    const [backendStatus, setBackendStatus] = useState<ConnectionStatus>('disconnected');
    const { initializeCameras } = useCameraStore();
    const { initializeDefaultRace, race, rankings, setBackendConnected } = useRaceStore();

    const tabs = [
        { id: 'ptz' as TabId, label: t('tabs.ptzControl'), icon: Camera },
        { id: 'grid' as TabId, label: t('tabs.cameraGrid'), icon: Grid3X3 },
        { id: 'track' as TabId, label: t('tabs.track'), icon: Map },
        { id: 'cameras' as TabId, label: t('tabs.rtspCameras'), icon: Video },
        { id: 'settings' as TabId, label: t('tabs.raceSettings'), icon: Settings },
    ];

    // Initialize and connect to backend
    useEffect(() => {
        initializeCameras();
        initializeDefaultRace();

        setConnectionStatusCallback((status) => {
            setBackendStatus(status);
            setBackendConnected(status === 'connected');
        });

        // Connect to real backend directly
        connectToBackend();

        return () => {
            disconnectFromBackend();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement) return;
            const key = parseInt(e.key);
            if (key >= 1 && key <= 5) {
                setActiveTab(tabs[key - 1].id);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const renderContent = () => {
        switch (activeTab) {
            case 'ptz': return <PTZControlPanel />;
            case 'grid': return <CameraGrid />;
            case 'track': return <Track2DView />;
            case 'cameras': return <CameraSettings />;
            case 'settings': return <RaceSettings />;
        }
    };

    const getStatusColor = () => {
        switch (backendStatus) {
            case 'connected': return 'bg-[var(--accent)]';
            case 'connecting': return 'bg-yellow-500 animate-pulse';
            case 'error': return 'bg-red-500';
            default: return 'bg-gray-500';
        }
    };

    const getStatusText = () => {
        switch (backendStatus) {
            case 'connected': return t('footer.backendConnected');
            case 'connecting': return t('common.connecting');
            case 'error': return t('footer.connectionError');
            default: return t('common.disconnected');
        }
    };

    const getStatusLabel = () => {
        if (race.status === 'pending') return t('header.ready');
        if (race.status === 'active') return t('header.live');
        return t('header.finished');
    };

    return (
        <div className="h-screen w-screen flex flex-col bg-[var(--background)]">
            {/* Header */}
            <header className="flex-shrink-0 border-b border-[var(--border)] bg-[var(--surface)]">
                <div className="px-6 py-4 flex items-center justify-between">
                    {/* Left */}
                    <div className="flex items-center gap-4">
                        <h1 className="text-lg font-semibold text-[var(--text-primary)]">
                            {t('header.operatorPanel')}
                        </h1>
                        <span className="text-sm text-[var(--text-muted)]">•</span>
                        <span className="text-sm text-[var(--text-secondary)]">{race.name}</span>

                        <span className={`
              px-2.5 py-1 rounded text-xs font-medium
              ${race.status === 'pending' ? 'bg-[var(--surface-light)] text-[var(--text-muted)]' :
                                race.status === 'active' ? 'bg-[var(--accent)] text-white' :
                                    'bg-[var(--warning)] text-black'}
            `}>
                            {getStatusLabel()}
                        </span>
                    </div>

                    {/* Right */}
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-3 text-sm text-[var(--text-muted)]">
                            <span>{rankings.length} {t('header.horses')}</span>
                            <span>•</span>
                            <span>{t('header.lap')} {rankings[0]?.currentLap || 1}/{race.totalLaps}</span>
                        </div>

                        <LanguageSelector />

                        <a
                            href="/"
                            target="_blank"
                            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-[var(--text-secondary)] 
                         hover:text-[var(--text-primary)] border border-[var(--border)] hover:border-[var(--text-muted)] transition-colors"
                        >
                            <ExternalLink className="w-4 h-4" />
                            {t('header.viewer')}
                        </a>
                    </div>
                </div>

                {/* Tabs */}
                <div className="px-6 flex gap-1">
                    {tabs.map((tab, i) => {
                        const Icon = tab.icon;
                        const isActive = activeTab === tab.id;

                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`
                  flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-t-lg transition-colors cursor-pointer
                  ${isActive ? 'tab-active' : 'tab-inactive'}
                `}
                            >
                                <Icon className="w-4 h-4" />
                                {tab.label}
                                <span className="text-[10px] text-[var(--text-muted)] ml-1">{i + 1}</span>
                            </button>
                        );
                    })}
                </div>
            </header>

            {/* Content */}
            <main className="flex-1 overflow-auto">
                {renderContent()}
            </main>

            {/* Footer */}
            <footer className="flex-shrink-0 border-t border-[var(--border)] px-6 py-2">
                <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
                    <span>{t('footer.horseRacingSystem')}</span>
                    <div className="flex items-center gap-2">
                        <Wifi className="w-3 h-3" />
                        <span className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
                        <span>{getStatusText()}</span>
                    </div>
                </div>
            </footer>
        </div>
    );
};
