// Kazakh translations - Қазақша аудармалар
export const kk = {
    // Common
    common: {
        save: 'Сақтау',
        cancel: 'Бас тарту',
        close: 'Жабу',
        loading: 'Жүктелуде...',
        error: 'Қате',
        success: 'Сәтті',
        online: 'Онлайн',
        offline: 'Офлайн',
        connecting: 'Қосылуда...',
        connected: 'Қосылды',
        disconnected: 'Ажыратылды',
    },

    // Header
    header: {
        operatorPanel: 'Оператор панелі',
        viewer: 'Көрермен',
        horses: 'ат',
        lap: 'Айналым',
        ready: 'Дайын',
        live: 'Тікелей эфир',
        finished: 'Аяқталды',
    },

    // Tabs
    tabs: {
        ptzControl: 'PTZ басқару',
        cameraGrid: 'Камера торы',
        track: 'Жарыс жолы',
        rtspCameras: 'RTSP Камералар',
        raceSettings: 'Жарыс параметрлері',
    },

    // Track View
    track: {
        liveRaceTrack: 'Тікелей жарыс жолы',
        stadium: 'стадион',
        camera: 'Камера',
        horseDetected: 'Ат анықталды',
        startFinish: 'Старт/Финиш',
    },

    // Race Settings
    race: {
        raceConfiguration: 'Жарыс конфигурациясы',
        raceName: 'Жарыс атауы',
        totalLaps: 'Жалпы айналымдар',
        trackLength: 'Жол ұзындығы',
        startFinishPosition: 'Старт/Финиш позициясы',
        startRace: 'Жарысты бастау',
        stopRace: 'Жарысты тоқтату',
        resetRace: 'Жарысты қалпына келтіру',
        addHorse: 'Ат қосу',
        removeHorse: 'Атты алып тастау',
        horseName: 'Ат аты',
        horseNumber: 'Ат нөмірі',
        jockeyName: 'Шабандоз аты',
        color: 'Түс',
    },

    // Camera Settings
    camera: {
        cameraConfiguration: 'Камера конфигурациясы',
        manageRtspStreams: 'RTSP ағындарын басқару',
        ptzCameras: 'PTZ Камералар',
        fixedCameras: 'Тұрақты камералар',
        active: 'белсенді',
        startStream: 'Ағынды бастау',
        stopStream: 'Ағынды тоқтату',
        streamServer: 'Ағын сервері',
        requiresFfmpeg: 'Жүйеде FFmpeg орнатылған болуы керек',
    },

    // Public Display
    display: {
        raceTime: 'Жарыс уақыты',
        distance: 'Қашықтық',
        speed: 'Жылдамдық',
        leader: 'Көшбасшы',
        winner: 'Жеңімпаз',
        raceFinished: 'Жарыс аяқталды!',
        connectingToCamera: 'Камераға қосылуда...',
        cameraOffline: 'Камера офлайн',
        gapToLeader: 'Көшбасшыдан артта қалу',
    },

    // Footer
    footer: {
        horseRacingSystem: 'Ат жарысын тарату жүйесі',
        switchToRealBackend: 'Нақты бэкендке ауысу',
        switchToMockMode: 'Тест режиміне ауысу',
        mockMode: 'Тест режимі',
        backendConnected: 'Бэкенд қосылды',
        connectionError: 'Қосылу қатесі',
    },

    // PTZ Control
    ptz: {
        ptzCameraControl: 'PTZ камера басқару',
        pan: 'Панорама',
        tilt: 'Көлбеу',
        zoom: 'Масштаб',
        preset: 'Алдын ала орнату',
        home: 'Басты бет',
        selectCamera: 'Камераны таңдау',
    },

    // Camera Grid
    cameraGrid: {
        analyticsCameras: 'Аналитикалық камералар',
        cameraCount: 'камера — YOLO + CNN анықтау (көрерменге көрсетілмейді)',
        horseDetected: 'ат анықталды',
        horsesDetected: 'ат анықталды',
        online: 'Онлайн',
        offline: 'Офлайн',
        horseDetectedLegend: 'Ат анықталды',
    },

    // Camera Settings
    cameraSettings: {
        title: 'Камера конфигурациясы',
        description: 'RTSP камера ағындарын басқару',
        ptzBroadcast: 'PTZ камералар (Тарату)',
        analyticsDetection: 'Аналитикалық камералар (Анықтау)',
        active: 'белсенді',
        backendServer: 'Сервер',
        rtspPlaceholder: 'rtsp://логин:құпиясөз@ip:порт/ағын',
        save: 'Сақтау',
        stopStream: 'Ағынды тоқтату',
        startStream: 'Ағынды бастау',
    },

    // PTZ Control Panel
    ptzPanel: {
        title: 'PTZ камера басқаруы',
        description: 'Жалпыға ортақ дисплей үшін белсенді тарату камерасын таңдаңыз',
        onAir: 'ЭФИРДЕ',
        online: 'Онлайн',
        offline: 'Офлайн',
        position: 'Позиция',
        keyboardTip: 'Кеңес: Жылдам ауыстыру үшін 1-{{count}} пернелерін басыңыз',
    },

    // Race Settings Page
    raceSettings: {
        title: 'Жарыс параметрлері',
        description: 'Жарыс параметрлері мен аттарды баптау',
        raceConfiguration: 'Жарыс конфигурациясы',
        raceName: 'Жарыс атауы',
        totalLaps: 'Жалпы айналымдар',
        startFinishPosition: 'Старт/Финиш позициясы',
        raceControls: 'Жарыс басқаруы',
        startRace: 'Жарысты бастау',
        stopRace: 'Жарысты тоқтату',
        reset: 'Қалпына келтіру',
        status: 'Күй',
        ready: 'Дайын',
        racing: 'Жарыс жүріп жатыр',
        finished: 'Аяқталды',
        settings: 'Параметрлер',
        saveSettings: 'Параметрлерді сақтау',
        horses: 'Аттар',
        addHorse: 'Ат қосу',
        horseName: 'Ат аты',
        jockeyName: 'Шабандоз аты',
        colorOverride: 'Түсті өзгерту',
        removeHorse: 'Атты жою',
        noHorses: 'Әлі ат қосылмады. Бастау үшін "Ат қосу" батырмасын басыңыз.',
    },

    // PTZ Camera Display
    ptzDisplay: {
        noCameraFeed: 'Бейне ағыны жоқ',
        live: 'ТІКЕЛЕЙ ЭФИР',
        hd: 'HD',
        race: 'Жарыс',
        defaultRaceName: 'Үлкен чемпионат',
    },

    // Ranking Board
    ranking: {
        liveStandings: 'Ағымдағы рейтинг',
        leader: 'Көшбасшы:',
    },

    // Stream Players
    stream: {
        connectingWebRTC: 'WebRTC арқылы қосылуда...',
        connectingStream: 'Ағынға қосылуда...',
        streamOffline: 'Ағын офлайн',
        webrtcLabel: 'WebRTC',
    },
};
