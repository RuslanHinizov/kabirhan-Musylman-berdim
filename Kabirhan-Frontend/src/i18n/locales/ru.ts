// Russian translations - Русские переводы
export const ru = {
    // Common
    common: {
        save: 'Сохранить',
        cancel: 'Отмена',
        close: 'Закрыть',
        loading: 'Загрузка...',
        error: 'Ошибка',
        success: 'Успешно',
        online: 'Онлайн',
        offline: 'Офлайн',
        connecting: 'Подключение...',
        connected: 'Подключено',
        disconnected: 'Отключено',
    },

    // Header
    header: {
        operatorPanel: 'Панель оператора',
        viewer: 'Просмотр',
        horses: 'лошадей',
        lap: 'Круг',
        ready: 'Готов',
        live: 'Прямой эфир',
        finished: 'Завершено',
    },

    // Tabs
    tabs: {
        ptzControl: 'Управление PTZ',
        cameraGrid: 'Сетка камер',
        track: 'Трасса',
        rtspCameras: 'RTSP Камеры',
        raceSettings: 'Настройки скачек',
    },

    // Track View
    track: {
        liveRaceTrack: 'Трасса в реальном времени',
        stadium: 'стадион',
        camera: 'Камера',
        horseDetected: 'Лошадь обнаружена',
        startFinish: 'Старт/Финиш',
    },

    // Race Settings
    race: {
        raceConfiguration: 'Настройка скачек',
        raceName: 'Название скачек',
        totalLaps: 'Всего кругов',
        trackLength: 'Длина трассы',
        startFinishPosition: 'Позиция старта/финиша',
        startRace: 'Начать скачки',
        stopRace: 'Остановить скачки',
        resetRace: 'Сбросить скачки',
        addHorse: 'Добавить лошадь',
        removeHorse: 'Удалить лошадь',
        horseName: 'Имя лошади',
        horseNumber: 'Номер лошади',
        jockeyName: 'Имя жокея',
        color: 'Цвет',
    },

    // Camera Settings
    camera: {
        cameraConfiguration: 'Настройка камер',
        manageRtspStreams: 'Управление RTSP потоками',
        ptzCameras: 'PTZ Камеры',
        fixedCameras: 'Фиксированные камеры',
        active: 'активно',
        startStream: 'Запустить поток',
        stopStream: 'Остановить поток',
        streamServer: 'Сервер потока',
        requiresFfmpeg: 'Требуется установленный FFmpeg',
    },

    // Public Display
    display: {
        raceTime: 'Время скачек',
        distance: 'Дистанция',
        speed: 'Скорость',
        leader: 'Лидер',
        winner: 'Победитель',
        raceFinished: 'Скачки завершены!',
        connectingToCamera: 'Подключение к камере...',
        cameraOffline: 'Камера офлайн',
        gapToLeader: 'Отставание от лидера',
    },

    // Footer
    footer: {
        horseRacingSystem: 'Система трансляции скачек',
        switchToRealBackend: 'Переключить на реальный бэкенд',
        switchToMockMode: 'Переключить на тестовый режим',
        mockMode: 'Тестовый режим',
        backendConnected: 'Бэкенд подключен',
        connectionError: 'Ошибка подключения',
    },

    // PTZ Control
    ptz: {
        ptzCameraControl: 'Управление PTZ камерой',
        pan: 'Панорама',
        tilt: 'Наклон',
        zoom: 'Масштаб',
        preset: 'Пресет',
        home: 'Домой',
        selectCamera: 'Выбрать камеру',
    },

    // Camera Grid
    cameraGrid: {
        analyticsCameras: 'Аналитические камеры',
        cameraCount: 'камер — YOLO + CNN распознавание (зрителям не показывается)',
        horseDetected: 'лошадь обнаружена',
        horsesDetected: 'лошадей обнаружено',
        online: 'Онлайн',
        offline: 'Офлайн',
        horseDetectedLegend: 'Лошадь обнаружена',
    },

    // Camera Settings
    cameraSettings: {
        title: 'Настройка камер',
        description: 'Управление RTSP потоками камер',
        ptzBroadcast: 'PTZ камеры (Трансляция)',
        analyticsDetection: 'Аналитические камеры (Распознавание)',
        active: 'активно',
        backendServer: 'Сервер',
        rtspPlaceholder: 'rtsp://логин:пароль@ip:порт/поток',
        save: 'Сохранить',
        stopStream: 'Остановить поток',
        startStream: 'Запустить поток',
    },

    // PTZ Control Panel
    ptzPanel: {
        title: 'Управление PTZ камерой',
        description: 'Выберите активную камеру для публичной трансляции',
        onAir: 'В ЭФИРЕ',
        online: 'Онлайн',
        offline: 'Офлайн',
        position: 'Позиция',
        keyboardTip: 'Совет: Нажмите 1-{{count}} для быстрого переключения',
    },

    // Race Settings Page
    raceSettings: {
        title: 'Настройки скачек',
        description: 'Настройка параметров скачек и лошадей',
        raceConfiguration: 'Конфигурация скачек',
        raceName: 'Название скачек',
        totalLaps: 'Всего кругов',
        startFinishPosition: 'Позиция старта/финиша',
        raceControls: 'Управление скачками',
        startRace: 'Начать скачки',
        stopRace: 'Остановить скачки',
        reset: 'Сбросить',
        status: 'Статус',
        ready: 'Готов',
        racing: 'Скачки идут',
        finished: 'Завершено',
        settings: 'Настройки',
        saveSettings: 'Сохранить настройки',
        horses: 'Лошади',
        addHorse: 'Добавить лошадь',
        horseName: 'Имя лошади',
        jockeyName: 'Имя жокея',
        colorOverride: 'Изменить цвет',
        removeHorse: 'Удалить лошадь',
        noHorses: 'Лошади ещё не добавлены. Нажмите "Добавить лошадь" для начала.',
    },

    // PTZ Camera Display
    ptzDisplay: {
        noCameraFeed: 'Нет видеопотока',
        live: 'ПРЯМОЙ ЭФИР',
        hd: 'HD',
        race: 'Скачки',
        defaultRaceName: 'Большой чемпионат',
    },

    // Ranking Board
    ranking: {
        liveStandings: 'Текущий рейтинг',
        leader: 'Лидер:',
        waitingForRace: 'Ожидание заезда...',
    },

    // Stream Players
    stream: {
        connectingWebRTC: 'Подключение через WebRTC...',
        connectingStream: 'Подключение к потоку...',
        streamOffline: 'Поток офлайн',
        webrtcLabel: 'WebRTC',
    },
};
