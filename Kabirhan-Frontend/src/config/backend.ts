const BACKEND_PORT = 8000;

const getBackendHost = (): string => {
    if (typeof window === 'undefined') {
        return '127.0.0.1';
    }

    const host = window.location.hostname;
    if (!host || host === 'localhost' || host === '::1') {
        return '127.0.0.1';
    }
    return host;
};

const getHttpProtocol = (): 'http:' | 'https:' => {
    if (typeof window !== 'undefined' && window.location.protocol === 'https:') {
        return 'https:';
    }
    return 'http:';
};

const getWsProtocol = (): 'ws:' | 'wss:' => (getHttpProtocol() === 'https:' ? 'wss:' : 'ws:');

export const BACKEND_HTTP_URL = `${getHttpProtocol()}//${getBackendHost()}:${BACKEND_PORT}`;
export const BACKEND_WS_URL = `${getWsProtocol()}//${getBackendHost()}:${BACKEND_PORT}/ws`;

