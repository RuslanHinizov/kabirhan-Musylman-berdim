import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import { en } from './locales/en';
import { tr } from './locales/tr';
import { ru } from './locales/ru';
import { kk } from './locales/kk';

const SUPPORTED_LANGS = ['en', 'tr', 'ru', 'kk'];

// Get saved language or detect from browser
const getSavedLanguage = (): string => {
    const saved = localStorage.getItem('language');
    if (saved && SUPPORTED_LANGS.includes(saved)) {
        return saved;
    }

    // Detect from browser
    const browserLang = navigator.language.split('-')[0];
    if (SUPPORTED_LANGS.includes(browserLang)) {
        return browserLang;
    }

    return 'en'; // Default to English
};

i18n
    .use(initReactI18next)
    .init({
        resources: {
            en: { translation: en },
            tr: { translation: tr },
            ru: { translation: ru },
            kk: { translation: kk },
        },
        lng: getSavedLanguage(),
        fallbackLng: 'en',
        interpolation: {
            escapeValue: false, // React already escapes
        },
    });

// Save language preference and sync across tabs
export const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang);
    localStorage.setItem('language', lang);
};

// Listen for language changes from other tabs/windows
if (typeof window !== 'undefined') {
    window.addEventListener('storage', (event) => {
        if (event.key === 'language' && event.newValue) {
            const newLang = event.newValue;
            if (SUPPORTED_LANGS.includes(newLang) && i18n.language !== newLang) {
                console.log(`🌍 Language changed in another tab: ${newLang}`);
                i18n.changeLanguage(newLang);
            }
        }
    });
}

// Available languages
export const languages = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'tr', name: 'Türkçe', flag: '🇹🇷' },
    { code: 'ru', name: 'Русский', flag: '🇷🇺' },
    { code: 'kk', name: 'Қазақша', flag: '🇰🇿' },
];

export default i18n;

