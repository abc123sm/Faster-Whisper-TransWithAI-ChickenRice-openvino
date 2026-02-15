"""
Ultra-modern, dependency-free i18n module using JSON.

This is a lightweight, modern internationalization solution that:
- Uses JSON files (human-readable, easy to edit)
- No external dependencies
- Supports nested keys with dot notation
- Interpolation with {variable} syntax
- Pluralization support
- Lazy loading
- Type hints for better IDE support
"""

import os
import sys
import json
import locale
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum


class PluralForm(Enum):
    """Plural forms for different languages."""
    ZERO = "zero"
    ONE = "one"
    TWO = "two"
    FEW = "few"
    MANY = "many"
    OTHER = "other"


@dataclass
class LocaleInfo:
    """Information about a locale."""
    code: str
    language: str
    region: Optional[str] = None
    script: Optional[str] = None

    @property
    def language_code(self) -> str:
        """Get just the language part."""
        return self.language

    @property
    def full_code(self) -> str:
        """Get full locale code."""
        parts = [self.language]
        if self.script:
            parts.append(self.script)
        if self.region:
            parts.append(self.region)
        return '-'.join(parts)


class PluralRules:
    """Simplified plural rules for common languages."""

    @staticmethod
    def get_plural_form(locale_code: str, count: Union[int, float]) -> PluralForm:
        """
        Get the appropriate plural form for a count in a given locale.

        This is a simplified version of CLDR plural rules.
        """
        lang = locale_code.split('-')[0].lower()
        n = abs(count)

        # Languages with single form (Chinese, Japanese, Korean, Thai, etc.)
        if lang in ['zh', 'ja', 'ko', 'th', 'vi', 'id', 'ms']:
            return PluralForm.OTHER

        # English and Germanic languages
        if lang in ['en', 'de', 'nl', 'sv', 'da', 'no']:
            return PluralForm.ONE if n == 1 else PluralForm.OTHER

        # French, Portuguese, Spanish, Italian
        if lang in ['fr', 'pt', 'es', 'it']:
            if n == 0:
                return PluralForm.ZERO if lang == 'fr' else PluralForm.OTHER
            elif n == 1:
                return PluralForm.ONE
            else:
                return PluralForm.OTHER

        # Russian and Slavic languages (simplified)
        if lang in ['ru', 'uk', 'pl', 'cs', 'sk']:
            if n == 1:
                return PluralForm.ONE
            elif 2 <= n <= 4:
                return PluralForm.FEW
            else:
                return PluralForm.OTHER

        # Arabic (simplified)
        if lang == 'ar':
            if n == 0:
                return PluralForm.ZERO
            elif n == 1:
                return PluralForm.ONE
            elif n == 2:
                return PluralForm.TWO
            elif 3 <= n <= 10:
                return PluralForm.FEW
            elif 11 <= n <= 99:
                return PluralForm.MANY
            else:
                return PluralForm.OTHER

        # Default
        return PluralForm.OTHER


class ModernI18n:
    """
    Modern, lightweight i18n implementation using JSON.

    Features:
    - JSON-based translations (human-readable)
    - Nested key support with dot notation
    - Variable interpolation with {var} syntax
    - Smart pluralization
    - Locale auto-detection
    - Fallback chains
    - No external dependencies
    """

    def __init__(self,
                 locales_dir: Optional[Union[str, Path]] = None,
                 default_locale: str = 'zh-CN',
                 fallback_locale: str = 'en-US'):
        """
        Initialize the i18n system.

        Args:
            locales_dir: Directory containing JSON translation files
            default_locale: Default locale to use
            fallback_locale: Fallback locale for missing translations
        """
        self.locales_dir = Path(locales_dir or self._find_locales_dir())
        self.default_locale = default_locale
        self.fallback_locale = fallback_locale
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._current_locale: Optional[str] = None

        # Auto-detect and set locale
        detected = self._detect_locale()
        self.set_locale(detected)

    def _find_locales_dir(self) -> Path:
        """Find the locales directory."""
        # Check if running from PyInstaller bundle
        if getattr(sys, 'frozen', False):
            # Running from executable
            # sys._MEIPASS is the temporary folder where PyInstaller extracts files
            base_path = Path(sys._MEIPASS)
            possible_paths = [
                base_path / 'locales',
                Path(sys.executable).parent / 'locales',
            ]
        else:
            # Running from source
            possible_paths = [
                Path(__file__).parent.parent.parent / 'locales',
                Path(__file__).parent / 'locales',
                Path.cwd() / 'locales',
            ]

        for path in possible_paths:
            if path.exists() and path.is_dir():
                return path

        # Create default
        default_path = Path(__file__).parent.parent.parent / 'locales'
        default_path.mkdir(parents=True, exist_ok=True)
        return default_path

    def _detect_locale(self) -> str:
        """Auto-detect user's preferred locale."""
        # Environment variables
        for env_var in ['LANGUAGE', 'LANG', 'LC_ALL', 'LC_MESSAGES']:
            if lang := os.environ.get(env_var):
                return self._normalize_locale(lang.split(':')[0].split('.')[0])

        # System locale
        try:
            system_locale, _ = locale.getdefaultlocale()
            if system_locale:
                return self._normalize_locale(system_locale)
        except:
            pass

        # Windows-specific
        if sys.platform == 'win32':
            try:
                import ctypes
                lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
                locale_map = {
                    0x0804: 'zh-CN',
                    0x0404: 'zh-TW',
                    0x0409: 'en-US',
                    0x0411: 'ja-JP',
                    0x0412: 'ko-KR',
                }
                if lang_id in locale_map:
                    return locale_map[lang_id]
            except:
                pass

        return self.default_locale

    def _normalize_locale(self, locale_code: str) -> str:
        """Normalize locale code to standard format."""
        if not locale_code:
            return self.default_locale

        # Replace underscores
        locale_code = locale_code.replace('_', '-')

        # Add default region if needed
        if '-' not in locale_code:
            defaults = {
                'zh': 'zh-CN',
                'en': 'en-US',
                'ja': 'ja-JP',
                'ko': 'ko-KR',
                'es': 'es-ES',
                'fr': 'fr-FR',
                'de': 'de-DE',
                'it': 'it-IT',
                'pt': 'pt-BR',
                'ru': 'ru-RU',
            }
            locale_code = defaults.get(locale_code.lower(), locale_code)

        return locale_code

    @lru_cache(maxsize=10)
    def _load_translations(self, locale_code: str) -> Dict[str, Any]:
        """Load translations for a locale (cached)."""
        translations = {}

        # Try JSON file
        json_path = self.locales_dir / locale_code / 'messages.json'
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}", file=sys.stderr)

        return translations

    def set_locale(self, locale_code: str):
        """Set the current locale."""
        self._current_locale = self._normalize_locale(locale_code)
        # Pre-load translations
        self._translations[self._current_locale] = self._load_translations(self._current_locale)
        if self.fallback_locale != self._current_locale:
            self._translations[self.fallback_locale] = self._load_translations(self.fallback_locale)

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get value from nested dict using dot notation."""
        keys = key.split('.')
        value = data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return None
            else:
                return None

        return value

    def _interpolate(self, template: str, variables: Dict[str, Any]) -> str:
        """Interpolate variables in template string."""
        if not isinstance(template, str):
            return str(template)

        # Match {variable_name} or {variable_name:format}
        pattern = r'\{(\w+)(?::([^}]+))?\}'

        def replacer(match):
            var_name = match.group(1)
            format_spec = match.group(2)

            if var_name not in variables:
                return match.group(0)  # Keep original if variable not found

            value = variables[var_name]

            # Apply format if specified
            if format_spec:
                try:
                    if format_spec.endswith('f'):
                        # Float formatting like {value:0.2f}
                        decimals = int(format_spec[:-1].split('.')[-1]) if '.' in format_spec else 0
                        return f"{float(value):.{decimals}f}"
                    elif format_spec.isdigit():
                        # Padding like {value:5}
                        return str(value).zfill(int(format_spec))
                except:
                    pass

            return str(value)

        return re.sub(pattern, replacer, template)

    def get(self, key: str, **variables) -> str:
        """
        Get a translated string.

        Args:
            key: Translation key (supports dot notation)
            **variables: Variables for interpolation

        Returns:
            Translated and interpolated string
        """
        # Handle pluralization
        if 'count' in variables:
            plural_key = self._get_plural_key(key, variables['count'])
            result = self._get_translation(plural_key)
            if result is not None and result != plural_key:
                return self._interpolate(result, variables)

        # Regular translation
        result = self._get_translation(key)

        # Fallback to key if not found
        if result is None:
            result = key

        # Interpolate variables
        if variables:
            result = self._interpolate(result, variables)

        return result

    def _get_plural_key(self, base_key: str, count: Union[int, float]) -> str:
        """Get the plural form key."""
        plural_form = PluralRules.get_plural_form(self._current_locale, count)
        return f"{base_key}.{plural_form.value}"

    def _get_translation(self, key: str) -> Optional[str]:
        """Get translation from current or fallback locale."""
        # Try current locale
        if self._current_locale in self._translations:
            value = self._get_nested_value(self._translations[self._current_locale], key)
            if value is not None:
                return value

        # Try fallback locale
        if self.fallback_locale in self._translations:
            value = self._get_nested_value(self._translations[self.fallback_locale], key)
            if value is not None:
                return value

        return None

    def format_duration(self, seconds: float) -> str:
        """Format duration in a localized way."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if hours > 0:
            return self.get('time.duration_hours', hours=hours, minutes=minutes, seconds=secs)
        elif minutes > 0:
            return self.get('time.duration_minutes', minutes=minutes, seconds=secs)
        else:
            return self.get('time.duration_seconds', seconds=secs)

    def format_percentage(self, value: float, decimals: int = 1) -> str:
        """Format percentage in a localized way."""
        return self.get('format.percentage', value=value * 100, decimals=decimals)

    def format_file_count(self, count: int) -> str:
        """Format file count with proper pluralization."""
        return self.get('files.count', count=count)

    @property
    def current_locale(self) -> str:
        """Get current locale."""
        return self._current_locale

    @property
    def available_locales(self) -> List[str]:
        """Get list of available locales."""
        locales = []
        if self.locales_dir.exists():
            for path in self.locales_dir.iterdir():
                if path.is_dir() and (path / 'messages.json').exists():
                    locales.append(path.name)
        return sorted(locales)

    def has_key(self, key: str) -> bool:
        """Check if a translation key exists."""
        return self._get_translation(key) is not None

    def get_all_keys(self) -> List[str]:
        """Get all available translation keys."""
        keys = set()

        def extract_keys(data: Dict[str, Any], prefix: str = ''):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    extract_keys(value, full_key)
                else:
                    keys.add(full_key)

        for locale_code in [self._current_locale, self.fallback_locale]:
            if locale_code in self._translations:
                extract_keys(self._translations[locale_code])

        return sorted(keys)


# Global instance
_i18n: Optional[ModernI18n] = None


def init(locales_dir: Optional[Union[str, Path]] = None,
         default_locale: str = 'zh-CN',
         fallback_locale: str = 'en-US') -> ModernI18n:
    """Initialize the global i18n instance."""
    global _i18n
    _i18n = ModernI18n(locales_dir, default_locale, fallback_locale)
    return _i18n


def get_i18n() -> ModernI18n:
    """Get the global i18n instance."""
    global _i18n
    if _i18n is None:
        _i18n = init()
    return _i18n


# Convenience functions
def _(key: str, **variables) -> str:
    """Get translated string."""
    return get_i18n().get(key, **variables)


def set_locale(locale_code: str):
    """Set current locale."""
    get_i18n().set_locale(locale_code)


def get_locale() -> str:
    """Get current locale."""
    return get_i18n().current_locale


def available_locales() -> List[str]:
    """Get available locales."""
    return get_i18n().available_locales


# Format helpers
format_duration = lambda s: get_i18n().format_duration(s)
format_percentage = lambda v, d=1: get_i18n().format_percentage(v, d)
format_file_count = lambda c: get_i18n().format_file_count(c)

# Auto-initialize
init()