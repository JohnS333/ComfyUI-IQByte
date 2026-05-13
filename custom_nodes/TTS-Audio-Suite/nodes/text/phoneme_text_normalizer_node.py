"""
📝 Phoneme Text Normalizer Node

A universal text preprocessing node for multilingual TTS that handles special characters,
phonemization, and text normalization for languages like Polish, German, French, Spanish, etc.

This node provides various text transformation methods to make multilingual text compatible
with TTS models that may have been trained on different character sets or phoneme representations.
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Optional

class PhonemeTextNormalizer:
    """
    Universal text normalizer for multilingual TTS preprocessing.
    Handles special characters, phonemization, and text transformations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Enter your text here. Examples:\nPolish: Cześć, jak się masz?\nGerman: Schöne Grüße aus München!\nFrench: Bonjour, comment allez-vous?",
                    "tooltip": "Input text to normalize for TTS processing.\n\nSupports multilingual text with special characters like:\n• Polish: ą, ć, ę, ł, ń, ó, ś, ź, ż\n• German: ä, ö, ü, ß\n• French: à, é, ê, ç, etc.\n• And many other languages\n\nConnect this to any TTS engine for improved pronunciation."
                }),
                "method": (["Pass-through", "Unicode Decomposition", "IPA Phonemization", "Character Mapping"], {
                    "default": "Unicode Decomposition",
                    "tooltip": "Text processing method to apply:\n\n• Pass-through: No processing (original text)\n• Unicode Decomposition: ą→a̧, ć→ć (recommended for most cases)\n• IPA Phonemization: Full phonetic conversion (ą→ɔ̃, requires espeak)\n• Character Mapping: ASCII fallback (ą→a, ć→c)\n\nStart with Unicode Decomposition - it fixes most pronunciation issues."
                }),
                "language": (["Auto-detect", "Polish", "German", "French", "Spanish", "Portuguese", "Italian", "Czech", "Slovak", "Hungarian", "Norwegian", "Swedish", "Danish", "Finnish", "Dutch", "English"], {
                    "default": "Auto-detect",
                    "tooltip": "Language for processing (affects IPA Phonemization):\n\n• Auto-detect: Automatically detects language from text\n• Manual selection: Choose specific language for better accuracy\n\nLanguage detection looks for special characters:\n• Polish: ą, ę, ć, ł, etc.\n• German: ä, ö, ü, ß\n• French: à, é, ê, ç, etc.\n\nOnly affects IPA Phonemization method."
                }),
                "show_debug": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Show Before/After",
                    "label_off": "Hide Debug Info",
                    "tooltip": "Show debug information in console:\n\n• Original vs normalized text comparison\n• Character-by-character changes\n• Detected language (when auto-detecting)\n• Processing method used\n\nHelpful for testing which method works best for your language.\nDisable for cleaner console output in production."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("normalized_text",)
    FUNCTION = "normalize_text"
    CATEGORY = "TTS Audio Suite/Text"

    # Unicode decomposition mappings for common special characters
    DECOMPOSITION_MAP = {
        # Polish
        "ą": "a\u0328", "ę": "e\u0328", "ć": "c\u0301", "ń": "n\u0301",
        "ś": "s\u0301", "ź": "z\u0301", "ż": "z\u0307", "ó": "o\u0301",
        "Ą": "A\u0328", "Ę": "E\u0328", "Ć": "C\u0301", "Ń": "N\u0301",
        "Ś": "S\u0301", "Ź": "Z\u0301", "Ż": "Z\u0307", "Ó": "O\u0301",

        # German
        "ä": "a\u0308", "ö": "o\u0308", "ü": "u\u0308", "ß": "ss",
        "Ä": "A\u0308", "Ö": "O\u0308", "Ü": "U\u0308",

        # French
        "à": "a\u0300", "á": "a\u0301", "â": "a\u0302", "ã": "a\u0303", "ä": "a\u0308",
        "ç": "c\u0327", "è": "e\u0300", "é": "e\u0301", "ê": "e\u0302", "ë": "e\u0308",
        "ì": "i\u0300", "í": "i\u0301", "î": "i\u0302", "ï": "i\u0308",
        "ò": "o\u0300", "ó": "o\u0301", "ô": "o\u0302", "õ": "o\u0303", "ö": "o\u0308",
        "ù": "u\u0300", "ú": "u\u0301", "û": "u\u0302", "ü": "u\u0308",
        "À": "A\u0300", "Á": "A\u0301", "Â": "A\u0302", "Ã": "A\u0303", "Ä": "A\u0308",
        "Ç": "C\u0327", "È": "E\u0300", "É": "E\u0301", "Ê": "E\u0302", "Ë": "E\u0308",
        "Ì": "I\u0300", "Í": "I\u0301", "Î": "I\u0302", "Ï": "I\u0308",
        "Ò": "O\u0300", "Ó": "O\u0301", "Ô": "O\u0302", "Õ": "O\u0303", "Ö": "O\u0308",
        "Ù": "U\u0300", "Ú": "U\u0301", "Û": "U\u0302", "Ü": "U\u0308",

        # Spanish
        "ñ": "n\u0303", "Ñ": "N\u0303",

        # Portuguese (additional to French)
        "ā": "a\u0304", "ē": "e\u0304", "ī": "i\u0304", "ō": "o\u0304", "ū": "u\u0304",

        # Czech
        "č": "c\u030C", "ď": "d\u030C", "ě": "e\u030C", "ň": "n\u030C", "ř": "r\u030C",
        "š": "s\u030C", "ť": "t\u030C", "ů": "u\u030A", "ý": "y\u0301", "ž": "z\u030C",
        "Č": "C\u030C", "Ď": "D\u030C", "Ě": "E\u030C", "Ň": "N\u030C", "Ř": "R\u030C",
        "Š": "S\u030C", "Ť": "T\u030C", "Ů": "U\u030A", "Ý": "Y\u0301", "Ž": "Z\u030C",

        # Nordic languages
        "æ": "ae", "ø": "o\u0338", "å": "a\u030A",
        "Æ": "AE", "Ø": "O\u0338", "Å": "A\u030A",
    }

    # Character mappings for ASCII fallbacks
    ASCII_MAP = {
        # Polish
        "ą": "a", "ę": "e", "ć": "c", "ń": "n", "ś": "s", "ź": "z", "ż": "z", "ó": "o", "ł": "l",
        "Ą": "A", "Ę": "E", "Ć": "C", "Ń": "N", "Ś": "S", "Ź": "Z", "Ż": "Z", "Ó": "O", "Ł": "L",

        # German
        "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
        "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",

        # French, Spanish, Portuguese, Italian
        "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a",
        "ç": "c", "è": "e", "é": "e", "ê": "e", "ë": "e",
        "ì": "i", "í": "i", "î": "i", "ï": "i",
        "ñ": "n", "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o", "ø": "o",
        "ù": "u", "ú": "u", "û": "u", "ü": "u",
        "ý": "y", "ÿ": "y",

        # Uppercase versions
        "À": "A", "Á": "A", "Â": "A", "Ã": "A", "Ä": "A", "Å": "A",
        "Ç": "C", "È": "E", "É": "E", "Ê": "E", "Ë": "E",
        "Ì": "I", "Í": "I", "Î": "I", "Ï": "I",
        "Ñ": "N", "Ò": "O", "Ó": "O", "Ô": "O", "Õ": "O", "Ö": "O", "Ø": "O",
        "Ù": "U", "Ú": "U", "Û": "U", "Ü": "U",
        "Ý": "Y", "Ÿ": "Y",

        # Czech
        "č": "c", "ď": "d", "ě": "e", "ň": "n", "ř": "r", "š": "s", "ť": "t", "ů": "u", "ž": "z",
        "Č": "C", "Ď": "D", "Ě": "E", "Ň": "N", "Ř": "R", "Š": "S", "Ť": "T", "Ů": "U", "Ž": "Z",

        # Nordic
        "æ": "ae", "Æ": "AE",
    }

    def detect_language(self, text: str) -> str:
        """Detect language from text based on character patterns"""
        # Character sets for different languages
        language_chars = {
            'polish': 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ',
            'german': 'äöüßÄÖÜ',
            'french': 'àâæçéèêëîïôùûÀÂÆÇÉÈÊËÎÏÔÙÛ',
            'spanish': 'áéíñóúüÁÉÍÑÓÚÜ',
            'portuguese': 'àáâãçéêíóôõúÀÁÂÃÇÉÊÍÓÔÕÚ',
            'italian': 'àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ',
            'czech': 'áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ',
            'slovak': 'áäčďéíľĺňóôŕšťúýžÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ',
            'hungarian': 'áéíóöőúüűÁÉÍÓÖŐÚÜŰ',
            'norwegian': 'æøåÆØÅ',
            'swedish': 'äöåÄÖÅ',
            'danish': 'æøåÆØÅ',
            'finnish': 'äöåÄÖÅ',
            'dutch': ''  # Uses mostly standard Latin
        }

        # Count character matches for each language
        scores = {}
        for lang, chars in language_chars.items():
            if chars:  # Skip empty character sets
                score = sum(1 for char in text if char in chars)
                if score > 0:
                    scores[lang] = score

        # Return language with highest score, default to English
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'english'

    def apply_unicode_decomposition(self, text: str) -> str:
        """Apply Unicode decomposition to special characters"""
        result = ""
        for char in text:
            if char in self.DECOMPOSITION_MAP:
                result += self.DECOMPOSITION_MAP[char]
            else:
                result += char
        return result

    def apply_character_mapping(self, text: str) -> str:
        """Apply ASCII character mapping as fallback"""
        result = ""
        for char in text:
            if char in self.ASCII_MAP:
                result += self.ASCII_MAP[char]
            else:
                result += char
        return result

    def apply_ipa_phonemization(self, text: str, language: str) -> str:
        """Apply IPA phonemization using the same method as the reference Polish F5-TTS"""
        import re

        # Try standard phonemizer first (Linux/Mac/standard)
        try:
            from phonemizer import phonemize

            # Map language names to codes (same as reference)
            lang_codes = {
                'polish': 'pl', 'german': 'de', 'french': 'fr', 'spanish': 'es',
                'portuguese': 'pt', 'italian': 'it', 'czech': 'cs', 'slovak': 'sk',
                'hungarian': 'hu', 'norwegian': 'no', 'swedish': 'sv', 'danish': 'da',
                'finnish': 'fi', 'dutch': 'nl', 'english': 'en'
            }

            lang_code = lang_codes.get(language.lower(), 'en')

            # Use exact same parameters as reference implementation
            ipa_text = phonemize(
                text,
                language=lang_code,
                backend='espeak',
                strip=False,
                preserve_punctuation=True,
                with_stress=True
            )

            # Apply same cleanup as reference implementation
            # Remove language markings like (en), (cmn), (de), (pl), (ru)
            ipa_text = re.sub(r'\([a-z]{2,3}\)', '', ipa_text)
            ipa_text = re.sub(r'tʃˈaɪniːzlˈe̞tə', '', ipa_text)
            ipa_text = re.sub(r'tʃˈaɪniːzɭˈetə', '', ipa_text)
            ipa_text = re.sub(r'dʒˈapəniːzlˈe̞tə', '', ipa_text)
            ipa_text = re.sub(r'dʒˈapəniːzɭˈetə', '', ipa_text)

            return ipa_text

        except Exception as e1:
            # Try Windows-specific espeak-phonemizer-windows as fallback
            try:
                from espeak_phonemizer import Phonemizer

                # Map language names to espeak voice codes
                voice_map = {
                    'polish': 'pl', 'german': 'de', 'french': 'fr', 'spanish': 'es',
                    'portuguese': 'pt', 'italian': 'it', 'czech': 'cs', 'slovak': 'sk',
                    'hungarian': 'hu', 'norwegian': 'no', 'swedish': 'sv', 'danish': 'da',
                    'finnish': 'fi', 'dutch': 'nl', 'english': 'en'
                }

                voice = voice_map.get(language.lower(), 'en')
                phonemizer = Phonemizer()
                ipa_text = phonemizer.phonemize(text, voice=voice)

                # Apply same cleanup as reference implementation
                ipa_text = re.sub(r'\([a-z]{2,3}\)', '', ipa_text)
                ipa_text = re.sub(r'tʃˈaɪniːzlˈe̞tə', '', ipa_text)
                ipa_text = re.sub(r'tʃˈaɪniːzɭˈetə', '', ipa_text)
                ipa_text = re.sub(r'dʒˈapəniːzlˈe̞tə', '', ipa_text)
                ipa_text = re.sub(r'dʒˈapəniːzɭˈetə', '', ipa_text)

                return ipa_text

            except Exception as e2:
                print(f"⚠️ Both phonemizers failed - standard: {e1}, windows: {e2}")
                print("⚠️ Falling back to Unicode decomposition")
                return self.apply_unicode_decomposition(text)

    def normalize_text(self, text: str, method: str, language: str, show_debug: bool = True) -> Tuple[str]:
        """Main normalization function"""
        if not text.strip():
            return (text,)

        original_text = text

        # Auto-detect language if requested
        if language == "Auto-detect":
            detected_lang = self.detect_language(text)
            if show_debug:
                print(f"🔍 Auto-detected language: {detected_lang}")
        else:
            detected_lang = language.lower()

        # Apply selected method
        if method == "Pass-through":
            normalized_text = text

        elif method == "Unicode Decomposition":
            normalized_text = self.apply_unicode_decomposition(text)

        elif method == "IPA Phonemization":
            normalized_text = self.apply_ipa_phonemization(text, detected_lang)

        elif method == "Character Mapping":
            normalized_text = self.apply_character_mapping(text)

        else:
            # Fallback to pass-through
            normalized_text = text

        # Show debug information if requested
        if show_debug and normalized_text != original_text:
            print(f"📝 Phoneme Text Normalizer")
            print(f"   Method: {method}")
            print(f"   Language: {detected_lang if language == 'Auto-detect' else language}")
            print(f"   Original:   '{original_text[:100]}{'...' if len(original_text) > 100 else ''}'")
            print(f"   Normalized: '{normalized_text[:100]}{'...' if len(normalized_text) > 100 else ''}'")

            # Show character-by-character changes for short texts (only for same-length transformations)
            if len(original_text) <= 50 and len(original_text) == len(normalized_text):
                changes = []
                for i, (orig, norm) in enumerate(zip(original_text, normalized_text)):
                    if orig != norm:
                        changes.append(f"{orig}→{norm}")
                if changes:
                    print(f"   Changes: {', '.join(changes)}")
            elif len(original_text) != len(normalized_text) and len(original_text) <= 50:
                print(f"   Text length changed: {len(original_text)} → {len(normalized_text)} characters")

        return (normalized_text,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "PhonemeTextNormalizer": PhonemeTextNormalizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhonemeTextNormalizer": "📝 Phoneme Text Normalizer"
}