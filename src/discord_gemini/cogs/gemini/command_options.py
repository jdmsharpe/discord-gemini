from discord.commands import OptionChoice

from ...util import LYRIA_REALTIME_MODEL

CHAT_MODEL_CHOICES = [
    OptionChoice(name="Gemini 3.1 Pro", value="gemini-3.1-pro-preview"),
    OptionChoice(name="Gemini 3.1 Flash Lite", value="gemini-3.1-flash-lite-preview"),
    OptionChoice(name="Gemini 3.0 Flash", value="gemini-3-flash-preview"),
    OptionChoice(name="Gemini 2.5 Pro", value="gemini-2.5-pro"),
    OptionChoice(name="Gemini 2.5 Flash", value="gemini-2.5-flash"),
    OptionChoice(name="Gemini 2.5 Flash Lite", value="gemini-2.5-flash-lite"),
    OptionChoice(name="Gemini 2.0 Flash", value="gemini-2.0-flash"),
    OptionChoice(name="Gemini 2.0 Flash Lite", value="gemini-2.0-flash-lite"),
]

MEDIA_RESOLUTION_CHOICES = [
    OptionChoice(name="Low", value="MEDIA_RESOLUTION_LOW"),
    OptionChoice(name="Medium", value="MEDIA_RESOLUTION_MEDIUM"),
    OptionChoice(name="High", value="MEDIA_RESOLUTION_HIGH"),
]

THINKING_LEVEL_CHOICES = [
    OptionChoice(name="Minimal", value="minimal"),
    OptionChoice(name="Low", value="low"),
    OptionChoice(name="Medium", value="medium"),
    OptionChoice(name="High", value="high"),
]

IMAGE_MODEL_CHOICES = [
    OptionChoice(name="Gemini 3.1 Flash Image", value="gemini-3.1-flash-image-preview"),
    OptionChoice(name="Gemini 3.0 Pro Image", value="gemini-3-pro-image-preview"),
    OptionChoice(name="Gemini 2.5 Flash Image", value="gemini-2.5-flash-image"),
    OptionChoice(name="Imagen 4", value="imagen-4.0-generate-001"),
    OptionChoice(name="Imagen 4 Ultra", value="imagen-4.0-ultra-generate-001"),
    OptionChoice(name="Imagen 4 Fast", value="imagen-4.0-fast-generate-001"),
]

IMAGE_ASPECT_RATIO_CHOICES = [
    OptionChoice(name="Square (1:1)", value="1:1"),
    OptionChoice(name="Portrait (3:4)", value="3:4"),
    OptionChoice(name="Landscape (4:3)", value="4:3"),
    OptionChoice(name="Portrait (9:16)", value="9:16"),
    OptionChoice(name="Landscape (16:9)", value="16:9"),
]

PERSON_GENERATION_CHOICES = [
    OptionChoice(name="Don't Allow", value="dont_allow"),
    OptionChoice(name="Allow Adults", value="allow_adult"),
    OptionChoice(name="Allow All", value="allow_all"),
]

IMAGE_SIZE_CHOICES = [
    OptionChoice(name="1K", value="1k"),
    OptionChoice(name="2K", value="2k"),
]

VIDEO_MODEL_CHOICES = [
    OptionChoice(name="Veo 3.1 Lite Preview", value="veo-3.1-lite-generate-preview"),
    OptionChoice(name="Veo 3.1 Preview", value="veo-3.1-generate-preview"),
    OptionChoice(name="Veo 3.1 Fast Preview", value="veo-3.1-fast-generate-preview"),
    OptionChoice(name="Veo 3", value="veo-3.0-generate-001"),
    OptionChoice(name="Veo 3 Fast", value="veo-3.0-fast-generate-001"),
    OptionChoice(name="Veo 2", value="veo-2.0-generate-001"),
]

VIDEO_ASPECT_RATIO_CHOICES = [
    OptionChoice(name="Landscape (16:9)", value="16:9"),
    OptionChoice(name="Portrait (9:16)", value="9:16"),
]

VIDEO_RESOLUTION_CHOICES = [
    OptionChoice(name="720p", value="720p"),
    OptionChoice(name="1080p", value="1080p"),
    OptionChoice(name="4K", value="4k"),
]

TTS_MODEL_CHOICES = [
    OptionChoice(name="Gemini 2.5 Flash Preview TTS", value="gemini-2.5-flash-preview-tts"),
    OptionChoice(name="Gemini 2.5 Pro Preview TTS", value="gemini-2.5-pro-preview-tts"),
]

TTS_VOICE_CHOICES = [
    OptionChoice(name="Kore (Firm)", value="Kore"),
    OptionChoice(name="Puck (Upbeat)", value="Puck"),
    OptionChoice(name="Charon (Informative)", value="Charon"),
    OptionChoice(name="Zephyr (Bright)", value="Zephyr"),
    OptionChoice(name="Fenrir (Excitable)", value="Fenrir"),
    OptionChoice(name="Leda (Youthful)", value="Leda"),
    OptionChoice(name="Orus (Firm)", value="Orus"),
    OptionChoice(name="Aoede (Breezy)", value="Aoede"),
    OptionChoice(name="Callirrhoe (Easy-going)", value="Callirrhoe"),
    OptionChoice(name="Autonoe (Bright)", value="Autonoe"),
    OptionChoice(name="Enceladus (Breathy)", value="Enceladus"),
    OptionChoice(name="Iapetus (Clear)", value="Iapetus"),
    OptionChoice(name="Umbriel (Easy-going)", value="Umbriel"),
    OptionChoice(name="Algieba (Smooth)", value="Algieba"),
    OptionChoice(name="Despina (Smooth)", value="Despina"),
    OptionChoice(name="Erinome (Clear)", value="Erinome"),
    OptionChoice(name="Algenib (Gravelly)", value="Algenib"),
    OptionChoice(name="Rasalgethi (Informative)", value="Rasalgethi"),
    OptionChoice(name="Laomedeia (Upbeat)", value="Laomedeia"),
    OptionChoice(name="Achernar (Soft)", value="Achernar"),
    OptionChoice(name="Alnilam (Firm)", value="Alnilam"),
    OptionChoice(name="Schedar (Even)", value="Schedar"),
    OptionChoice(name="Gacrux (Mature)", value="Gacrux"),
    OptionChoice(name="Achird (Friendly)", value="Achird"),
    OptionChoice(name="Sulafat (Warm)", value="Sulafat"),
]

MUSIC_MODEL_CHOICES = [
    OptionChoice(name="Lyria 3 Pro Preview", value="lyria-3-pro-preview"),
    OptionChoice(name="Lyria 3 Clip Preview", value="lyria-3-clip-preview"),
    OptionChoice(name="Lyria RealTime Experimental", value=LYRIA_REALTIME_MODEL),
]

MUSIC_SCALE_CHOICES = [
    OptionChoice(name="C Major / A Minor", value="C_MAJOR_A_MINOR"),
    OptionChoice(name="D♭ Major / B♭ Minor", value="D_FLAT_MAJOR_B_FLAT_MINOR"),
    OptionChoice(name="D Major / B Minor", value="D_MAJOR_B_MINOR"),
    OptionChoice(name="E♭ Major / C Minor", value="E_FLAT_MAJOR_C_MINOR"),
    OptionChoice(name="E Major / C# Minor", value="E_MAJOR_D_FLAT_MINOR"),
    OptionChoice(name="F Major / D Minor", value="F_MAJOR_D_MINOR"),
    OptionChoice(name="G♭ Major / E♭ Minor", value="G_FLAT_MAJOR_E_FLAT_MINOR"),
    OptionChoice(name="G Major / E Minor", value="G_MAJOR_E_MINOR"),
    OptionChoice(name="A♭ Major / F Minor", value="A_FLAT_MAJOR_F_MINOR"),
    OptionChoice(name="A Major / F# Minor", value="A_MAJOR_G_FLAT_MINOR"),
    OptionChoice(name="B♭ Major / G Minor", value="B_FLAT_MAJOR_G_MINOR"),
    OptionChoice(name="B Major / G# Minor", value="B_MAJOR_A_FLAT_MINOR"),
]

__all__ = [
    "CHAT_MODEL_CHOICES",
    "IMAGE_ASPECT_RATIO_CHOICES",
    "IMAGE_MODEL_CHOICES",
    "IMAGE_SIZE_CHOICES",
    "MEDIA_RESOLUTION_CHOICES",
    "MUSIC_MODEL_CHOICES",
    "MUSIC_SCALE_CHOICES",
    "PERSON_GENERATION_CHOICES",
    "THINKING_LEVEL_CHOICES",
    "TTS_MODEL_CHOICES",
    "TTS_VOICE_CHOICES",
    "VIDEO_ASPECT_RATIO_CHOICES",
    "VIDEO_MODEL_CHOICES",
    "VIDEO_RESOLUTION_CHOICES",
]
