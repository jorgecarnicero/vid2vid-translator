import re

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
VIDEO_ENTRADA = "prueba.mp4"
AUDIO_TEMPORAL = "audio_temp.ogg"
AUDIO_SALIDA = "voz_ingles_sincronizada.wav"
VIDEO_FINAL = "video_final_doblado.mp4"
CACHE_DIR = "cache_doblaje"
MAX_HILOS = 4

# --- Parámetros de "pulido" de audio ---
MIN_INICIO_MS = 250          # nunca empieza a hablar antes de este instante
UMBRAL_CONTINUIDAD_MS = 250  # huecos menores a esto se tratan como continuidad (crossfade), no como pausa real
CROSSFADE_MS = 30            # unión suave cuando dos trozos van pegados/solapados
FADE_EDGES_MS = 15           # fade in/out de cada segmento TTS para evitar clics
UMBRAL_PAUSA_AGRUPACION = 0.3  # para agrupar palabras en frases

# Timeouts de red (evita cuelgues silenciosos si Ollama/VoiceBox se atascan)
TIMEOUT_OLLAMA = 90
TIMEOUT_VOICEBOX = 60
MAX_INTENTOS_STATUS = 180  # ~3 minutos de polling máximo por segmento

# APIs Locales
URL_OLLAMA = "http://localhost:11434/api/chat"
URL_VOICEBOX = "http://127.0.0.1:17493/generate"

# Parámetros de los Modelos
MODELO_OLLAMA = "qwen2.5:7b"
VOICE_PROFILE = "cd4bdfe9-bae8-4360-a0cb-880bbc62f645"
TARGET_LANGUAGE = "en"

# Sube esto si cambias el prompt de traducción o el algoritmo de anclaje,
# para invalidar la caché vieja sin tener que borrarla a mano.
CACHE_VERSION = "v2"

MAX_PAUSA_INTERNA = 1200

# ==========================================
# UTILIDADES DE TEXTO / ANCLAJE
# ==========================================
ANCHOR_REGEX = re.compile(r'"([^"]{2,40})"|\'([^\']{2,40})\'|\b([A-Z][a-zA-Z0-9_]{2,})\b')