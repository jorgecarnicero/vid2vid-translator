import os
import io
import re
import subprocess
import unicodedata
import requests
from transformers import pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import time
import json
import uuid

from constantes import *

def normalizar(txt):
    """minúsculas, sin acentos, sin puntuación en los extremos (para comparar palabras)."""
    txt = txt.lower().strip(",.!?;:\"'()[]¿¡")
    txt = ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')
    return txt


def detectar_anchor_es(texto):
    """
    Busca en el texto español un candidato a 'palabra ancla': algo entre comillas,
    una palabra con mayúscula que NO sea simplemente el inicio de la frase, o un
    identificador tipo nombre de archivo/botón. Devuelve el string tal cual aparece
    en el texto, o None si no hay ningún candidato claro.
    """
    texto_stripped = texto.strip()
    for m in ANCHOR_REGEX.finditer(texto):
        candidato = next(g for g in m.groups() if g)
        entre_comillas = bool(m.group(1) or m.group(2))
        if not entre_comillas and texto_stripped.startswith(candidato):
            continue
        return candidato
    return None


def buscar_tiempo_palabra_es(anchor_es, palabras_es, t_inicio_ventana, t_fin_ventana):
    """
    Dado el listado de timestamps POR PALABRA de Whisper, localiza en qué segundo
    exacto empieza el anchor dentro de la ventana [t_inicio_ventana, t_fin_ventana].
    """
    anchor_tokens = normalizar(anchor_es).split()
    if not anchor_tokens:
        return None

    candidatos = [
        p for p in palabras_es
        if p["timestamp"][0] is not None
        and (t_inicio_ventana - 0.05) <= p["timestamp"][0] <= (t_fin_ventana + 0.05)
    ]

    for i in range(len(candidatos)):
        ventana = candidatos[i:i + len(anchor_tokens)]
        if len(ventana) < len(anchor_tokens):
            break
        tokens_ventana = [normalizar(p["text"]) for p in ventana]
        if tokens_ventana == anchor_tokens:
            return candidatos[i]["timestamp"][0]

    if len(anchor_tokens) == 1:
        for p in candidatos:
            if anchor_tokens[0] in normalizar(p["text"]):
                return p["timestamp"][0]

    return None


def extraer_anchor(texto_traducido):
    """
    Busca el marcador [[...]] que Ollama debe insertar alrededor del equivalente
    en inglés del anchor. Devuelve (texto_limpio_sin_corchetes, anchor_en, pos_inicio, pos_fin).
    """
    m = re.search(r'\[\[(.*?)\]\]', texto_traducido)
    if not m:
        return texto_traducido, None, None, None
    anchor_en = m.group(1).strip()
    texto_limpio = texto_traducido[:m.start()] + anchor_en + texto_traducido[m.end():]
    return texto_limpio, anchor_en, m.start(), m.start() + len(anchor_en)


# ==========================================
# FUNCIONES DE PROCESAMIENTO DE AUDIO
# ==========================================
def ajustar_velocidad_audio(audio_segment, factor):
    """Ajuste de velocidad Thread-Safe usando UUIDs para evitar colisiones en paralelo."""
    factor = max(0.85, min(factor, 1.3))

    uid = uuid.uuid4().hex
    temp_in = f"temp_in_{uid}.wav"
    temp_out = f"temp_out_{uid}.wav"

    try:
        audio_segment.export(temp_in, format="wav")

        res = subprocess.run(
            ["ffmpeg", "-y", "-i", temp_in, "-filter:a", f"atempo={factor}", temp_out],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg atempo falló: {res.stderr.decode(errors='ignore')}")

        nuevo_audio = AudioSegment.from_file(temp_out, format="wav")
    finally:
        if os.path.exists(temp_in): os.remove(temp_in)
        if os.path.exists(temp_out): os.remove(temp_out)

    return nuevo_audio


def expandir_audio_en_hueco(segmento_tts, duracion_objetivo_ms):
    """
    Sistema de Clustering: ralentiza levemente y reparte las pausas de forma humana
    para que el segmento ocupe exactamente duracion_objetivo_ms.
    """
    if duracion_objetivo_ms <= 0:
        return segmento_tts[:1]

    duracion_tts = len(segmento_tts)

    factor_base = max(0.85, duracion_tts / duracion_objetivo_ms)
    if factor_base < 0.95:
        segmento_tts = ajustar_velocidad_audio(segmento_tts, factor_base)
        duracion_tts = len(segmento_tts)

    if duracion_tts < duracion_objetivo_ms * 0.90:
        trozos = split_on_silence(
            segmento_tts,
            min_silence_len=200,
            silence_thresh=-45,
            keep_silence=150
        )

        if len(trozos) > 1:
            duracion_palabras = sum(len(t) for t in trozos)
            silencio_a_repartir = max(0, duracion_objetivo_ms - duracion_palabras)

            silencio_por_hueco_bruto = silencio_a_repartir // (len(trozos) - 1)
            
            silencio_interno = min(silencio_por_hueco_bruto, MAX_PAUSA_INTERNA)

            silencio_gastado_dentro = silencio_interno * (len(trozos) - 1)
            silencio_sobrante_extremos = silencio_a_repartir - silencio_gastado_dentro

            silencio_inicio = int(silencio_sobrante_extremos * 0.20)
            silencio_final = int(silencio_sobrante_extremos * 0.80)

            audio_expandido = AudioSegment.silent(duration=silencio_inicio)
            tramo_silencio = AudioSegment.silent(duration=silencio_interno)

            for idx, trozo in enumerate(trozos):
                trozo_suavizado = trozo.fade_in(30).fade_out(30)
                audio_expandido += trozo_suavizado

                if idx < len(trozos) - 1:
                    audio_expandido += tramo_silencio

            audio_expandido += AudioSegment.silent(duration=silencio_final)
            return audio_expandido

    return segmento_tts


def localizar_punto_de_corte(audio, texto_antes, texto_completo):
    """
    NUEVO: en vez de generar dos audios TTS independientes (que suena discontinuo
    y a veces repite/traga fonemas en la unión), generamos el audio UNA SOLA VEZ
    de forma continua y localizamos aquí, dentro de ese audio ya generado, el punto
    de silencio real que separa 'texto_antes' del resto — usando detect_nonsilent,
    que sí devuelve posiciones absolutas dentro del audio original (a diferencia de
    split_on_silence, que solo devuelve los trozos sueltos sin su timing).
    """
    n_palabras_antes = len(texto_antes.split())
    if n_palabras_antes <= 0:
        return 0

    try:
        umbral = audio.dBFS - 16 if audio.dBFS != float("-inf") else -45
        rangos = detect_nonsilent(audio, min_silence_len=90, silence_thresh=umbral, seek_step=10)
    except Exception:
        rangos = []

    if len(rangos) >= n_palabras_antes:
        _, fin_ultima_palabra_antes = rangos[n_palabras_antes - 1]
        if n_palabras_antes < len(rangos):
            inicio_siguiente_palabra, _ = rangos[n_palabras_antes]
            return (fin_ultima_palabra_antes + inicio_siguiente_palabra) // 2
        return fin_ultima_palabra_antes

    # Fallback: si detect_nonsilent no encontró suficientes "palabras" (audio muy
    # fluido, sin silencios claros entre ellas), repartimos proporcionalmente por
    # nº de caracteres, que es peor pero razonable como último recurso.
    proporcion = len(texto_antes) / max(len(texto_completo), 1)
    return int(len(audio) * proporcion)


# ==========================================
# FUNCIONES DE IA (TRADUCCIÓN Y TTS)
# ==========================================
def generar_resumen_global(fragmentos_agrupados):
    """
    NUEVO: antes de traducir frase a frase, le pedimos a Ollama un resumen de
    TODO el vídeo + los términos técnicos/nombres propios recurrentes. Este
    resumen se inyecta luego en cada llamada de traducción como contexto fijo,
    así el LLM no "olvida" cómo tradujo algo 10 frases atrás y mantiene
    terminología y tono consistentes en todo el vídeo.
    """
    texto_completo = " ".join(f["text"].strip() for f in fragmentos_agrupados)

    system_prompt = (
        "You will receive the full Spanish transcript of a software tutorial video. "
        "Summarize in 2-4 sentences what the tutorial is about. Then list any recurring "
        "technical terms, UI labels, filenames or proper nouns that appear more than once "
        "and MUST be translated the exact same way every time they appear. "
        "Keep the whole response short and in English. Output plain text, no markdown."
    )
    payload = {
        "model": MODELO_OLLAMA,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": texto_completo}
        ],
        "stream": False
    }
    try:
        response = requests.post(URL_OLLAMA, json=payload, timeout=TIMEOUT_OLLAMA)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except Exception as e:
        print(f"[!] No se pudo generar el resumen global (se continúa sin él): {e}")
        return ""


def traducir_texto(texto_espanol, traduccion_anterior="", texto_espanol_anterior="", anchor_es=None,
                    contexto_global=""):
    """Traduce el texto español al inglés usando Ollama. Devuelve (texto_en, anchor_en)."""

    bloque_contexto_global = (
        f"FULL VIDEO CONTEXT (use this to keep terminology and tone consistent across the "
        f"whole video, but translate ONLY the current sentence below):\n{contexto_global}\n\n"
        if contexto_global else ""
    )

    system_prompt = (
        f"{bloque_contexto_global}"
        "Act as a professional script adapter for educational dubbing. "
        "Translate the Spanish sentence to English for a software tutorial. "
        "CRITICAL RULES: "
        "1. TIMING: English is shorter, so do not summarize. Use conversational fillers naturally, BUT VARY THEM. "
        "2. AVOID REPETITION: DO NOT start with the same filler as the previous sentence. "
        "3. ACCURACY & CUT-OFFS: DO NOT invent steps. If the text cuts off abruptly, translate it up to that point and leave it cut off. "
        "4. CONTINUITY (CRUCIAL): These are sequential chunks. If the previous chunk ended mid-sentence (e.g., ends in 'the', 'a', 'and'), your translation MUST seamlessly continue the exact grammar of the PREVIOUS ENGLISH TRANSLATION. Do not force it into a standalone complete sentence. "
        "5. NUMBERS: Convert ALL digits into their written word form (e.g., '100' must be translated as 'one hundred'). "
        "6. LANGUAGE: You MUST output ONLY in English. NEVER output Chinese characters, system warnings, or errors. "
        "7. ANCHOR MARKING: If the user message includes a line 'ANCHOR: <word/phrase>', that marks a critical "
        "Spanish term (a UI label, filename, proper noun or code identifier) visible on screen. It MUST appear in "
        "your English translation. Wrap its English equivalent (or the literal string itself, if it is a filename/"
        "code identifier that should stay unchanged) in double square brackets, exactly where it naturally falls "
        "in the English sentence, e.g. [[Save]] or [[config.json]]. Only wrap that one term, nothing else. If "
        "there is no ANCHOR line, do not use brackets at all. "
        "8. TERMINOLOGY: If the FULL VIDEO CONTEXT above lists recurring terms, translate them exactly as "
        "specified there every time they appear. "
        "Return ONLY the translated English chunk."
    )

    if traduccion_anterior and texto_espanol_anterior:
        user_content = (
            f"PREVIOUS SPANISH: '{texto_espanol_anterior}'\n"
            f"PREVIOUS ENGLISH: '{traduccion_anterior}'\n\n"
            f"CURRENT SPANISH TO TRANSLATE: '{texto_espanol}'"
        )
    else:
        user_content = f"CURRENT SPANISH TO TRANSLATE: '{texto_espanol}'"

    if anchor_es:
        user_content += f"\n\nANCHOR: '{anchor_es}'"

    payload = {
        "model": MODELO_OLLAMA,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "stream": False
    }
    response = requests.post(URL_OLLAMA, json=payload, timeout=TIMEOUT_OLLAMA)
    response.raise_for_status()
    contenido = response.json()["message"]["content"].strip()

    texto_limpio, anchor_en, _, _ = extraer_anchor(contenido)
    return texto_limpio, anchor_en


def generar_audio_voicebox(texto_ingles):
    """Genera audio en inglés usando VoiceBox local."""
    payload = {
        "profile_id": VOICE_PROFILE,
        "text": texto_ingles,
        "language": TARGET_LANGUAGE,
        "seed": 0,
        "model_size": "1.7B",
        "instruct": "",
        "engine": "qwen",
        "personality": False,
        "max_chunk_chars": 800,
        "crossfade_ms": 50,
        "normalize": True,
        "effects_chain": []
    }

    response = requests.post(URL_VOICEBOX, json=payload, headers={"Content-Type": "application/json"},
                              timeout=TIMEOUT_VOICEBOX)
    response.raise_for_status()

    job_id = response.json().get("id")
    if not job_id:
        raise Exception("VoiceBox no devolvió ID.")

    url_status = f"{URL_VOICEBOX}/{job_id}/status"

    intentos = 0
    while True:
        intentos += 1
        if intentos > MAX_INTENTOS_STATUS:
            raise Exception(f"VoiceBox no completó el job {job_id} tras {MAX_INTENTOS_STATUS}s de espera.")

        time.sleep(1)
        res_status = requests.get(url_status, timeout=TIMEOUT_VOICEBOX)
        texto_crudo = res_status.text.strip()

        if texto_crudo.startswith("data:"):
            texto_crudo = texto_crudo.replace("data:", "", 1).strip()

        try:
            status = json.loads(texto_crudo).get("status")
        except Exception:
            continue

        if status == "generating":
            continue
        elif status == "completed":
            break
        elif status == "error":
            raise Exception(f"Fallo en VoiceBox para el job {job_id}.")

    res_audio = requests.get(f"http://127.0.0.1:17493/audio/{job_id}", timeout=TIMEOUT_VOICEBOX)
    return res_audio.content


def generar_audio_con_anchor(texto_en, anchor_char_start, t_inicio_seg, t_anchor_es, duracion_objetivo_ms,
                              ruta_wav_cache, ruta_txt_cache):
    """
    Genera el TTS del segmento completo en UNA SOLA llamada (más natural, sin
    discontinuidades de prosodia), y luego localiza dentro de ese audio ya
    generado el punto de silencio real donde cae el anchor, para poder estirar
    cada mitad por separado sin haber tenido que generar dos audios
    independientes. Esto es lo que antes causaba el efecto de "se lía o repite
    palabras": pegar dos generaciones de TTS que no compartían contexto acústico.
    """
    if os.path.exists(ruta_txt_cache) and os.path.exists(ruta_wav_cache):
        return AudioSegment.from_file(ruta_wav_cache, format="wav")

    texto_antes = texto_en[:anchor_char_start].strip()

    objetivo_anchor_ms = int((t_anchor_es - t_inicio_seg) * 1000)
    objetivo_anchor_ms = max(0, min(objetivo_anchor_ms, duracion_objetivo_ms))

    audio_completo = AudioSegment.from_file(io.BytesIO(generar_audio_voicebox(texto_en)), format="wav")

    if not texto_antes or objetivo_anchor_ms < 150:
        # El anchor prácticamente está al principio: partir no aporta nada.
        audio = expandir_audio_en_hueco(audio_completo, duracion_objetivo_ms)
    else:
        punto_corte_ms = localizar_punto_de_corte(audio_completo, texto_antes, texto_en)
        punto_corte_ms = max(1, min(punto_corte_ms, len(audio_completo) - 1))

        parte_1 = audio_completo[:punto_corte_ms]
        parte_2 = audio_completo[punto_corte_ms:]

        parte_1 = expandir_audio_en_hueco(parte_1, objetivo_anchor_ms)
        parte_2 = expandir_audio_en_hueco(parte_2, duracion_objetivo_ms - objetivo_anchor_ms)

        parte_1 = parte_1.fade_out(FADE_EDGES_MS)
        parte_2 = parte_2.fade_in(FADE_EDGES_MS)
        audio = parte_1 + parte_2

    with open(ruta_txt_cache, "w", encoding="utf-8") as f:
        json.dump({"texto_en": texto_en, "anchor_char_start": anchor_char_start}, f)
    audio.export(ruta_wav_cache, format="wav")
    return audio


def worker_procesar_audio(data, indice):
    texto_en = data["texto_en"]
    ruta_txt = data["ruta_txt"]
    ruta_wav = data["ruta_wav"]
    timestamp = data["timestamp"]

    try:
        duracion_objetivo_ms = None
        if timestamp[1] is not None:
            duracion_objetivo_ms = int((timestamp[1] - timestamp[0]) * 1000)

        usa_anchor = (
            data.get("anchor_en") and duracion_objetivo_ms
            and data.get("anchor_char_start") is not None
            and data.get("t_anchor_es") is not None
        )

        if usa_anchor:
            segmento_tts = generar_audio_con_anchor(
                texto_en, data["anchor_char_start"], timestamp[0], data["t_anchor_es"],
                duracion_objetivo_ms, ruta_wav, ruta_txt
            )
        else:
            if os.path.exists(ruta_txt) and os.path.exists(ruta_wav):
                segmento_tts = AudioSegment.from_file(ruta_wav, format="wav")
            else:
                audio_bytes = generar_audio_voicebox(texto_en)
                with open(ruta_txt, "w", encoding="utf-8") as f:
                    json.dump({"texto_en": texto_en, "anchor_char_start": None}, f)
                with open(ruta_wav, "wb") as f:
                    f.write(audio_bytes)
                segmento_tts = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

            if duracion_objetivo_ms:
                segmento_tts = expandir_audio_en_hueco(segmento_tts, duracion_objetivo_ms)

        segmento_tts = segmento_tts.fade_in(FADE_EDGES_MS).fade_out(FADE_EDGES_MS)

        return indice, segmento_tts

    except Exception as e:
        print(f"[!] Error en hilo {indice+1}: {e}")
        return indice, None