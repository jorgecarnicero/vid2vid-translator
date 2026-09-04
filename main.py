import os
import subprocess
import torch
from transformers import pipeline
from pydub import AudioSegment
import hashlib
import json
import concurrent.futures
import gc

from constantes import *
from functions import generar_resumen_global, traducir_texto, detectar_anchor_es, buscar_tiempo_palabra_es, \
    worker_procesar_audio

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# ==========================================
# FLUJO PRINCIPAL DE EJECUCIÓN
# ==========================================
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. EXTRACCIÓN
    print("\n[1/7] Extrayendo audio del video...")
    res_extract = subprocess.run(
        ["ffmpeg", "-i", VIDEO_ENTRADA, "-vn", "-c:a", "libvorbis", "-q:a", "4", "-y", AUDIO_TEMPORAL],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    if res_extract.returncode != 0:
        raise RuntimeError(f"ffmpeg no pudo extraer el audio: {res_extract.stderr.decode(errors='ignore')}")

    # 2. TRANSCRIPCIÓN (timestamps POR PALABRA: necesarios para el anclaje)
    print("\n[2/7] Transcribiendo y mapeando tiempos con Whisper Turbo...")
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                           device=device, dtype=dtype, chunk_length_s=30, batch_size=8,
                           ignore_warning=True, return_timestamps="word")

    palabras_es_todas = transcriber(AUDIO_TEMPORAL, generate_kwargs={"language": "spanish"})["chunks"]

    fragmentos_agrupados = []
    if len(palabras_es_todas) > 0:
        actual = palabras_es_todas[0].copy()
        for siguiente in palabras_es_todas[1:]:
            t_fin_actual, t_inicio_siguiente = actual["timestamp"][1], siguiente["timestamp"][0]
            if t_fin_actual is None or t_inicio_siguiente is None:
                fragmentos_agrupados.append(actual); actual = siguiente.copy(); continue

            if (t_inicio_siguiente - t_fin_actual) < UMBRAL_PAUSA_AGRUPACION:
                actual["text"] += " " + siguiente["text"].strip()
                actual["timestamp"] = (actual["timestamp"][0], siguiente["timestamp"][1])
            else:
                fragmentos_agrupados.append(actual); actual = siguiente.copy()
        fragmentos_agrupados.append(actual)

    print(f"Bloques detectados: {len(fragmentos_agrupados)}")

    print("      -> Limpiando VRAM de la gráfica...")
    del transcriber
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("      -> [OK] Memoria lista para Ollama y VoiceBox.")

    # 3. RESUMEN GLOBAL (NUEVO): contexto de todo el vídeo para consistencia terminológica
    print("\n[3/7] Generando resumen global del vídeo para dar contexto al traductor...")
    contexto_global = generar_resumen_global(fragmentos_agrupados)
    if contexto_global:
        print(f"      -> Contexto global:\n{contexto_global}\n")
    else:
        print("      -> Sin contexto global (se continúa solo con contexto local por frase).")

    # 4. TRADUCCIÓN SECUENCIAL (+ detección y localización de anchors)
    print("\n[4/7] FASE A: Traducción Secuencial (Conservando Contexto)...")
    textos_preparados = []
    traduccion_anterior = ""
    texto_es_anterior = ""

    for i, fragmento in enumerate(fragmentos_agrupados):
        texto_es = fragmento["text"].strip()
        timestamp = fragmento["timestamp"]
        if timestamp[0] is None:
            continue

        anchor_es = detectar_anchor_es(texto_es)
        t_anchor_es = None
        if anchor_es and timestamp[1] is not None:
            t_anchor_es = buscar_tiempo_palabra_es(anchor_es, palabras_es_todas, timestamp[0], timestamp[1])

        string_hash = f"{texto_es}_{VOICE_PROFILE}_{TARGET_LANGUAGE}_{CACHE_VERSION}".encode('utf-8')
        hash_id = hashlib.md5(string_hash).hexdigest()
        ruta_txt = os.path.join(CACHE_DIR, f"{hash_id}.json")
        ruta_wav = os.path.join(CACHE_DIR, f"{hash_id}.wav")

        anchor_en, anchor_char_start = None, None

        if os.path.exists(ruta_txt):
            with open(ruta_txt, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            texto_en = cache_data["texto_en"]
            anchor_char_start = cache_data.get("anchor_char_start")
            if anchor_char_start is not None and anchor_es:
                anchor_en = texto_en[anchor_char_start:].split()[0] if texto_en[anchor_char_start:] else None
        else:
            texto_en, anchor_en = traducir_texto(texto_es, traduccion_anterior, texto_es_anterior, anchor_es,
                                                  contexto_global)
            if anchor_en:
                pos = texto_en.find(anchor_en)
                if pos != -1:
                    anchor_char_start = pos

        etiqueta_anchor = f"\n    ANCHOR: '{anchor_es}' -> '{anchor_en}'" if anchor_en else ""
        print(f"[{i+1}] ES: {texto_es}\n    EN: {texto_en}{etiqueta_anchor}\n")

        textos_preparados.append({
            "texto_es": texto_es, "texto_en": texto_en, "hash_id": hash_id,
            "ruta_txt": ruta_txt, "ruta_wav": ruta_wav, "timestamp": timestamp,
            "anchor_en": anchor_en, "anchor_char_start": anchor_char_start,
            "t_anchor_es": t_anchor_es
        })
        traduccion_anterior = texto_en
        texto_es_anterior = texto_es

    # 5. GENERACIÓN PARALELA
    print(f"\n[5/7] FASE B: Generación de Audio en Paralelo (Max {MAX_HILOS} hilos)...")
    audios_procesados = {}
    fallidos = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_HILOS) as executor:
        futuros = {executor.submit(worker_procesar_audio, data, i): i for i, data in enumerate(textos_preparados)}
        for futuro in concurrent.futures.as_completed(futuros):
            idx = futuros[futuro]
            _, segmento_tts = futuro.result()
            audios_procesados[idx] = segmento_tts
            if segmento_tts is None:
                fallidos.append(idx)
            print(f"      -> Audio [{idx+1}/{len(textos_preparados)}] completado y ajustado.")

    if fallidos:
        print(f"\n[!] AVISO: {len(fallidos)} bloque(s) fallaron y quedarán MUDOS en el vídeo final: "
              f"{[i+1 for i in fallidos]}")

    # 6. ENSAMBLAJE
    print("\n[6/7] FASE C: Ensamblando la pista cronológicamente...")
    audio_final = AudioSegment.empty()
    tiempo_actual_ms = 0

    for i in range(len(textos_preparados)):
        segmento_tts = audios_procesados.get(i)
        if not segmento_tts:
            continue

        inicio_objetivo_ms = int(textos_preparados[i]["timestamp"][0] * 1000)

        if i == 0 and inicio_objetivo_ms < MIN_INICIO_MS:
            inicio_objetivo_ms = MIN_INICIO_MS

        gap_ms = inicio_objetivo_ms - tiempo_actual_ms

        if gap_ms > UMBRAL_CONTINUIDAD_MS:
            # Hueco real (pausa en el vídeo original): se respeta tal cual.
            audio_final += AudioSegment.silent(duration=gap_ms)
            audio_final += segmento_tts
        else:
            # Gap pequeño/negativo: se trata como continuidad -> crossfade en vez
            # de silencio seco, para que no suene cortado entre frases seguidas.
            if len(audio_final) > 0:
                crossfade = min(CROSSFADE_MS, len(segmento_tts) - 1, len(audio_final) - 1)
                crossfade = max(crossfade, 0)
                audio_final = audio_final.append(segmento_tts, crossfade=crossfade)
            else:
                audio_final += segmento_tts

        tiempo_actual_ms = len(audio_final)

    audio_final.export(AUDIO_SALIDA, format="wav")
    if os.path.exists(AUDIO_TEMPORAL):
        os.remove(AUDIO_TEMPORAL)

    # 7. RENDERIZADO FINAL
    print("\n[7/7] Inyectando el nuevo audio limpio en el vídeo...")
    comando_merge = [
        "ffmpeg", "-i", VIDEO_ENTRADA, "-i", AUDIO_SALIDA,
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
        "-c:a", "aac", "-b:a", "192k", "-y", VIDEO_FINAL
    ]

    res = subprocess.run(comando_merge, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"\n¡PIPELINE COMPLETADO! Vídeo generado: '{VIDEO_FINAL}'")
        if fallidos:
            print(f"[!] Recuerda: {len(fallidos)} bloque(s) quedaron sin doblar (ver aviso arriba).")
    else:
        print(f"\n[ERROR] FFmpeg falló:\n{res.stderr}")


if __name__ == "__main__":
    main()