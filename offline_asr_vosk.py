import argparse
import json
import logging
import os
import queue
import re
import shutil
import sys
import time
import wave
import zipfile
from typing import List, Optional

import numpy as np

# Необязательные зависимости
try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from urllib.request import urlopen
except Exception:
    urlopen = None

from vosk import KaldiRecognizer, Model, SetLogLevel


# ---------------------- ЛОГИРОВАНИЕ ----------------------

def setup_logging(log_file: Optional[str] = None):
    level = logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    logging.basicConfig(level=level, format='[%(asctime)s] %(levelname)s: %(message)s', handlers=handlers)


logger = logging.getLogger(__name__)


# ---------------------- УТИЛИТЫ МОДЕЛЕЙ ----------------------

def _is_valid_model_dir(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    expected = ['am', 'conf']  # простая эвристика
    existing = set(os.listdir(model_dir))
    return all(x in existing for x in expected)


def _is_url(s: str) -> bool:
    return s.startswith('http://') or s.startswith('https://')


def _target_dir_from_zip_name(zip_path_or_url: str) -> str:
    base = os.path.basename(zip_path_or_url)
    if base.endswith('.zip'):
        base = base[:-4]
    return os.path.join('models', base)


def extract_model_from_zip(zip_path: str, target_dir: str) -> str:
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    tmp_extract_parent = target_dir + '_extract'
    if os.path.exists(tmp_extract_parent):
        shutil.rmtree(tmp_extract_parent, ignore_errors=True)
    os.makedirs(tmp_extract_parent, exist_ok=True)

    logger.info('Распаковываю модель из %s ...', zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_extract_parent)

    entries = os.listdir(tmp_extract_parent)
    if len(entries) != 1:
        candidate = None
        for e in entries:
            if e.startswith('vosk-model'):
                candidate = e
                break
        top_dir = os.path.join(tmp_extract_parent, candidate or entries[0])
    else:
        top_dir = os.path.join(tmp_extract_parent, entries[0])

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    shutil.move(top_dir, target_dir)
    shutil.rmtree(tmp_extract_parent, ignore_errors=True)

    logger.info('Модель распакована: %s', target_dir)
    return target_dir


def prepare_model(model_arg: str) -> str:
    """Принимает путь к папке модели или URL/путь к ZIP.
    Возвращает путь к папке модели, готовой к использованию."""
    if not model_arg:
        raise ValueError('Нужно указать --model (путь к папке модели или URL/ZIP)')

    # 1) Папка
    if os.path.isdir(model_arg):
        if not _is_valid_model_dir(model_arg):
            raise FileNotFoundError(f'Папка модели некорректна: {model_arg}')
        return model_arg

    # 2) Локальный ZIP
    if os.path.isfile(model_arg) and model_arg.lower().endswith('.zip'):
        target_dir = _target_dir_from_zip_name(model_arg)
        return extract_model_from_zip(model_arg, target_dir)

    # 3) URL на ZIP
    if _is_url(model_arg):
        if urlopen is None:
            raise RuntimeError('Невозможно скачать URL: urlopen недоступен')
        target_dir = _target_dir_from_zip_name(model_arg)
        tmp_zip = target_dir + '.zip'
        os.makedirs(os.path.dirname(tmp_zip), exist_ok=True)
        logger.info('Скачиваю модель из %s ...', model_arg)
        with urlopen(model_arg) as resp, open(tmp_zip, 'wb') as out:
            total = int(resp.headers.get('Content-Length', '0')) or None
            read = 0
            chunk = 1024 * 1024
            last_print = time.time()
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                out.write(data)
                read += len(data)
                now = time.time()
                if total and now - last_print > 0.5:
                    percent = read * 100 // total
                    logger.info('Загрузка: %d%%', percent)
                    last_print = now
        model_dir = extract_model_from_zip(tmp_zip, target_dir)
        try:
            os.remove(tmp_zip)
        except Exception:
            pass
        return model_dir

    raise FileNotFoundError(f'--model не является ни папкой, ни ZIP, ни URL: {model_arg}')


# ---------------------- ГРАММАТИКА ----------------------

def load_grammar(grammar: Optional[str], grammar_file: Optional[str]) -> Optional[List[str]]:
    if grammar and grammar.strip():
        return [p.strip() for p in grammar.split(',') if p.strip()]
    if grammar_file and os.path.isfile(grammar_file):
        with open(grammar_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return None


# ---------------------- УСТРОЙСТВА ----------------------

def list_input_devices() -> List[dict]:
    if sd is None:
        raise RuntimeError('sounddevice недоступен')
    devices = sd.query_devices()
    res = []
    for i, d in enumerate(devices):
        if d.get('max_input_channels', 0) > 0:
            info = dict(index=i, name=d.get('name'),
                        default_samplerate=d.get('default_samplerate'),
                        max_input_channels=d.get('max_input_channels'))
            res.append(info)
    return res


# ---------------------- РАСПОЗНАВАНИЕ ----------------------

def make_recognizer(model_path: str, sample_rate: int, grammar_list: Optional[List[str]], max_alternatives: int) -> KaldiRecognizer:
    model = Model(model_path)
    if grammar_list:
        rec = KaldiRecognizer(model, sample_rate, json.dumps(grammar_list, ensure_ascii=False))
    else:
        rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)
    if max_alternatives > 0:
        rec.SetMaxAlternatives(max_alternatives)
    return rec


def write_json(results: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def recognize_from_wav(model_path: str, wav_path: str, grammar_list: Optional[List[str]], max_alternatives: int) -> List[dict]:
    wf = wave.open(wav_path, 'rb')
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise ValueError("Ожидается WAV: моно (1 канал), 16‑бит PCM. Сконвертируйте через ffmpeg.")
    sample_rate = wf.getframerate()

    rec = make_recognizer(model_path, sample_rate, grammar_list, max_alternatives)

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    final = json.loads(rec.FinalResult())
    if final and final.get('text'):
        results.append(final)
    return results


# Захват с микрофона: пробуем RawInputStream (int16),
# при ошибке — fallback на InputStream (float32 -> int16)

def recognize_from_mic(model_path: str, device: Optional[int], samplerate: Optional[int], blocksize: int,
                        grammar_list: Optional[List[str]], max_alternatives: int,
                        save_json: Optional[str] = None) -> List[dict]:
    if sd is None:
        raise RuntimeError('sounddevice не установлен или недоступен')

    # samplerate
    if samplerate is None:
        device_info = sd.query_devices(device, 'input') if device is not None else sd.query_devices(kind='input')
        samplerate = int(device_info['default_samplerate'])

    rec = make_recognizer(model_path, samplerate, grammar_list, max_alternatives)

    q = queue.Queue()

    def raw_callback(indata, frames, time_info, status):
        if status:
            logger.warning('SoundDevice status: %s', status)
        q.put(bytes(indata))

    def float_callback(indata, frames, time_info, status):
        if status:
            logger.warning('SoundDevice status: %s', status)
        x = (np.clip(indata, -1.0, 1.0) * 32767.0).astype(np.int16)
        q.put(x.tobytes())

    results: List[dict] = []
    partial_printed = ''

    def capture_loop(open_stream):
        nonlocal partial_printed
        with open_stream:
            print("Начинаю запись. Говорите... (Ctrl+C для выхода)")
            try:
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        if res.get('text'):
                            print("[ASR]", res['text'])
                        results.append(res)
                    else:
                        partial = json.loads(rec.PartialResult())
                        if partial.get('partial') and partial['partial'] != partial_printed:
                            partial_printed = partial['partial']
                            print("[...]", partial_printed[:120], end='\r', flush=True)
            except KeyboardInterrupt:
                print("\nОстанавливаюсь...")
                final = json.loads(rec.FinalResult())
                if final and final.get('text'):
                    results.append(final)

    # Пытаемся RawInputStream, иначе fallback на float
    try:
        stream = sd.RawInputStream(samplerate=samplerate, blocksize=blocksize, device=device,
                                   dtype='int16', channels=1, callback=raw_callback)
        capture_loop(stream)
    except Exception as e:
        logger.warning('RawInputStream не удалось открыть (%s), пробую float32...', e)
        stream = sd.InputStream(samplerate=samplerate, blocksize=blocksize, device=device,
                                dtype='float32', channels=1, callback=float_callback)
        capture_loop(stream)

    if save_json:
        write_json(results, save_json)
        print(f"JSON сохранён: {save_json}")

    return results


# ---------------------- АРГУМЕНТЫ И MAIN ----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Офлайн распознавание речи на Vosk (микрофон/файл) — простой CLI")

    # --model обязателен и может быть ПАПКОЙ модели ИЛИ URL/ZIP
    p.add_argument('--model', required=True, help='Путь к папке модели Vosk или URL/путь к ZIP архиву модели')

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--mic', action='store_true', help='Распознавание с микрофона')
    src.add_argument('--from-file', help='Путь к WAV‑файлу (моно, 16‑бит PCM)')

    # Микрофон
    p.add_argument('--device', type=int, help='Индекс входного аудио‑устройства (см. --list-devices)')
    p.add_argument('--samplerate', type=int, help='Частота дискретизации (Гц), напр. 16000')
    p.add_argument('--blocksize', type=int, default=4000, help='Размер блока захвата (фреймы)')
    p.add_argument('--list-devices', action='store_true', help='Показать список входных устройств и выйти')

    # Распознавание
    p.add_argument('--grammar', help='Список фраз/слов через запятую для ограничения словаря (опционально)')
    p.add_argument('--grammar-file', help='Путь к файлу со словами/фразами (по одной строке)')
    p.add_argument('--max-alt', type=int, default=0, help='Число альтернативных гипотез (0 — выкл)')

    # Вывод и логи (уровни фиксированы, можно лишь указать файл)
    p.add_argument('--save-json', help='Сохранить результаты распознавания в JSON')
    p.add_argument('--log-file', help='Путь к файлу логов приложения')

    return p.parse_args()


def main():
    args = parse_args()

    setup_logging(args.log_file)
    SetLogLevel(0)

    if args.list_devices:
        if sd is None:
            print('sounddevice недоступен')
            return
        devices = list_input_devices()
        print('Входные устройства:')
        for d in devices:
            print(f"  [#{d['index']}] {d['name']} | rate={d['default_samplerate']} | ch={d['max_input_channels']}")
        return

    # Подготовка модели (папка или URL/ZIP)
    try:
        model_dir = prepare_model(args.model)
    except Exception as e:
        logger.exception('Не удалось подготовить модель: %s', e)
        print('Ошибка подготовки модели. Проверьте путь/URL к модели (можно указать .zip или распакованную папку).')
        return

    grammar_list = load_grammar(args.grammar, args.grammar_file)

    if args.mic:
        recognize_from_mic(model_dir, args.device, args.samplerate, args.blocksize,
                           grammar_list, args.max_alt,
                           save_json=args.save_json)
        return

    if args.from_file:
        results = recognize_from_wav(model_dir, args.from_file, grammar_list, args.max_alt)
        if args.save_json:
            write_json(results, args.save_json)
            print(f"JSON сохранён: {args.save_json}")
        final_text = ' '.join([r.get('text', '') for r in results]).strip()
        print('Итоговый текст:', final_text)
        return


if __name__ == '__main__':
    main()
