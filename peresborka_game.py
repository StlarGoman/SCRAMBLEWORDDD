#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПЕРЕСБОРКА — консольная версия игры (Python + Groq API).

Логика полностью повторяет веб-версию:
  1. Есть заготовленные тексты.
  2. ИИ (Groq) перемешивает слова текста.
  3. Игрок собирает текст обратно, вводя номера слов в правильном порядке.
  4. ИИ оценивает сборку от 0 до 100; совпадение 1 в 1 даёт ровно 100.

Запуск в VS Code:
  - создайте рядом с файлом .env со строкой  GROQ_API_KEY=gsk_...
    (или задайте переменную окружения GROQ_API_KEY);
  - откройте файл и нажмите F5 / кнопку "Run",
    либо в терминале:  python peresborka_game.py

Зависимостей нет — только стандартная библиотека Python 3.10+.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------
# Настройки
# --------------------------------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
SCORES_FILE = Path(__file__).with_name("scores.json")


def load_api_key() -> str:
    """Берём ключ из переменной окружения или из .env рядом со скриптом."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


API_KEY = load_api_key()

# --------------------------------------------------------------------------
# Заготовленные тексты (те же, что и в веб-версии)
# --------------------------------------------------------------------------

TEXTS: list[tuple[str, str]] = [
    ("Проверенная мудрость", "Без труда не вытащишь и рыбку из пруда."),
    ("Утро в саду", "Ранняя пташка проснулась и запела в тихом утреннем саду."),
    (
        "Свет далёких звёзд",
        "Каждую ночь обсерватория на горе ловит свет далёких звёзд, "
        "который летел к нам тысячи лет сквозь холодную пустоту космоса.",
    ),
    (
        "Городская симфония",
        "Маленький город просыпался медленно: сначала звенела дверь пекарни, "
        "затем на площадь выезжали велосипедисты, а к восьми утра у старого "
        "фонтана уже играл уличный скрипач.",
    ),
    (
        "Старый маяк",
        "Старый маяк стоял на мысе уже сто лет, и каждый вечер, когда туман "
        "полз с моря, его жёлтый глаз медленно поворачивался к горизонту, "
        "будто проверяя, не заблудился ли кто-нибудь в тёмной воде.",
    ),
    (
        "Письмо в будущее",
        "Если однажды ты найдёшь это письмо в старой книге, знай: мы верили, "
        "что мир станет добрее, что люди научатся слушать друг друга, и что "
        "каждое утро будет начинаться с надежды, а не со спешки.",
    ),
]

# --------------------------------------------------------------------------
# Цвета для терминала (ANSI — работает в терминале VS Code)
# --------------------------------------------------------------------------


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    VIOLET = "\033[95m"
    GOLD = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"


def paint(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"


# --------------------------------------------------------------------------
# Работа с текстом: токены, метрики, локальная оценка
# --------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return [w for w in text.strip().split() if w]


def position_accuracy(original: list[str], attempt: list[str]) -> float:
    """Доля слов, стоящих ровно на своих позициях."""
    if not original:
        return 0.0
    hits = sum(
        1 for i, w in enumerate(original) if i < len(attempt) and attempt[i] == w
    )
    return hits / len(original)


def lcs_length(a: list[str], b: list[str]) -> int:
    """Длина наибольшей общей подпоследовательности (динамическое программирование)."""
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def order_accuracy(original: list[str], attempt: list[str]) -> float:
    """Доля слов, сохранивших взаимный порядок."""
    if not original:
        return 0.0
    return lcs_length(original, attempt) / len(original)


def local_base(original: list[str], attempt: list[str]) -> int:
    """Локальная 'якорная' оценка 0..100 — страхует оценку ИИ."""
    pos = position_accuracy(original, attempt)
    order = order_accuracy(original, attempt)
    return round(100 * (0.45 * pos + 0.55 * order))


def same_multiset(a: list[str], b: list[str]) -> bool:
    return sorted(a) == sorted(b)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


# --------------------------------------------------------------------------
# Клиент Groq (OpenAI-совместимый chat/completions, JSON-режим)
# --------------------------------------------------------------------------


def groq_chat_json(messages: list[dict], temperature: float, max_tokens: int) -> dict:
    """Один запрос к Groq с ответом строго в JSON."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        data = json.load(response)
    content = data["choices"][0]["message"]["content"] or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def shuffle_with_ai(words: list[str]) -> list[str] | None:
    """Просим ИИ перемешать слова. None — если модель ответила некорректно."""
    if not API_KEY:
        return None
    try:
        result = groq_chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        'Ты — движок игры-головоломки. Перемешай слова в СЛУЧАЙНОМ '
                        "порядке: ровно те же слова, ничего не добавляй и не удаляй, "
                        "слова не изменяй. Ответ — строго JSON "
                        '{"words": [...]} без пояснений.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Перемешай ровно эти {len(words)} слов: {json.dumps(words, ensure_ascii=False)}",
                },
            ],
            temperature=1.0,
            max_tokens=1024,
        )
        shuffled = [str(w) for w in result.get("words", [])]
        if not same_multiset(words, shuffled) or shuffled == words:
            return None
        return shuffled
    except Exception:
        return None


def judge_with_ai(original: str, attempt: str, exact: bool) -> dict | None:
    """ИИ-судья: {'score': int, 'verdict': str, 'feedback': str} или None."""
    if not API_KEY:
        return None
    try:
        result = groq_chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты — строгий, но обаятельный судья русской игры «Пересборка». "
                        "Сравни ОРИГИНАЛ с ВЕРСИЕЙ ИГРОКА и оцени 0–100: порядок слов, "
                        "смысл, грамотность. 100 — только за полное дословное совпадение. "
                        "Верни строго JSON: {\"score\": число, \"verdict\": \"2–4 слова\", "
                        "\"feedback\": \"2–3 предложения на русском, обращение на «ты»\"}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"ОРИГИНАЛ:\n{original}\n\nВЕРСИЯ ИГРОКА:\n{attempt}\n\n"
                        + (
                            "ПОДСКАЗКА: совпадение дословное — поставь 100 и поздравь."
                            if exact
                            else "ПОДСКАЗКА: совпадение НЕ дословное, 100 ставить нельзя."
                        )
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=600,
        )
        score = int(result.get("score"))
        return {
            "score": clamp(score, 0, 100),
            "verdict": str(result.get("verdict", "Хорошая попытка"))[:60],
            "feedback": str(result.get("feedback", "Сборка учтена."))[:600],
        }
    except Exception:
        return None


# --------------------------------------------------------------------------
# Таблица рекордов (JSON-файл рядом со скриптом)
# --------------------------------------------------------------------------


def load_scores() -> list[dict]:
    if SCORES_FILE.exists():
        try:
            return json.loads(SCORES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_score(entry: dict) -> int:
    """Сохраняет результат, возвращает место в общем зачёте (1 = лучший)."""
    scores = load_scores()
    scores.append(entry)
    scores.sort(key=lambda s: (s["score"], -s["seconds"]), reverse=True)
    SCORES_FILE.write_text(
        json.dumps(scores[:50], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sum(1 for s in scores if s["score"] > entry["score"]) + 1


def show_leaderboard() -> None:
    scores = load_scores()[:10]
    print(paint("\n  ── ТАБЛИЦА РЕКОРДОВ ─────────────────", C.DIM))
    if not scores:
        print(paint("  Пока пусто — стань первым!", C.DIM))
        return
    for i, s in enumerate(scores, 1):
        mark = paint(" 1В1", C.GOLD) if s.get("exact") else ""
        rank_txt = paint(f"{i:>2}", C.GOLD if i == 1 else C.DIM)
        score_txt = paint(f"{s['score']:>3}", C.BOLD)
        print(
            f"  {rank_txt} {s['name']:<16} {score_txt} "
            f"{C.DIM}{s['text'][:24]:<26} {s['seconds']}с{C.RESET}{mark}"
        )


# --------------------------------------------------------------------------
# Игровой цикл
# --------------------------------------------------------------------------


def play_round(title: str, content: str, player: str) -> None:
    original = tokenize(content)

    # 1. ИИ перемешивает (с локальным фолбэком)
    print(paint("\n  ИИ перемешивает слова…", C.VIOLET))
    shuffled = shuffle_with_ai(original)
    by_ai = shuffled is not None
    if not by_ai:
        shuffled = original[:]
        while shuffled == original:
            random.shuffle(shuffled)
    print(
        paint(
            f"  Перемешано {'ИИ' if by_ai else 'локально'}. "
            f"Собери текст из {len(original)} слов!",
            C.DIM,
        )
    )

    # 2. Сборка: игрок вводит номера слов в нужном порядке
    started = time.time()
    attempt: list[str] | None = None
    moves = 0
    while attempt is None:
        print()
        for i, word in enumerate(shuffled, 1):
            print(f"    {paint(f'{i:>2}', C.CYAN)}  {word}")
        print(
            paint(
                "\n  Введи номера слов через пробел в правильном порядке.\n"
                "  Команды: «п» — перетасовать порядок показа, «с» — сдаться.",
                C.DIM,
            )
        )
        raw = input(paint("  > ", C.BOLD)).strip().lower()
        moves += 1

        if raw == "с":
            print(paint(f"\n  Оригинал был: {content}", C.DIM))
            return
        if raw == "п":
            random.shuffle(shuffled)
            continue

        try:
            numbers = [int(x) for x in re.split(r"[\s,]+", raw) if x]
        except ValueError:
            print(paint("  Нужны только номера слов через пробел.", C.RED))
            continue
        if sorted(numbers) != list(range(1, len(shuffled) + 1)):
            print(
                paint(
                    f"  Используй каждый номер от 1 до {len(shuffled)} ровно один раз.",
                    C.RED,
                )
            )
            continue
        attempt = [shuffled[i - 1] for i in numbers]
        print(paint("\n  Твоя сборка: ", C.DIM) + " ".join(attempt))
        if input(paint("  Проверить? [Enter — да, «н» — заново] ", C.BOLD)).strip().lower() == "н":
            attempt = None

    seconds = round(time.time() - started)

    # 3. Оценка: точное совпадение = 100, иначе ИИ-судья (зафиксированный локально)
    exact = attempt == original
    base = local_base(original, attempt)
    print(paint("\n  ИИ оценивает сборку…", C.VIOLET))
    judge = judge_with_ai(content, " ".join(attempt), exact)

    if exact:
        score = 100
        verdict = judge["verdict"] if judge else "Идеальное совпадение!"
        feedback = (
            judge["feedback"]
            if judge
            else "Текст собран слово в слово. Абсолютный результат!"
        )
    elif judge:
        score = clamp(judge["score"], max(0, base - 20), min(97, base + 20))
        verdict, feedback = judge["verdict"], judge["feedback"]
    else:
        score = min(97, base)
        verdict = "Хорошая попытка" if score >= 40 else "Текст развалился"
        feedback = "ИИ-судья недоступен, оценка выставлена локальным алгоритмом."

    # 4. Итог
    color = C.GOLD if exact else C.GREEN if score >= 80 else C.RED if score < 40 else C.CYAN
    print()
    print(paint(f"  ═══ ОЦЕНКА: {score}/100 — {verdict} ═══", color))
    print(f"  {feedback}")
    print(
        paint(
            f"\n  Слова на своих местах: {position_accuracy(original, attempt):.0%}   "
            f"Сохранённый порядок: {order_accuracy(original, attempt):.0%}   "
            f"Время: {seconds}с   Ходы: {moves}",
            C.DIM,
        )
    )
    if not exact:
        print(paint(f"\n  Оригинал: {content}", C.DIM))
    else:
        print(paint("  ★ Совпадение ОДИН В ОДИН!", C.GOLD))

    rank = save_score(
        {
            "name": player,
            "score": score,
            "exact": exact,
            "text": title,
            "seconds": seconds,
            "ts": time.strftime("%Y-%m-%d %H:%M"),
        }
    )
    print(paint(f"  Место в таблице: #{rank}", C.CYAN if rank <= 10 else C.DIM))
    show_leaderboard()


def main() -> None:
    print(paint("\n  ╔══════════════════════════════════╗", C.VIOLET))
    print(paint("  ║        П Е Р Е С Б О Р К А       ║", C.VIOLET))
    print(paint("  ║   ИИ перемешал — ты собираешь    ║", C.VIOLET))
    print(paint("  ╚══════════════════════════════════╝", C.VIOLET))
    if not API_KEY:
        print(
            paint(
                "\n  ВНИМАНИЕ: GROQ_API_KEY не найден — перемешивание и судейство\n"
                "  будут локальными. Добавь ключ в .env рядом со скриптом.",
                C.GOLD,
            )
        )

    player = input(paint("\n  Твоё имя [Игрок]: ", C.BOLD)).strip() or "Игрок"

    while True:
        print(paint("\n  ── ВЫБЕРИ ТЕКСТ ─────────────────────", C.DIM))
        for i, (title, content) in enumerate(TEXTS, 1):
            n = len(tokenize(content))
            level = "лёгкий" if n <= 10 else "средний" if n <= 24 else "сложный"
            print(f"    {paint(str(i), C.CYAN)}  {title} {C.DIM}({n} слов, {level}){C.RESET}")
        print(f"    {paint('т', C.CYAN)}  таблица рекордов")
        print(f"    {paint('в', C.CYAN)}  выход")

        choice = input(paint("  > ", C.BOLD)).strip().lower()
        if choice == "в":
            print(paint("  Пока! Возвращайся за идеальной соткой.", C.VIOLET))
            return
        if choice == "т":
            show_leaderboard()
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(TEXTS):
            title, content = TEXTS[int(choice) - 1]
            try:
                play_round(title, content, player)
            except KeyboardInterrupt:
                print(paint("\n  Раунд прерван.", C.DIM))
        else:
            print(paint("  Такого варианта нет.", C.RED))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(paint("\n  Выход.", C.DIM))
        sys.exit(0)
