from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

import requests

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
TELEGRAM_MAX_CHARS = 3800

DEFAULT_PROMPT = (
    "Sos un asistente de seguridad. Con los datos agregados de detecciones de "
    "camaras que te paso, redacta un informe claro y conciso en espanol. IMPORTANTE: "
    "varios patrones con el mismo tag son LA MISMA PERSONA, tratalos como una sola. "
    "Inclui: resumen general, personas mas frecuentes (agrupadas por tag), actividad "
    "por camara y por dia, y sobre todo marca cualquier PATRON ANOMALO o inusual "
    "(horarios raros, picos de actividad, personas desconocidas o no reconocidas, "
    "apariciones fuera de lo comun). Usa vinetas y se breve."
)

DEFAULT_CHAT_SYSTEM = (
    "Sos un analista de seguridad para Sentinex Vision. Responde en espanol, "
    "directo y con criterio operacional. Usa el contexto agregado de detecciones "
    "del rango de fechas indicado y el historial reciente del chat. Si los datos "
    "no alcanzan para concluir algo, dilo claramente. Al evaluar personas, varios "
    "patrones con el mismo tag son la misma persona. Distingue desconocidos, "
    "repeticiones, camaras, horarios y actividad inusual."
)


def report_config() -> Dict[str, Any]:
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "telegram": bool(
            os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
            and os.getenv("TELEGRAM_CHAT_ID", "").strip()
        ),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "defaultPrompt": DEFAULT_PROMPT,
    }


def build_summary(
    detections: List[Dict[str, Any]],
    identities: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
) -> str:
    total = len(detections)
    by_status: Counter = Counter()
    by_day: Counter = Counter()
    by_camera: Counter = Counter()
    per_pattern: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "scores": [], "first": "", "last": "", "cameras": set(), "name": "", "tag": ""}
    )
    # Agrupacion por persona: patrones con el mismo tag = la misma persona.
    per_person: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "scores": [], "patterns": set(), "cameras": set(), "first": "", "last": ""}
    )

    for det in detections:
        status = det.get("matchStatus", "") or "desconocido"
        by_status[status] += 1
        timestamp = str(det.get("timestamp", ""))
        day = timestamp[:10]
        if day:
            by_day[day] += 1
        camera = det.get("cameraId") or det.get("source") or "CAM"
        by_camera[camera] += 1

        key = det.get("identityId") or det.get("name") or "sin-id"
        bucket = per_pattern[key]
        bucket["count"] += 1
        bucket["name"] = det.get("name", bucket["name"]) or bucket["name"]
        bucket["tag"] = det.get("tag", bucket["tag"]) or bucket["tag"]
        score = det.get("score")
        if isinstance(score, (int, float)):
            bucket["scores"].append(float(score))
        if camera:
            bucket["cameras"].add(camera)
        if timestamp:
            if not bucket["first"] or timestamp < bucket["first"]:
                bucket["first"] = timestamp
            if not bucket["last"] or timestamp > bucket["last"]:
                bucket["last"] = timestamp

        tag = (det.get("tag") or "").strip()
        person_key = tag if tag else f"(sin tag) {det.get('name', '') or 'desconocido'}"
        person = per_person[person_key]
        person["count"] += 1
        person["patterns"].add(det.get("identityId") or det.get("name") or "")
        if isinstance(score, (int, float)):
            person["scores"].append(float(score))
        if camera:
            person["cameras"].add(camera)
        if timestamp:
            if not person["first"] or timestamp < person["first"]:
                person["first"] = timestamp
            if not person["last"] or timestamp > person["last"]:
                person["last"] = timestamp

    lines: List[str] = []
    lines.append(f"Rango: {start_date} a {end_date}")
    lines.append(f"Total detecciones: {total}")
    lines.append(f"Patrones/identidades totales en el sistema: {len(identities)}")

    if by_status:
        estado = ", ".join(f"{key}={value}" for key, value in by_status.most_common())
        lines.append(f"Por estado: {estado}")

    if by_camera:
        camaras = ", ".join(f"{key}={value}" for key, value in by_camera.most_common())
        lines.append(f"Por camara: {camaras}")

    if by_day:
        lines.append("Por dia:")
        for day, count in sorted(by_day.items()):
            lines.append(f"  {day}: {count}")

    if per_person:
        lines.append("Por persona (mismo tag = misma persona, agrupa varios patrones):")
        ranked_person = sorted(per_person.items(), key=lambda kv: kv[1]["count"], reverse=True)
        for person_name, item in ranked_person[:20]:
            scores = item["scores"]
            avg = sum(scores) / len(scores) if scores else 0.0
            cameras = ",".join(sorted(item["cameras"])) or "CAM"
            first = item["first"][11:19] if len(item["first"]) >= 19 else item["first"][:10]
            last = item["last"][11:19] if len(item["last"]) >= 19 else item["last"][:10]
            lines.append(
                f"  {person_name}: {item['count']} detecciones en {len(item['patterns'])} patron(es), "
                f"score prom {avg:.2f}, camaras {cameras}, primera {first}, ultima {last}"
            )

    if per_pattern:
        lines.append("Detalle por patron (top 20 por cantidad):")
        ranked = sorted(per_pattern.values(), key=lambda item: item["count"], reverse=True)
        for item in ranked[:20]:
            scores = item["scores"]
            avg = sum(scores) / len(scores) if scores else 0.0
            name = item["name"] or "sin-nombre"
            tag = f" [{item['tag']}]" if item["tag"] else ""
            cameras = ",".join(sorted(item["cameras"])) or "CAM"
            first = item["first"][11:19] if len(item["first"]) >= 19 else item["first"][:10]
            last = item["last"][11:19] if len(item["last"]) >= 19 else item["last"][:10]
            lines.append(
                f"  {name}{tag}: {item['count']} detecciones, score prom {avg:.2f}, "
                f"camaras {cameras}, primera {first}, ultima {last}"
            )

    return "\n".join(lines)


def generate_report(summary: str, prompt: str, *, model: str, api_key: str, timeout: float = 90.0) -> str:
    system_prompt = (prompt or DEFAULT_PROMPT).strip()
    response = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Datos agregados de detecciones:\n\n{summary}"},
            ],
            "temperature": 0.4,
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI {response.status_code}: {response.text[:300]}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_chat_response(
    summary: str,
    message: str,
    history: List[Dict[str, str]],
    *,
    model: str,
    api_key: str,
    timeout: float = 90.0,
) -> str:
    full_history = [
        {
            "role": item.get("role", "user") if item.get("role") in {"user", "assistant"} else "user",
            "content": str(item.get("content", "")),
        }
        for item in history
        if str(item.get("content", "")).strip()
    ]
    
    chat_system_prompt = os.getenv("OPENAI_CHAT_SYSTEM_PROMPT", DEFAULT_CHAT_SYSTEM).strip()
    
    messages = []
    if chat_system_prompt:
        messages.append({"role": "system", "content": chat_system_prompt})
        
    messages.extend([
        {"role": "user", "content": f"Contexto actual de detecciones:\n\n{summary}"},
        *full_history,
        {"role": "user", "content": message.strip()},
    ])
    response = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.25,
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI {response.status_code}: {response.text[:300]}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def send_telegram(text: str, *, token: str, chat_id: str, timeout: float = 30.0) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chunk in _split_chunks(text, TELEGRAM_MAX_CHARS):
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True},
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Telegram {response.status_code}: {response.text[:300]}")


def _split_chunks(text: str, size: int) -> List[str]:
    text = text or ""
    if len(text) <= size:
        return [text] if text else [""]
    chunks: List[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > size and current:
            chunks.append(current)
            current = line
        elif len(candidate) > size:
            # una sola linea muy larga: cortar duro
            for i in range(0, len(line), size):
                chunks.append(line[i : i + size])
            current = ""
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
