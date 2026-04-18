"""Temporal context — provides current date/time to the model in Portuguese."""
from datetime import datetime


_WEEKDAYS_PT = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
_MONTHS_PT = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
               "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]


def _now_local() -> datetime:
    return datetime.now()


def format_date_pt(dt: datetime) -> str:
    day_name = _WEEKDAYS_PT[dt.weekday()]
    month_name = _MONTHS_PT[dt.month - 1]
    return f"{day_name}, {dt.day} de {month_name} de {dt.year}"


def _period_of_day(hour: int) -> str:
    if hour < 12:
        return "manhã"
    elif hour < 18:
        return "tarde"
    else:
        return "noite"


def build_temporal_context() -> str:
    """Build the temporal block injected into the system prompt (no cache)."""
    now = _now_local()
    date_str = format_date_pt(now)
    time_str = now.strftime("%H:%M")
    iso_str = now.strftime("%Y-%m-%d")
    period = _period_of_day(now.hour)

    lines = [
        "## Data e Hora (fonte única — não usar conhecimento de treino)",
        f"- Hoje: **{date_str}** ({iso_str})",
        f"- Hora local: {time_str} ({period})",
        "- Quando perguntarem que dia/data/hora/período é, usa APENAS as linhas acima.",
    ]
    return "\n".join(lines)
