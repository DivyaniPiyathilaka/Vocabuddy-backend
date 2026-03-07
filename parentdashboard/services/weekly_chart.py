"""
Weekly progress chart: Monday–Sunday weeks, last 4 weeks, label format "4–10 Mar".
Used for the "කාලයත් සමග ප්‍රගතිය" (Progress with time) line chart.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


def _week_monday(d: datetime) -> datetime:
    """Return the Monday 00:00:00 of the week containing d (ISO week: Monday–Sunday)."""
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    # weekday(): Monday=0, Sunday=6
    days_since_monday = d.weekday()
    monday = d - timedelta(days=days_since_monday)
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def _week_label(monday: datetime) -> str:
    """Format week as '4–10 Mar' (same month) or '28 Feb–6 Mar' (span)."""
    sunday = monday + timedelta(days=6)
    if monday.month == sunday.month:
        return f"{monday.day}–{sunday.day} {monday.strftime('%b')}"
    return f"{monday.day} {monday.strftime('%b')}–{sunday.day} {sunday.strftime('%b')}"


def build_weekly_trend_with_dates_last_4_weeks(
    attempts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Group attempts by calendar week (Monday–Sunday), take the last 4 weeks,
    and return one point per week with label "4–10 Mar" and accuracy %.

    attempts: list of dicts with keys "date" (datetime) and "iscorrect" (bool).
    Returns: [{"date": "4–10 Mar", "accuracy": 72.5}, ...] oldest week first.
    """
    if not attempts:
        return []

    now = datetime.now(timezone.utc)
    this_week_monday = _week_monday(now)
    # Last 4 weeks: current week + 3 previous (by Monday date)
    week_mondays: List[datetime] = []
    for i in range(4):
        monday = this_week_monday - timedelta(weeks=i)
        week_mondays.append(monday)
    week_mondays.reverse()  # oldest first

    # Bucket attempts by week (Monday as key)
    buckets: Dict[datetime, Dict[str, int]] = {}
    for attempt in attempts:
        dt = attempt.get("date")
        if dt is None:
            continue
        if hasattr(dt, "to_datetime"):
            dt = dt.to_datetime()
        if not isinstance(dt, datetime):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        mon = _week_monday(dt)
        if mon not in week_mondays:
            continue
        bucket = buckets.setdefault(mon, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if bool(attempt.get("iscorrect")):
            bucket["correct"] += 1

    result: List[Dict[str, Any]] = []
    for monday in week_mondays:
        bucket = buckets.get(monday, {"total": 0, "correct": 0})
        total = bucket["total"]
        correct = bucket["correct"]
        accuracy = (correct / total * 100.0) if total > 0 else 0.0
        label = _week_label(monday)
        result.append({"date": label, "accuracy": round(accuracy, 2)})
    return result
