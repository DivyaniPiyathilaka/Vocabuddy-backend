"""Speech statistics service and mock repository for parent dashboard.

This module is intentionally self-contained and uses a simple service /
repository pattern so that a real Firestore-backed repository can replace
the mock implementation later without changing the FastAPI routes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

# In-memory cache for dashboard stats (charts): child_user_id -> (stats_dict, expiry_timestamp, last_activity_ts)
# When serving from cache we re-check last_activity; if DB changed we invalidate so updated data is used.
_dashboard_stats_cache: Dict[str, Tuple[Dict[str, Any], float, Optional[datetime]]] = {}
_DASHBOARD_STATS_TTL_SECONDS = 90  # 1.5 minutes

from parentdashboard.schemas.speech_stats import (
    PhonemeStat,
    SpeechStatsResponse,
    WeeklyProgressPoint,
)
from parentdashboard.data.firebase_client import get_firestore_client
from parentdashboard.services.weekly_chart import build_weekly_trend_with_dates_last_4_weeks
from firebase_admin import firestore

# Logged-in user's Firebase UID. Use for defaults when request does not send child_id.
# Replace with auth context when login is wired.
LOGGED_IN_USER_UID = "sUXK8GwJC6QNPQ7PCxSXEzT3TH63"

# Map legacy app child IDs to the child's Firebase UID (users/{uid}/sessions/...).
CHILD_ID_TO_FIREBASE_UID: Dict[str, str] = {
    "child_001": LOGGED_IN_USER_UID,
    "mock-child": LOGGED_IN_USER_UID,
}


def _resolve_child_uid(child_id: Optional[str]) -> str:
    """Return the Firestore user document ID for this child (Firebase UID or mapped)."""
    if not child_id or not child_id.strip():
        return ""
    cid = child_id.strip()
    return CHILD_ID_TO_FIREBASE_UID.get(cid, cid)


def _parse_firestore_datetime(value: Any) -> Optional[datetime]:
    """Convert Firestore timestamp or datetime to timezone-aware UTC datetime."""
    if value is None:
        return None
    if hasattr(value, "to_datetime"):
        dt = value.to_datetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class SpeechRecordRepository(Protocol):
    """Abstract repository for fetching raw speech records."""

    def get_speech_records(self, child_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a list of raw pronunciation records for a child.

        Each record must look like:
        {
            "word": "සපත්තුව",
            "is_correct": True,
            "timestamp": datetime  # Firestore Timestamp converted to datetime
        }
        """
        ...


@dataclass
class FirestoreSpeechRepository:
    """Firestore-backed repository for speech practice history.

    Reads from:
        users/{child_user_id}/sessions/{sessionId}/practice/{practiceId}/attempts
    with word_progress statuses: wrong, pending, success.
    """

    def get_speech_records(self, child_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return speech records for a child. child_id can be Firebase UID or legacy id (e.g. child_001)."""
        child_user_id = _resolve_child_uid(child_id)
        if not child_user_id:
            return []

        attempts = get_child_performance_data(child_user_id=child_user_id)

        records: List[Dict[str, Any]] = []
        for attempt in attempts:
            word = str(attempt.get("word", ""))
            is_correct = bool(attempt.get("iscorrect"))
            ts = attempt.get("date")

            # Ensure we have a datetime timestamp
            if hasattr(ts, "to_datetime"):
                ts_dt = ts.to_datetime()
            else:
                ts_dt = ts

            if not isinstance(ts_dt, datetime):
                continue

            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)

            records.append(
                {
                    "word": word,
                    "is_correct": is_correct,
                    "timestamp": ts_dt,
                }
            )

        records.sort(key=lambda r: r["timestamp"], reverse=True)
        return records


class SpeechStatsService:
    """Business logic for computing speech statistics from raw records."""

    def __init__(self, repository: SpeechRecordRepository) -> None:
        self._repository = repository

    def get_stats(self, child_id: Optional[str] = None) -> SpeechStatsResponse:
        records = self._repository.get_speech_records(child_id=child_id)
        return self._calculate_stats(records)

    def _calculate_stats(self, records: Iterable[Dict[str, Any]]) -> SpeechStatsResponse:
        records_list = list(records)

        # Limit analytics to the most recent 8 sessions (based on session date).
        # Each record's "timestamp" comes from the parent session's date.
        session_timestamps: List[datetime] = []
        seen_ts: set[datetime] = set()
        for rec in sorted(
            records_list,
            key=lambda r: r.get("timestamp") or datetime.min,
            reverse=True,
        ):
            ts = rec.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            if ts not in seen_ts:
                seen_ts.add(ts)
                session_timestamps.append(ts)

        latest_sessions = set(session_timestamps[:8])
        if latest_sessions:
            records_list = [
                r
                for r in records_list
                if isinstance(r.get("timestamp"), datetime)
                and r["timestamp"] in latest_sessions
            ]
        total_words = len(records_list)
        total_correct = sum(1 for r in records_list if bool(r.get("is_correct")))

        overall_accuracy = (
            (total_correct / total_words) * 100.0 if total_words > 0 else 0.0
        )

        phoneme_breakdown = self._calculate_phoneme_breakdown(records_list)
        weekly_progress = self._calculate_weekly_progress(records_list)

        return SpeechStatsResponse(
            overall_accuracy=round(overall_accuracy, 2),
            total_words=total_words,
            total_correct=total_correct,
            phoneme_breakdown=phoneme_breakdown,
            weekly_progress=weekly_progress,
        )

    def _calculate_phoneme_breakdown(
        self, records: List[Dict[str, Any]]
    ) -> List[PhonemeStat]:
        # Legacy phoneme breakdown (S/R/T/K/N) is no longer used in the
        # new analytics and AI flows, so we return an empty list to keep
        # the API shape without computing unused categories.
        return []

    def _calculate_weekly_progress(
        self, records: List[Dict[str, Any]]
    ) -> List[WeeklyProgressPoint]:
        """Aggregate accuracy per session date (progress over time).

        We assume records have already been limited to the most recent sessions.
        Each point represents one calendar date, labelled like "04 Feb".
        """
        if not records:
            return []

        # Group by calendar date
        daily_buckets: Dict[datetime.date, Dict[str, int]] = {}

        for rec in records:
            ts = rec.get("timestamp")
            if not isinstance(ts, datetime):
                continue

            day = ts.date()
            bucket = daily_buckets.setdefault(day, {"total": 0, "correct": 0})
            bucket["total"] += 1
            if bool(rec.get("is_correct")):
                bucket["correct"] += 1

        if not daily_buckets:
            return []

        # Sort dates chronologically
        sorted_days = sorted(daily_buckets.keys())

        points: List[WeeklyProgressPoint] = []
        for day in sorted_days:
            bucket = daily_buckets[day]
            total = bucket["total"]
            correct = bucket["correct"]
            accuracy = (correct / total * 100.0) if total > 0 else 0.0

            # Label by date, e.g. "04 Feb"
            label = day.strftime("%d %b")
            points.append(
                WeeklyProgressPoint(
                    date=label,
                    accuracy=round(accuracy, 2),
                    total_words=total,
                    correct_words=correct,
                )
            )

        return points

    def get_monthly_session_count(self, child_id: Optional[str] = None) -> int:
        """Count practice records from the 1st of the current month to now."""
        records = self._repository.get_speech_records(child_id=child_id)
        if not records:
            return 0

        now = datetime.now(timezone.utc)
        first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        count = 0
        for rec in records:
            ts = rec.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if first_of_month <= ts <= now:
                count += 1

        return count


def _get_attempts_from_session_ref(
    session_ref: Any, session_date: Optional[datetime]
) -> List[Dict[str, Any]]:
    """Get word attempts from a session doc (objects subcollection). session_date used for each attempt."""
    if session_date is None:
        return []
    result: List[Dict[str, Any]] = []
    objects_ref = session_ref.collection("objects")
    for obj_doc in objects_ref.stream():
        data = obj_doc.to_dict() or {}
        word_progress = data.get("word_progress") or []
        if word_progress:
            for item in word_progress:
                if not isinstance(item, dict):
                    continue
                status = (item.get("status") or "").strip().lower()
                if status not in ("success", "wrong"):
                    continue
                word_str = str(item.get("word", ""))
                result.append(
                    {"word": word_str, "iscorrect": status == "success", "date": session_date}
                )
        else:
            status = (data.get("status") or "").strip().lower()
            if status in ("success", "wrong"):
                result.append(
                    {"word": str(data.get("word", "")), "iscorrect": status == "success", "date": session_date}
                )
    return result


def _get_attempts_from_practice_ref(
    practice_doc: Any, session_date: Optional[datetime]
) -> List[Dict[str, Any]]:
    """Get word attempts from a practice doc (attempts subcollection or word_progress on doc).
    Supports two Firestore shapes:
    - attempt docs with word_progress array: [{"status": "success"|"wrong", "word": "..."}]
    - attempt docs with top-level status and word (one attempt per doc)
    """
    if session_date is None:
        return []
    result: List[Dict[str, Any]] = []
    practice_data = practice_doc.to_dict() or {}
    word_progress = practice_data.get("word_progress") or []
    attempts_ref = practice_doc.reference.collection("attempts")
    for attempt_doc in attempts_ref.stream():
        attempt_data = attempt_doc.to_dict() or {}
        # Shape 1: one attempt per doc with top-level status and word
        status_top = (attempt_data.get("status") or "").strip().lower()
        word_top = attempt_data.get("word")
        if status_top in ("success", "wrong") and word_top is not None:
            result.append(
                {
                    "word": str(word_top),
                    "iscorrect": status_top == "success",
                    "date": session_date,
                }
            )
            continue
        # Shape 2: attempt doc has word_progress array
        wp = attempt_data.get("word_progress") or word_progress
        for item in wp:
            if not isinstance(item, dict):
                continue
            status = (item.get("status") or "").strip().lower()
            if status not in ("success", "wrong"):
                continue
            result.append(
                {
                    "word": str(item.get("word", "")),
                    "iscorrect": status == "success",
                    "date": session_date,
                }
            )
    if not result and word_progress:
        for item in word_progress:
            if not isinstance(item, dict):
                continue
            status = (item.get("status") or "").strip().lower()
            if status not in ("success", "wrong"):
                continue
            result.append(
                {
                    "word": str(item.get("word", "")),
                    "iscorrect": status == "success",
                    "date": session_date,
                }
            )
    return result


def get_attempts_from_latest_practice_per_session(
    child_user_id: str,
) -> List[Dict[str, Any]]:
    """
    Return all word attempts from the latest practice in each session (last 30 days).
    For each session: reads practice subcollection, latest practice by created_at,
    then practice → attempts (word_progress: success/wrong). Used for නිවැරදි බව.
    Sessions with no practice subcollection fall back to session-level objects.
    """
    child_user_id = _resolve_child_uid(child_user_id)
    if not child_user_id:
        return []

    client = get_firestore_client()
    sessions_ref = (
        client.collection("users").document(child_user_id).collection("sessions")
    )
    session_docs = list(sessions_ref.stream())

    now = datetime.now(timezone.utc)
    cutoff_30_days = now - timedelta(days=30)

    # Build (created_at, session_doc); keep only sessions in the last 30 days
    session_list: List[Tuple[datetime, Any]] = []
    for session_doc in session_docs:
        data = session_doc.to_dict() or {}
        session_dt = _parse_firestore_datetime(data.get("created_at"))
        if session_dt is None:
            continue
        if cutoff_30_days <= session_dt <= now:
            session_list.append((session_dt, session_doc))

    result: List[Dict[str, Any]] = []
    for _dt, session_doc in session_list:
        data = session_doc.to_dict() or {}
        session_dt = _parse_firestore_datetime(data.get("created_at"))

        practice_ref = session_doc.reference.collection("practice")
        practice_docs = list(practice_ref.stream())
        if practice_docs:
            # Has practice subcollection: take latest by created_at
            with_dates: List[Tuple[datetime, Any]] = []
            for pdoc in practice_docs:
                pdata = pdoc.to_dict() or {}
                p_dt = _parse_firestore_datetime(pdata.get("created_at"))
                if p_dt is None:
                    continue
                with_dates.append((p_dt, pdoc))
            if with_dates:
                with_dates.sort(key=lambda x: x[0], reverse=True)
                practice_dt, latest_practice = with_dates[0]
                result.extend(
                    _get_attempts_from_practice_ref(latest_practice, practice_dt)
                )
        else:
            # No practice subcollection: session is the practice (use objects)
            result.extend(
                _get_attempts_from_session_ref(session_doc.reference, session_dt)
            )
    return result


def get_accuracy_from_latest_practice_per_session(child_user_id: str) -> float:
    """
    නිවැරදි බව: total percentage of correct words over the last 30 days.

    For each session in the last 30 days, takes the latest practice from the
    session's practice subcollection; collects all word attempts (success/wrong
    from practice → attempts → word_progress). Then: (total correct) / (total words) × 100.
    Sessions without a practice subcollection fall back to session-level objects if present.
    """
    attempts = get_attempts_from_latest_practice_per_session(child_user_id)
    if not attempts:
        return 0.0
    total = len(attempts)
    correct = sum(1 for a in attempts if bool(a.get("iscorrect")))
    return (correct / total * 100.0) if total > 0 else 0.0


def get_average_accuracy_per_session_last_30_days(child_user_id: str) -> float:
    """
    නිවැරදි බව for card: average of each session's accuracy (latest practice only, last 30 days).
    Each session contributes one value (its latest practice accuracy); return the mean of those.
    """
    child_user_id = _resolve_child_uid(child_user_id)
    if not child_user_id:
        return 0.0

    client = get_firestore_client()
    sessions_ref = (
        client.collection("users").document(child_user_id).collection("sessions")
    )
    session_docs = list(sessions_ref.stream())
    now = datetime.now(timezone.utc)
    cutoff_30_days = now - timedelta(days=30)

    session_list: List[Tuple[datetime, Any]] = []
    for session_doc in session_docs:
        data = session_doc.to_dict() or {}
        session_dt = _parse_firestore_datetime(data.get("created_at"))
        if session_dt is None:
            continue
        if cutoff_30_days <= session_dt <= now:
            session_list.append((session_dt, session_doc))

    if not session_list:
        return 0.0
    accuracies = [_accuracy_from_session(session_doc.reference)[0] for _dt, session_doc in session_list]
    return sum(accuracies) / len(accuracies) if accuracies else 0.0


def get_monthly_practice_count(child_user_id: str) -> int:
    """Count of practices done in the current calendar month (session docs + practice docs with created_at in month)."""
    child_user_id = _resolve_child_uid(child_user_id)
    if not child_user_id:
        return 0
    now = datetime.now(timezone.utc)
    first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    client = get_firestore_client()
    sessions_ref = (
        client.collection("users").document(child_user_id).collection("sessions")
    )
    count = 0
    for session_doc in sessions_ref.stream():
        data = session_doc.to_dict() or {}
        session_dt = _parse_firestore_datetime(data.get("created_at"))
        if session_dt is not None and first_of_month <= session_dt <= now:
            count += 1
        practice_ref = session_doc.reference.collection("practice")
        for practice_doc in practice_ref.stream():
            pdata = practice_doc.to_dict() or {}
            p_dt = _parse_firestore_datetime(pdata.get("created_at"))
            if p_dt is not None and first_of_month <= p_dt <= now:
                count += 1
    return count


def get_practice_count_last_7_days(child_user_id: str) -> int:
    """Count of practices done in the last 7 days (session docs + practice docs with created_at in window)."""
    child_user_id = _resolve_child_uid(child_user_id)
    if not child_user_id:
        return 0
    now = datetime.now(timezone.utc)
    cutoff_7_days = now - timedelta(days=7)

    client = get_firestore_client()
    sessions_ref = (
        client.collection("users").document(child_user_id).collection("sessions")
    )
    count = 0
    for session_doc in sessions_ref.stream():
        data = session_doc.to_dict() or {}
        session_dt = _parse_firestore_datetime(data.get("created_at"))
        if session_dt is not None and cutoff_7_days <= session_dt <= now:
            count += 1
        practice_ref = session_doc.reference.collection("practice")
        for practice_doc in practice_ref.stream():
            pdata = practice_doc.to_dict() or {}
            p_dt = _parse_firestore_datetime(pdata.get("created_at"))
            if p_dt is not None and cutoff_7_days <= p_dt <= now:
                count += 1
    return count


def get_target_sounds_last_4_sessions(child_user_id: str) -> List[str]:
    """ඉලක්ක ශබ්ද: request.letter from each session created within the last 30 days (newest first)."""
    child_user_id = _resolve_child_uid(child_user_id)
    if not child_user_id:
        return []

    client = get_firestore_client()
    sessions_ref = (
        client.collection("users").document(child_user_id).collection("sessions")
    )
    now = datetime.now(timezone.utc)
    cutoff_30_days = now - timedelta(days=30)
    session_infos: List[Tuple[datetime, str]] = []
    for session_doc in sessions_ref.stream():
        data = session_doc.to_dict() or {}
        session_dt = _parse_firestore_datetime(data.get("created_at"))
        if session_dt is None:
            continue
        if cutoff_30_days <= session_dt <= now:
            request = data.get("request") or {}
            letter = str(request.get("letter", "")).strip()
            session_infos.append((session_dt, letter))
    session_infos.sort(key=lambda x: x[0], reverse=True)
    return [letter for _dt, letter in session_infos]


def get_child_performance_data(
    child_user_id: str,
    session_docs: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Crawl Firestore path and return a flat list of word attempts.

    Path: users/{child_user_id}/sessions/{sessionId}/practice/{practiceId}/attempts.

    For each practice document, uses created_at as the date. For each attempt
    document, reads the word_progress array; each item has status
    ("wrong" | "pending" | "success") and word. Only success and wrong are
    included in the result (pending is excluded from accuracy).

    If session_docs is provided, uses that list instead of streaming sessions
    again (avoids redundant Firestore reads when caller already has session docs).

    Returns:
        List of dicts: [{'word': '...', 'iscorrect': True, 'date': datetime}, ...]
    """
    child_user_id = _resolve_child_uid(child_user_id)
    if not child_user_id:
        return []

    if session_docs is not None:
        session_iter: Iterable[Any] = session_docs
    else:
        client = get_firestore_client()
        sessions_ref = (
            client.collection("users").document(child_user_id).collection("sessions")
        )
        session_iter = sessions_ref.stream()

    result: List[Dict[str, Any]] = []

    for session_doc in session_iter:
        practice_ref = session_doc.reference.collection("practice")
        for practice_doc in practice_ref.stream():
            practice_data = practice_doc.to_dict() or {}
            session_date = _parse_firestore_datetime(practice_data.get("created_at"))
            if session_date is None:
                continue

            attempts_ref = practice_doc.reference.collection("attempts")
            for attempt_doc in attempts_ref.stream():
                attempt_data = attempt_doc.to_dict() or {}
                status_top = (attempt_data.get("status") or "").strip().lower()
                if status_top in ("success", "wrong") and attempt_data.get("word") is not None:
                    result.append(
                        {
                            "word": str(attempt_data.get("word", "")),
                            "iscorrect": status_top == "success",
                            "date": session_date,
                        }
                    )
                    continue
                word_progress = attempt_data.get("word_progress") or practice_data.get("word_progress") or []
                for item in word_progress:
                    if not isinstance(item, dict):
                        continue
                    status = (item.get("status") or "").strip().lower()
                    if status not in ("success", "wrong"):
                        continue
                    result.append(
                        {
                            "word": str(item.get("word", "")),
                            "iscorrect": status == "success",
                            "date": session_date,
                        }
                    )

    return result


def _practice_accuracy_from_attempts(practice_doc_ref: Any) -> float:
    """Compute accuracy from attempts (word_progress array or top-level status/word per doc)."""
    total = 0
    correct = 0
    attempts_ref = practice_doc_ref.collection("attempts")
    for attempt_doc in attempts_ref.stream():
        data = attempt_doc.to_dict() or {}
        status_top = (data.get("status") or "").strip().lower()
        if status_top in ("success", "wrong"):
            total += 1
            if status_top == "success":
                correct += 1
            continue
        word_progress = data.get("word_progress") or []
        for item in word_progress:
            if not isinstance(item, dict):
                continue
            status = (item.get("status") or "").strip().lower()
            if status == "success":
                correct += 1
                total += 1
            elif status == "wrong":
                total += 1
    if total == 0:
        practice_doc = practice_doc_ref.get()
        if practice_doc.exists:
            practice_data = practice_doc.to_dict() or {}
            for item in practice_data.get("word_progress") or []:
                if not isinstance(item, dict):
                    continue
                status = (item.get("status") or "").strip().lower()
                if status == "success":
                    correct += 1
                    total += 1
                elif status == "wrong":
                    total += 1
    return (correct / total * 100.0) if total > 0 else 0.0


def _accuracy_from_session(session_ref: Any) -> Tuple[float, Optional[datetime]]:
    """
    Compute accuracy for a session doc. Tries 'practice' -> 'attempts' first (so we get
    practice created_at for the label), then 'objects' if no practice data.
    Returns (accuracy, date_for_label): date_for_label is the practice created_at when value
    comes from a practice; None when from objects only.
    """
    # Try practice -> attempts first so we get the practice date for the chart label (same as value)
    practice_ref = session_ref.collection("practice")
    best_acc = 0.0
    best_dt: Optional[datetime] = None
    for practice_doc in practice_ref.stream():
        pdata = practice_doc.to_dict() or {}
        p_dt = _parse_firestore_datetime(
            pdata.get("created_at") or pdata.get("createdAt")
        )
        p_acc = _practice_accuracy_from_attempts(practice_doc.reference)
        if p_acc > 0:
            if p_dt is not None and (best_dt is None or p_dt > best_dt):
                best_acc = p_acc
                best_dt = p_dt
            elif best_dt is None and best_acc == 0.0:
                best_acc = p_acc
    if best_acc > 0:
        return (best_acc, best_dt)
    # Fallback: objects subcollection (no per-attempt date, caller will use session date)
    total = 0
    correct = 0
    objects_ref = session_ref.collection("objects")
    for obj_doc in objects_ref.stream():
        data = obj_doc.to_dict() or {}
        word_progress = data.get("word_progress") or []
        if word_progress:
            for item in word_progress:
                if not isinstance(item, dict):
                    continue
                status = (item.get("status") or "").strip().lower()
                if status == "success":
                    correct += 1
                    total += 1
                elif status == "wrong":
                    total += 1
        else:
            status = (data.get("status") or "").strip().lower()
            if status in ("success", "wrong"):
                total += 1
                if status == "success":
                    correct += 1
    if total > 0:
        return (correct / total * 100.0, None)
    return (0.0, None)


def get_latest_activity_timestamp(child_user_id: str) -> Optional[datetime]:
    """
    One cheap Firestore read: latest session created_at for this child.
    Used to invalidate cache when DB has new data (new session/practice) within TTL.
    """
    resolved = _resolve_child_uid(child_user_id) or child_user_id
    if not resolved:
        return None
    try:
        client = get_firestore_client()
        sessions_ref = (
            client.collection("users")
            .document(resolved)
            .collection("sessions")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(1)
        )
        for doc in sessions_ref.stream():
            data = doc.to_dict() or {}
            return _parse_firestore_datetime(data.get("created_at"))
    except Exception:
        return None
    return None


def _get_cached_dashboard_stats(child_user_id: str) -> Optional[Dict[str, Any]]:
    """Return cached dashboard stats if present, not expired, and DB unchanged; otherwise None."""
    now = time.time()
    if child_user_id not in _dashboard_stats_cache:
        return None
    data, expiry, last_ts = _dashboard_stats_cache[child_user_id]
    if now >= expiry:
        del _dashboard_stats_cache[child_user_id]
        return None
    current_ts = get_latest_activity_timestamp(child_user_id)
    if current_ts != last_ts:
        del _dashboard_stats_cache[child_user_id]
        return None
    return data


def _set_cached_dashboard_stats(
    child_user_id: str, data: Dict[str, Any], last_activity_ts: Optional[datetime] = None
) -> None:
    """Store dashboard stats in cache with TTL and last-activity timestamp for validation."""
    if last_activity_ts is None:
        last_activity_ts = get_latest_activity_timestamp(child_user_id)
    _dashboard_stats_cache[child_user_id] = (
        data,
        time.time() + _DASHBOARD_STATS_TTL_SECONDS,
        last_activity_ts,
    )


def get_dashboard_stats_cached(child_user_id: str) -> Dict[str, Any]:
    """Return dashboard stats for a child, using in-memory cache when valid and DB unchanged."""
    resolved = _resolve_child_uid(child_user_id) or child_user_id
    cached = _get_cached_dashboard_stats(resolved)
    if cached is not None:
        return cached
    result = get_dashboard_stats(child_user_id)
    _set_cached_dashboard_stats(resolved, result)
    return result


def get_dashboard_stats(child_user_id: str) -> Dict[str, Any]:
    """Compute child dashboard statistics from Firestore.

    Path: users/{child_user_id} for profile; users/{child_user_id}/sessions for sessions
    and practice/attempts/word_progress (status: wrong, pending, success).
    child_user_id can be Firebase UID or legacy id (mapped via CHILD_ID_TO_FIREBASE_UID).
    """
    child_user_id = _resolve_child_uid(child_user_id)
    empty_stats: Dict[str, Any] = {
        "child_name": "",
        "child_age": 0,
        "total_sessions": 0,
        "global_accuracy": 0.0,
        "practice_count_last_7_days": 0,
        "weekly_trend": [],
        "word_category_progress": [],
    }
    if not child_user_id:
        return empty_stats

    client = get_firestore_client()

    # Child metadata from users/{child_user_id}
    child_ref = client.collection("users").document(child_user_id)
    child_doc = child_ref.get()
    child_raw = child_doc.to_dict() or {} if child_doc.exists else {}
    child_name = str(child_raw.get("name", "")) if child_raw.get("name") is not None else ""
    child_age_raw = child_raw.get("age")
    try:
        child_age = int(child_age_raw) if child_age_raw is not None else 0
    except Exception:
        child_age = 0

    sessions_ref = (
        client.collection("users").document(child_user_id).collection("sessions")
    )
    session_docs = list(sessions_ref.stream())
    total_sessions = len(session_docs)

    # Reuse session_docs to avoid streaming sessions again (major perf win).
    attempts = get_child_performance_data(
        child_user_id=child_user_id, session_docs=session_docs
    )

    # Limit to most recent 8 practice dates
    session_dates: List[datetime] = []
    seen_dates: set[datetime] = set()
    for attempt in sorted(
        attempts, key=lambda a: a.get("date") or datetime.min, reverse=True
    ):
        dt = attempt.get("date")
        if not isinstance(dt, datetime):
            continue
        if dt not in seen_dates:
            seen_dates.add(dt)
            session_dates.append(dt)

    latest_sessions = set(session_dates[:8])
    if latest_sessions:
        attempts = [
            a
            for a in attempts
            if isinstance(a.get("date"), datetime) and a["date"] in latest_sessions
        ]

    total_words = len(attempts)
    total_correct = sum(1 for a in attempts if bool(a.get("iscorrect")))
    global_accuracy = (
        (total_correct / total_words) * 100.0 if total_words > 0 else 0.0
    )

    # Progress with time: week-by-week (Monday–Sunday, last 4 weeks, label e.g. "4–10 Mar")
    weekly_trend_with_dates = build_weekly_trend_with_dates_last_4_weeks(attempts)
    weekly_trend = [p["accuracy"] for p in weekly_trend_with_dates]

    # Word category: every session in the last 30 days; each bar = one session, value = latest practice accuracy for that session
    now_30 = datetime.now(timezone.utc)
    cutoff_30_days = now_30 - timedelta(days=30)
    session_infos: List[Tuple[datetime, Any, str]] = []
    for session_doc in session_docs:
        data = session_doc.to_dict() or {}
        session_dt = _parse_firestore_datetime(data.get("created_at"))
        if session_dt is None:
            continue
        if cutoff_30_days <= session_dt <= now_30:
            request = data.get("request") or {}
            letter = str(request.get("letter", "")).strip()
            session_infos.append((session_dt, session_doc.reference, letter))

    # Oldest first so the bar chart reads left-to-right chronologically
    session_infos.sort(key=lambda x: x[0])

    word_category_progress: List[Dict[str, Any]] = []
    for session_dt, session_ref, letter in session_infos:
        accuracy, practice_dt = _accuracy_from_session(session_ref)
        # Use the date of the practice that produced the value (last attempt), so label and value match
        label_dt = practice_dt if practice_dt is not None else session_dt
        label_date = label_dt.strftime("%d %b")  # e.g. "06 Mar"
        letter_part = letter or "—"
        label = f"{letter_part} ({label_date})"
        word_category_progress.append({"label": label, "value": round(accuracy, 2)})

    practice_count_7_days = get_practice_count_last_7_days(child_user_id)

    return {
        "child_name": child_name,
        "child_age": child_age,
        "total_sessions": int(total_sessions),
        "global_accuracy": round(global_accuracy, 2),
        "practice_count_last_7_days": practice_count_7_days,
        "weekly_trend": weekly_trend,
        "weekly_trend_with_dates": weekly_trend_with_dates,
        "word_category_progress": word_category_progress,
    }


def get_child_summary(child_id: str) -> str:
    """Build a concise child summary string for personalization.

    child_id is the child's Firebase UID (users/{child_id}/sessions/...).
    """
    repository = FirestoreSpeechRepository()
    stats_service = SpeechStatsService(repository=repository)

    stats = stats_service.get_stats(child_id=child_id)
    monthly_count = stats_service.get_monthly_session_count(child_id=child_id)

    dashboard = get_dashboard_stats_cached(child_user_id=child_id)
    word_categories = dashboard.get("word_category_progress", [])

    recent_sessions_summary_lines = []
    for item in word_categories:
        label = str(item.get("label", ""))
        value = float(item.get("value", 0.0))
        recent_sessions_summary_lines.append(f"- {label}: {value:.1f}% accuracy")

    recent_sessions_summary = (
        "\n".join(recent_sessions_summary_lines)
        if recent_sessions_summary_lines
        else "No recent target-word sessions found."
    )

    summary = (
        f"Child Summary for {child_id}:\n"
        f"- Overall accuracy (recent sessions): {stats.overall_accuracy:.1f}%\n"
        f"- Total words practiced this month: {monthly_count}\n"
        f"- Recent target-word session performance:\n"
        f"{recent_sessions_summary}"
    )

    return summary

