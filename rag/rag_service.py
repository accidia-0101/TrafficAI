from sentence_transformers import SentenceTransformer
from pgvector.django import CosineDistance
from events.models import Event, Camera
from django.db.models import Count, Q
from django.db.models.functions import TruncDate
from datetime import datetime, timedelta
from django.utils import timezone
import re

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True)
    return _embedder


def parse_time_filter(query: str):
    query_lower = query.lower()
    now = timezone.now()
    start_time = None
    end_time = None

    specific_date_match = re.search(r"on (\d{4}-\d{2}-\d{2})", query_lower)
    if specific_date_match:
        date_str = specific_date_match.group(1)
        specific_date = datetime.strptime(date_str, "%Y-%m-%d")
        specific_date = timezone.make_aware(specific_date)
        start_time = specific_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = specific_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif "today" in query_lower:
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = now
    elif "yesterday" in query_lower:
        yesterday = now - timedelta(days=1)
        start_time = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif "last 24 hours" in query_lower or "past 24 hours" in query_lower:
        start_time = now - timedelta(hours=24)
        end_time = now
    elif "last two hours" in query_lower or "last 2 hours" in query_lower:
        start_time = now - timedelta(hours=2)
        end_time = now
    elif "last two weeks" in query_lower or "last 2 weeks" in query_lower or "past two weeks" in query_lower:
        start_time = now - timedelta(weeks=2)
        end_time = now
    elif "this week" in query_lower and "last week" not in query_lower:
        start_time = now - timedelta(days=now.weekday())
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = now
    elif "last week" in query_lower and "this week" not in query_lower:
        start_of_this_week = now - timedelta(days=now.weekday())
        start_of_this_week = start_of_this_week.replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = start_of_this_week - timedelta(days=7)
        end_time = start_of_this_week - timedelta(seconds=1)
    elif "this month" in query_lower and "last month" not in query_lower:
        start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_time = now
    elif "last month" in query_lower and "this month" not in query_lower:
        first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_time = first_of_this_month - timedelta(seconds=1)
        start_time = (first_of_this_month - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif "past month" in query_lower or "last 30 days" in query_lower:
        start_time = now - timedelta(days=30)
        end_time = now
    else:
        weeks_match = re.search(r"(?:last|past)\s+(\d+)\s+weeks?", query_lower)
        if weeks_match:
            weeks = int(weeks_match.group(1))
            start_time = now - timedelta(weeks=weeks)
            end_time = now
        else:
            days_match = re.search(r"(?:last|past|in)\s+(\d+)\s+days?", query_lower)
            if days_match:
                days = int(days_match.group(1))
                start_time = now - timedelta(days=days)
                end_time = now

    time_range_match = re.search(r"between (\d{1,2}:\d{2}) and (\d{1,2}:\d{2})", query_lower)
    hour_start, hour_end = None, None
    if time_range_match:
        hour_start = datetime.strptime(time_range_match.group(1), "%H:%M").time()
        hour_end = datetime.strptime(time_range_match.group(2), "%H:%M").time()

    return start_time, end_time, hour_start, hour_end


def parse_camera_filter(query: str):
    query_lower = query.lower()
    cameras = re.findall(r"camera\s*(\d+)", query_lower)
    if cameras:
        result = []
        for cam_num in cameras:
            found = False
            for cam in Camera.objects.values_list("camera_id", flat=True):
                cam_str = str(cam).lower().replace(" ", "").replace("_", "").replace("-", "")
                if cam_str == cam_num or cam_str == f"camera{cam_num}":
                    result.append(cam)
                    found = True
                    break
            if not found:
                result.append(cam_num)
        return result if len(result) > 1 else (result[0] if result else None)
    for cam in Camera.objects.values_list("camera_id", flat=True):
        if str(cam).lower() in query_lower:
            return cam
    return None


def parse_weather_filter(query: str):
    query_lower = query.lower()
    weathers = []
    if "rain" in query_lower:
        weathers.append("rainy")
    if "clear" in query_lower or "sunny" in query_lower:
        weathers.append("clear")
    return weathers if weathers else None


def parse_event_type(query: str):
    query_lower = query.lower()
    if "near-miss" in query_lower or "near miss" in query_lower:
        return "near-miss"
    if "accident" in query_lower or "crash" in query_lower:
        return "accident"
    return None


def parse_confidence_filter(query: str):
    match = re.search(r"confidence.*?(?:below|under|less than)\s*([\d.]+)", query.lower())
    if match:
        return ("lt", float(match.group(1)))
    match = re.search(r"confidence.*?(?:above|over|greater than)\s*([\d.]+)", query.lower())
    if match:
        return ("gt", float(match.group(1)))
    return None


def detect_query_type(query: str):
    query_lower = query.lower()
    comparison_keywords = ["compare", "comparison", "versus", "vs", "more than", "less than",
                           "how many more", "how much", "decrease", "increase", "compared"]
    aggregation_keywords = ["how many", "count", "total", "overall", "which camera has",
                            "which days", "earliest", "most recent", "types of", "least confidence"]

    for kw in comparison_keywords:
        if kw in query_lower:
            return "comparison"
    for kw in aggregation_keywords:
        if kw in query_lower:
            return "aggregation"
    if any(kw in query_lower for kw in ["list", "show", "filter"]):
        return "filtered"
    return "factual"


def build_base_queryset(query: str):
    qs = Event.objects.all()

    cam_ids = parse_camera_filter(query)
    if cam_ids:
        if isinstance(cam_ids, list):
            qs = qs.filter(camera__camera_id__in=cam_ids)
        else:
            qs = qs.filter(camera__camera_id=cam_ids)

    start_time, end_time, hour_start, hour_end = parse_time_filter(query)
    if start_time:
        qs = qs.filter(timestamp__gte=start_time)
    if end_time:
        qs = qs.filter(timestamp__lte=end_time)
    if hour_start and hour_end:
        qs = qs.filter(timestamp__time__gte=hour_start, timestamp__time__lte=hour_end)

    weathers = parse_weather_filter(query)
    if weathers:
        weather_q = Q()
        for w in weathers:
            weather_q |= Q(weather__icontains=w)
        qs = qs.filter(weather_q)

    event_type = parse_event_type(query)
    if event_type:
        qs = qs.filter(type__iexact=event_type)

    conf_filter = parse_confidence_filter(query)
    if conf_filter:
        op, val = conf_filter
        if op == "lt":
            qs = qs.filter(confidence__lt=val)
        else:
            qs = qs.filter(confidence__gt=val)

    return qs


def get_aggregation_stats(query: str):
    query_lower = query.lower()
    stats = {}

    if "total" in query_lower or "how many" in query_lower:
        qs = build_base_queryset(query)
        stats["total_count"] = qs.count()

    if "earliest" in query_lower:
        qs = build_base_queryset(query)
        earliest = qs.order_by("timestamp").first()
        if earliest:
            stats["earliest_event"] = format_event(earliest)

    if "most recent" in query_lower or "latest" in query_lower:
        qs = build_base_queryset(query)
        latest = qs.order_by("-timestamp").first()
        if latest:
            stats["most_recent_event"] = format_event(latest)

    if "least confidence" in query_lower:
        qs = build_base_queryset(query)
        least_conf = qs.order_by("confidence").first()
        if least_conf:
            stats["least_confidence_event"] = format_event(least_conf)

    if "which camera" in query_lower and "more" in query_lower:
        qs = build_base_queryset(query)
        camera_counts = qs.values("camera__camera_id").annotate(
            count=Count("event_id")
        ).order_by("-count")
        stats["camera_comparison"] = list(camera_counts)

    if "types of weather" in query_lower:
        weather_types = Event.objects.exclude(
            weather__isnull=True
        ).exclude(weather='').values_list("weather", flat=True).distinct()
        stats["weather_types"] = list(weather_types)
    elif "types" in query_lower:
        event_types = Event.objects.values_list("type", flat=True).distinct()
        stats["event_types"] = list(event_types)

    if "which days" in query_lower and "no accident" in query_lower:
        start_time, end_time, _, _ = parse_time_filter(query)
        if start_time and end_time:
            days_with_events = set(
                Event.objects.filter(
                    timestamp__gte=start_time,
                    timestamp__lte=end_time
                ).annotate(date=TruncDate("timestamp")).values_list("date", flat=True)
            )
            all_days = set()
            current = start_time.date()
            while current <= end_time.date():
                all_days.add(current)
                current += timedelta(days=1)
            days_without = all_days - days_with_events
            stats["days_without_accidents"] = sorted(days_without)

    return stats


def get_comparison_stats(query: str):
    query_lower = query.lower()
    stats = {}
    now = timezone.now()

    if "this week" in query_lower and "last week" in query_lower:
        start_of_this_week = now - timedelta(days=now.weekday())
        start_of_this_week = start_of_this_week.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_last_week = start_of_this_week - timedelta(days=7)

        this_week_count = Event.objects.filter(
            timestamp__gte=start_of_this_week, timestamp__lte=now
        ).count()
        last_week_count = Event.objects.filter(
            timestamp__gte=start_of_last_week, timestamp__lt=start_of_this_week
        ).count()
        stats["this_week_count"] = this_week_count
        stats["last_week_count"] = last_week_count
        stats["week_difference"] = this_week_count - last_week_count

    if "this month" in query_lower and "last month" in query_lower:
        first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_end = first_of_this_month - timedelta(seconds=1)
        first_of_last_month = (first_of_this_month - timedelta(days=1)).replace(day=1)

        this_month_count = Event.objects.filter(
            timestamp__gte=first_of_this_month, timestamp__lte=now
        ).count()
        last_month_count = Event.objects.filter(
            timestamp__gte=first_of_last_month, timestamp__lte=last_month_end
        ).count()
        stats["this_month_count"] = this_month_count
        stats["last_month_count"] = last_month_count
        stats["month_difference"] = this_month_count - last_month_count

    if ("clear" in query_lower or "sunny" in query_lower) and "rain" in query_lower:
        start_time, end_time, _, _ = parse_time_filter(query)
        qs = Event.objects.all()
        if start_time:
            qs = qs.filter(timestamp__gte=start_time)
        if end_time:
            qs = qs.filter(timestamp__lte=end_time)

        clear_count = qs.filter(weather__icontains="clear").count()
        rain_count = qs.filter(weather__icontains="rainy").count()
        stats["clear_weather_count"] = clear_count
        stats["rain_count"] = rain_count

    if "daytime" in query_lower and "nighttime" in query_lower:
        start_time, end_time, _, _ = parse_time_filter(query)
        qs = Event.objects.all()
        if start_time:
            qs = qs.filter(timestamp__gte=start_time)
        if end_time:
            qs = qs.filter(timestamp__lte=end_time)

        daytime_count = 0
        nighttime_count = 0
        for event in qs:
            hour = event.timestamp.hour
            if 8 <= hour < 20:
                daytime_count += 1
            else:
                nighttime_count += 1
        stats["daytime_count"] = daytime_count
        stats["nighttime_count"] = nighttime_count

    if "camera 1" in query_lower and "camera 2" in query_lower:
        start_time, end_time, _, _ = parse_time_filter(query)
        qs = Event.objects.all()
        if start_time:
            qs = qs.filter(timestamp__gte=start_time)
        if end_time:
            qs = qs.filter(timestamp__lte=end_time)

        cam1_id = parse_camera_filter("camera 1")
        cam2_id = parse_camera_filter("camera 2")
        if isinstance(cam1_id, list):
            cam1_id = cam1_id[0]
        if isinstance(cam2_id, list):
            cam2_id = cam2_id[0]
        cam1_count = qs.filter(camera__camera_id=cam1_id).count() if cam1_id else 0
        cam2_count = qs.filter(camera__camera_id=cam2_id).count() if cam2_id else 0
        stats["camera1_count"] = cam1_count
        stats["camera2_count"] = cam2_count
        stats["camera_difference"] = cam1_count - cam2_count

    return stats


def format_event(e):
    return {
        "event_id": str(e.event_id),
        "timestamp": e.timestamp.isoformat(),
        "camera": e.camera.camera_id,
        "type": e.type,
        "weather": e.weather or "clear",
        "confidence": float(e.confidence),
        "text": e.evidence_text,
    }


def search_similar_events(query: str, top_k: int = 5):
    model = get_embedder()
    q_vec = model.encode(query, normalize_embeddings=True).tolist()

    qs = build_base_queryset(query)

    d = CosineDistance("embedding", q_vec)
    qs = qs.annotate(dist=d).filter(dist__lt=0.8).order_by("dist")[:top_k]

    results = []
    for e in qs:
        event_data = format_event(e)
        event_data["similarity"] = float(1 - e.dist)
        results.append(event_data)

    return results


def build_context_for_llm(query: str, top_k: int = 10) -> str:
    query_type = detect_query_type(query)
    events = search_similar_events(query, top_k)

    stats = {}
    if query_type == "comparison":
        stats = get_comparison_stats(query)
    elif query_type == "aggregation":
        stats = get_aggregation_stats(query)
    else:
        qs = build_base_queryset(query)
        stats["filtered_count"] = qs.count()

    lines = []

    if stats:
        lines.append("=== Statistics ===")
        for key, value in stats.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            elif isinstance(value, list):
                lines.append(f"{key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("")

    if events:
        lines.append("=== Matching Events ===")
        for i, e in enumerate(events, 1):
            lines.append(
                f"{i}. [{e['timestamp']}] Camera {e['camera']} - {e['type']} "
                f"(weather: {e['weather']}, confidence: {e['confidence']:.2f})"
            )
            if e.get("text"):
                lines.append(f"   Detail: {e['text']}")
    else:
        lines.append("No matching events found.")

    return "\n".join(lines)