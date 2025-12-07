# -----------------------------------------------------------------------------
# Copyright (c) 2025
#
# Authors:
#   Liruo Wang
#       School of Electrical Engineering and Computer Science,
#       University of Ottawa
#       lwang032@uottawa.ca
#
#   Zhenyan Xing
#       School of Electrical Engineering and Computer Science,
#       University of Ottawa
#       zxing045@uottawa.ca
#
# All rights reserved.
# This file is totally written by Zhenyan Xing,modify by Liruo Wang.
# -----------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
from pgvector.django import CosineDistance
from events.models import Event, Camera
from django.db.models import Count, Q
from django.db.models.functions import TruncDate, TruncWeek, TruncMonth
from datetime import datetime, timedelta
from django.utils import timezone
import re

_embedder = None

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def get_embedder():
    global _embedder
    if _embedder is None:
        print("[DEBUG] Loading SentenceTransformer model...")
        _embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True)
    return _embedder


def parse_query_intent(query: str):
    query_lower = query.lower()

    intent = None
    if any(w in query_lower for w in ["most", "highest", "maximum", "more overall", "more accident"]):
        intent = "find_max"
    elif any(w in query_lower for w in ["least", "lowest", "minimum", "fewest"]):
        intent = "find_min"
    elif any(w in query_lower for w in ["no ", "without", "zero ", " any ", "free"]):
        intent = "find_absence"
    elif any(w in query_lower for w in ["how many", "count", "total"]):
        intent = "count"
    elif any(w in query_lower for w in ["compare", "versus", "vs ", "more than", "less than"]):
        intent = "compare"
    # Added explicit intent for recency to help logic downstream if needed
    elif any(w in query_lower for w in ["latest", "newest", "most recent", "current"]):
        intent = "find_latest"

    dimension = None
    if any(w in query_lower for w in ["camera", "cam "]):
        dimension = "camera"
    elif any(w in query_lower for w in [" day", "daily", "yesterday", "today", "date"]):
        dimension = "day"
    elif any(w in query_lower for w in ["week", "weekly"]):
        dimension = "week"
    elif any(w in query_lower for w in ["month", "monthly"]):
        dimension = "month"
    elif any(w in query_lower for w in ["weather", "rain", "clear", "sunny"]):
        dimension = "weather"
    elif any(w in query_lower for w in ["confidence"]):
        dimension = "confidence"

    print(f"[DEBUG] parse_query_intent: Intent='{intent}', Dimension='{dimension}'")
    return {"intent": intent, "dimension": dimension}


def handle_generic_which_query(query: str, intent_info: dict):
    # print(f"[DEBUG] handle_generic_which_query called with intent_info: {intent_info}")
    intent = intent_info.get("intent")
    dimension = intent_info.get("dimension")

    if not intent or not dimension:
        return None

    stats = {}
    start_time, end_time, _, _ = parse_time_filter(query)
    qs = build_base_queryset(query)

    if intent == "find_absence":
        if dimension == "day":
            if start_time and end_time:
                days_with_events = set(
                    qs.annotate(date=TruncDate("timestamp")).values_list("date", flat=True)
                )
                all_days = set()
                current = start_time.date()
                while current <= end_time.date():
                    all_days.add(current)
                    current += timedelta(days=1)
                days_without = all_days - days_with_events
                stats["days_without_events"] = sorted(days_without)
            else:
                today = timezone.now().date()
                has_event = qs.filter(timestamp__date=today).exists()
                stats["today_has_events"] = has_event

        elif dimension == "week":
            if start_time and end_time:
                weeks_with_events = set(
                    qs.annotate(week=TruncWeek("timestamp")).values_list("week", flat=True)
                )
                stats["weeks_with_events"] = sorted(weeks_with_events)
            else:
                this_week_start = timezone.now() - timedelta(days=timezone.now().weekday())
                this_week_start = this_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
                has_event = qs.filter(timestamp__gte=this_week_start).exists()
                stats["this_week_has_events"] = has_event

        elif dimension == "month":
            if start_time and end_time:
                months_with_events = set(
                    qs.annotate(month=TruncMonth("timestamp")).values_list("month", flat=True)
                )
                stats["months_with_events"] = sorted(months_with_events)
            else:
                this_month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                has_event = qs.filter(timestamp__gte=this_month_start).exists()
                stats["this_month_has_events"] = has_event

        elif dimension == "camera":
            all_cameras = set(Camera.objects.values_list("camera_id", flat=True))
            cameras_with_events = set(qs.values_list("camera__camera_id", flat=True).distinct())
            cameras_without = all_cameras - cameras_with_events
            stats["cameras_without_events"] = list(cameras_without)
            stats["cameras_with_events"] = list(cameras_with_events)

    elif intent == "find_max":
        if dimension == "camera":
            camera_counts = qs.values("camera__camera_id").annotate(
                count=Count("event_id")
            ).order_by("-count")
            stats["camera_ranking"] = list(camera_counts)
            if camera_counts:
                top = camera_counts.first()
                stats["top_camera"] = top["camera__camera_id"]
                stats["top_camera_count"] = top["count"]

        elif dimension == "day":
            day_counts = qs.annotate(
                date=TruncDate("timestamp")
            ).values("date").annotate(count=Count("event_id")).order_by("-count")
            if day_counts:
                top = day_counts.first()
                stats["busiest_day"] = top["date"]
                stats["busiest_day_count"] = top["count"]

        elif dimension == "weather":
            weather_counts = qs.exclude(weather__isnull=True).exclude(weather='').values(
                "weather"
            ).annotate(count=Count("event_id")).order_by("-count")
            stats["weather_ranking"] = list(weather_counts)

    elif intent == "find_min":
        if dimension == "camera":
            camera_counts = qs.values("camera__camera_id").annotate(
                count=Count("event_id")
            ).order_by("count")
            stats["camera_ranking"] = list(camera_counts)
            if camera_counts:
                bottom = camera_counts.first()
                stats["least_camera"] = bottom["camera__camera_id"]
                stats["least_camera_count"] = bottom["count"]

        elif dimension == "day":
            day_counts = qs.annotate(
                date=TruncDate("timestamp")
            ).values("date").annotate(count=Count("event_id")).order_by("count")
            if day_counts:
                bottom = day_counts.first()
                stats["quietest_day"] = bottom["date"]
                stats["quietest_day_count"] = bottom["count"]

        elif dimension == "confidence":
            least_conf = qs.order_by("confidence").first()
            if least_conf:
                stats["least_confidence_event"] = format_event(least_conf)

    return stats if stats else None


def parse_specific_date(query: str):
    query_lower = query.lower()

    # Format: 2025-12-01 or on 2025-12-01
    match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", query_lower)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    # Format: 12/01/2025 or 12-01-2025 (MM/DD/YYYY)
    match = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", query_lower)
    if match:
        month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    # Format: December 1, 2025 or Dec 1 2025 or December 1st, 2025
    month_pattern = "|".join(MONTH_MAP.keys())
    match = re.search(rf"({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})", query_lower)
    if match:
        month = MONTH_MAP[match.group(1)]
        day = int(match.group(2))
        year = int(match.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    # Format: 1 December 2025 or 1st Dec 2025
    match = re.search(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern}),?\s*(\d{{4}})", query_lower)
    if match:
        day = int(match.group(1))
        month = MONTH_MAP[match.group(2)]
        year = int(match.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    return None


def parse_date_range(query: str):
    query_lower = query.lower()

    # Format: from 2025-12-01 to 2025-12-05
    match = re.search(r"from\s+(\d{4}-\d{1,2}-\d{1,2})\s+to\s+(\d{4}-\d{1,2}-\d{1,2})", query_lower)
    if match:
        try:
            start = datetime.strptime(match.group(1), "%Y-%m-%d")
            end = datetime.strptime(match.group(2), "%Y-%m-%d")
            return start, end
        except ValueError:
            pass

    # Format: between 2025-12-01 and 2025-12-05 (date range, not time range)
    match = re.search(r"between\s+(\d{4}-\d{1,2}-\d{1,2})\s+and\s+(\d{4}-\d{1,2}-\d{1,2})", query_lower)
    if match:
        try:
            start = datetime.strptime(match.group(1), "%Y-%m-%d")
            end = datetime.strptime(match.group(2), "%Y-%m-%d")
            return start, end
        except ValueError:
            pass

    # Format: before 2025-12-01
    match = re.search(r"before\s+(\d{4}-\d{1,2}-\d{1,2})", query_lower)
    if match:
        try:
            end = datetime.strptime(match.group(1), "%Y-%m-%d")
            return None, end
        except ValueError:
            pass

    # Format: after 2025-12-01
    match = re.search(r"after\s+(\d{4}-\d{1,2}-\d{1,2})", query_lower)
    if match:
        try:
            start = datetime.strptime(match.group(1), "%Y-%m-%d")
            return start, None
        except ValueError:
            pass

    # Format: in December 2025 or in Dec 2025
    month_pattern = "|".join(MONTH_MAP.keys())
    match = re.search(rf"in\s+({month_pattern})\s+(\d{{4}})", query_lower)
    if match:
        month = MONTH_MAP[match.group(1)]
        year = int(match.group(2))
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(year, month + 1, 1) - timedelta(seconds=1)
        return start, end

    return None, None


def parse_time_filter(query: str):
    query_lower = query.lower()
    now = timezone.now()
    start_time = None
    end_time = None

    # First check for specific date
    specific_date = parse_specific_date(query)
    if specific_date:
        specific_date = timezone.make_aware(specific_date)
        start_time = specific_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = specific_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        # Check for date range
        range_start, range_end = parse_date_range(query)
        if range_start or range_end:
            if range_start:
                range_start = timezone.make_aware(range_start)
                start_time = range_start.replace(hour=0, minute=0, second=0, microsecond=0)
            if range_end:
                range_end = timezone.make_aware(range_end)
                end_time = range_end.replace(hour=23, minute=59, second=59, microsecond=999999)
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
            start_time = (first_of_this_month - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0,
                                                                           microsecond=0)
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

    # Parse time range (hour:minute) - applies on top of date filter
    time_range_match = re.search(r"between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})", query_lower)
    hour_start, hour_end = None, None
    if time_range_match:
        hour_start = datetime.strptime(time_range_match.group(1), "%H:%M").time()
        hour_end = datetime.strptime(time_range_match.group(2), "%H:%M").time()

    # print(f"[DEBUG] parse_time_filter: Start={start_time}, End={end_time}, H_Start={hour_start}, H_End={hour_end}")
    return start_time, end_time, hour_start, hour_end


def parse_camera_filter(query: str):
    query_lower = query.lower()

    # 1. 优先匹配明确的格式: "camera 1", "cam 2"
    cameras = re.findall(r"(?:camera|cam)\s*(\d+)", query_lower)
    if cameras:
        # ... (保留你原有的处理逻辑，检查DB是否存在)
        result = []
        for cam_num in cameras:
            # Check DB logic...
            result.append(cam_num)
        return result if len(result) > 1 else (result[0] if result else None)

    # 2. Fallback: 只有当数字作为“独立单词”出现时才认为是Camera ID
    all_cam_ids = Camera.objects.values_list("camera_id", flat=True)
    for cam in all_cam_ids:
        cam_str = str(cam).lower()
        if re.search(rf"\b{re.escape(cam_str)}\b", query_lower):
            return cam

    return None


def parse_weather_filter(query: str):
    query_lower = query.lower()

    # Don't filter by weather if it's in a date context
    if re.search(
            r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)",
            query_lower):
        if "clear" in query_lower:
            if not re.search(r"clear\s+(weather|condition|day|sky)", query_lower):
                if re.search(r"\d+\s*,?\s*\d{4}", query_lower):
                    return None

    weathers = []
    if "rain" in query_lower:
        weathers.append("rainy")
    if re.search(r"clear\s*(weather|condition|day|sky)?", query_lower) and "unclear" not in query_lower:
        if "clear weather" in query_lower or "clear condition" in query_lower or "during clear" in query_lower:
            weathers.append("clear")
        elif "clear" in query_lower and not re.search(
                r"(january|february|march|april|may|june|july|august|september|october|november|december)",
                query_lower):
            weathers.append("clear")
    if "sunny" in query_lower:
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

    intent_info = parse_query_intent(query)
    if intent_info["intent"] == "compare":
        print(f"[DEBUG] detect_query_type: Detected 'comparison' via parse_query_intent")
        return "comparison"
    if intent_info["intent"] in ["find_max", "find_min", "find_absence", "count", "find_latest"]:
        print(f"[DEBUG] detect_query_type: Detected 'aggregation' via parse_query_intent")
        return "aggregation"

    comparison_keywords = ["compare", "comparison", "versus", "vs", "more than", "less than",
                           "how many more", "how much", "decrease", "increase", "compared"]

    # ADDED: recency keywords to aggregation_keywords to ensure they trigger get_aggregation_stats
    aggregation_keywords = [
        "how many", "count", "total", "overall", "which camera has",
        "which days", "earliest", "most recent", "latest", "last", "newest", "current",
        "types of", "least confidence", "which", "no accident", "without"
    ]

    for kw in comparison_keywords:
        if kw in query_lower:
            print(f"[DEBUG] detect_query_type: Detected 'comparison' keyword '{kw}'")
            return "comparison"
    for kw in aggregation_keywords:
        if kw in query_lower:
            print(f"[DEBUG] detect_query_type: Detected 'aggregation' keyword '{kw}'")
            return "aggregation"
    if any(kw in query_lower for kw in ["list", "show", "filter"]):
        print(f"[DEBUG] detect_query_type: Detected 'filtered' query")
        return "filtered"

    print(f"[DEBUG] detect_query_type: Defaulting to 'factual'")
    return "factual"


def build_base_queryset(query: str):
    print(f"[DEBUG] build_base_queryset: Starting to build QS for: '{query}'")
    qs = Event.objects.all()

    cam_ids = parse_camera_filter(query)
    if cam_ids:
        print(f"    [DEBUG] Filter: Camera IDs = {cam_ids}")
        if isinstance(cam_ids, list):
            qs = qs.filter(camera__camera_id__in=cam_ids)
        else:
            qs = qs.filter(camera__camera_id=cam_ids)

    start_time, end_time, hour_start, hour_end = parse_time_filter(query)
    if start_time:
        print(f"    [DEBUG] Filter: Start Time = {start_time}")
        qs = qs.filter(timestamp__gte=start_time)
    if end_time:
        print(f"    [DEBUG] Filter: End Time = {end_time}")
        qs = qs.filter(timestamp__lte=end_time)
    if hour_start and hour_end:
        print(f"    [DEBUG] Filter: Time Range = {hour_start} to {hour_end}")
        qs = qs.filter(timestamp__time__gte=hour_start, timestamp__time__lte=hour_end)

    weathers = parse_weather_filter(query)
    if weathers:
        print(f"    [DEBUG] Filter: Weather = {weathers}")
        weather_q = Q()
        for w in weathers:
            weather_q |= Q(weather__icontains=w)
        qs = qs.filter(weather_q)

    event_type = parse_event_type(query)
    if event_type:
        print(f"    [DEBUG] Filter: Event Type = {event_type}")
        qs = qs.filter(type__iexact=event_type)

    conf_filter = parse_confidence_filter(query)
    if conf_filter:
        op, val = conf_filter
        print(f"    [DEBUG] Filter: Confidence {op} {val}")
        if op == "lt":
            qs = qs.filter(confidence__lt=val)
        else:
            qs = qs.filter(confidence__gt=val)

    return qs


def get_aggregation_stats(query: str):
    print("[DEBUG] get_aggregation_stats: calculating...")
    query_lower = query.lower()
    stats = {}

    intent_info = parse_query_intent(query)
    generic_stats = handle_generic_which_query(query, intent_info)
    if generic_stats:
        stats.update(generic_stats)

    if "total" in query_lower or "how many" in query_lower:
        qs = build_base_queryset(query)
        stats["total_count"] = qs.count()

    if "earliest" in query_lower:
        qs = build_base_queryset(query)
        earliest = qs.order_by("timestamp").first()
        if earliest:
            stats["earliest_event"] = format_event(earliest)

    # === UPDATED BLOCK FOR RECENCY ===
    # Check for keywords: "most recent", "latest", "newest", "current"
    # Also "last", but handle carefully to avoid conflict with "last week", "last month"

    recency_triggers = ["most recent", "latest", "newest", "current"]

    # "Last" refers to a specific event if it is NOT followed by time units
    is_asking_last_item = False
    if "last" in query_lower:
        # If user says "last week", "last 2 days", etc, they are likely asking for a range, not the single last event.
        time_units = ["week", "month", "year", "day", "hour", "minute", "second", "24 hours"]
        if not any(unit in query_lower for unit in time_units):
            is_asking_last_item = True

    if any(kw in query_lower for kw in recency_triggers) or is_asking_last_item:
        qs = build_base_queryset(query)
        latest = qs.order_by("-timestamp").first()
        if latest:
            # We use the key 'most_recent_event' so the LLM context builder recognizes it
            stats["most_recent_event"] = format_event(latest)
    # =================================

    if "least confidence" in query_lower and "least_confidence_event" not in stats:
        qs = build_base_queryset(query)
        least_conf = qs.order_by("confidence").first()
        if least_conf:
            stats["least_confidence_event"] = format_event(least_conf)

    if "which camera" in query_lower and "more" in query_lower and "camera_ranking" not in stats:
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
    elif "types" in query_lower and "weather_ranking" not in stats:
        event_types = Event.objects.values_list("type", flat=True).distinct()
        stats["event_types"] = list(event_types)

    if ("which days" in query_lower or "which day" in query_lower) and "no accident" in query_lower:
        if "days_without_events" not in stats:
            start_time, end_time, _, _ = parse_time_filter(query)
            if start_time and end_time:
                qs = build_base_queryset(query)
                days_with_events = set(
                    qs.annotate(date=TruncDate("timestamp")).values_list("date", flat=True)
                )
                all_days = set()
                current = start_time.date()
                while current <= end_time.date():
                    all_days.add(current)
                    current += timedelta(days=1)
                days_without = all_days - days_with_events
                stats["days_without_accidents"] = sorted(days_without)

    print(f"[DEBUG] get_aggregation_stats result: {stats.keys()}")
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


def search_similar_events(query: str, top_k: int = 20):
    print(f"[DEBUG] search_similar_events: Embedding query '{query}'")
    model = get_embedder()
    q_vec = model.encode(query, normalize_embeddings=True).tolist()

    qs = build_base_queryset(query)

    d = CosineDistance("embedding", q_vec)
    qs = qs.annotate(dist=d).filter(dist__lt=0.7).order_by("dist")[:top_k]
    # qs = qs.annotate(dist=d).order_by("dist")[:top_k]

    results = []
    print(f"[DEBUG] search_similar_events: Executing Vector Search...")
    for e in qs:
        event_data = format_event(e)
        event_data["similarity"] = float(1 - e.dist)
        results.append(event_data)

    print(f"[DEBUG] search_similar_events: Found {len(results)} matching events.")
    return results


def build_context_for_llm(query: str, top_k: int = 20) -> str:
    print(f"\n{'=' * 20}\n[DEBUG] START PROCESSING QUERY: {query}\n{'=' * 20}")

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
        print(f"[DEBUG] Factual query - Filtered count: {stats['filtered_count']}")

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

    print(f"[DEBUG] FINISHED PROCESSING. Context length: {len(lines)} lines.\n")
    return "\n".join(lines)