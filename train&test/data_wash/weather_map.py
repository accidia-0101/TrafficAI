# weather_map.py

# —— 最终训练类别（只有两个） ——
FINAL_WEATHER = ["clear", "rain"]

# —— 原始标签到训练标签的映射 ——
WEATHER_MAP = {
    # clear 类
    "clear": "clear",
    "overcast": "clear",
    "cloudy": "clear",
    "partly cloudy": "clear",

    # rain 类
    "rain": "rain",
    "rainy": "rain",

    # 其他全部不要（fog / mist / haze / smog / snowy / undefined）
    "fog": None,
    "foggy": None,
    "mist": None,
    "haze": None,
    "smog": None,
    "snowy": None,
    "undefined": None,
    None: None,
}

def map_weather(raw_weather):
    """把 BDD100K 原始天气字段映射到 2 类：clear / rain"""
    if raw_weather is None:
        return None
    raw_weather = raw_weather.lower().strip()
    return WEATHER_MAP.get(raw_weather, None)
