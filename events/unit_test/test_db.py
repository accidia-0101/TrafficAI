import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "DjangoTrafficAI.settings")  # ← 改成你的settings路径
django.setup()

from django.db import connection

try:
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database(), current_schema(), version();")
        row = cursor.fetchone()
        print("✅ 成功连接到数据库")
        print(f"当前数据库：{row[0]}")
        print(f"当前schema：{row[1]}")
        print(f"PostgreSQL版本：{row[2]}")
except Exception as e:
    print("❌ 连接失败：", e)
