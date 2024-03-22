"""123
"""

from datetime import datetime
import pytz

# about time

def time_now():
    """123
    """
    # 取得當下的 UTC 時間
    # UTC_NOW = datetime.utcnow() # 舊的
    utc_now = datetime.now()

    # 設定 UTC+8 的時區
    utc_8 = pytz.timezone('Asia/Taipei')

    # 將 UTC 時間轉換為 UTC+8 時區的時間
    now = utc_now.replace(tzinfo=pytz.utc).astimezone(utc_8)

    return now.strftime("%Y_%m%d_%H%M")
