import datetime
import requests


class CrazyData:
    def __init__(self, pid, ticket):
        self.pid = pid
        self.ticket = ticket
        self.api = 'https://data.6-79.cn/v1/segment'

    def push(self, waves, time=None):
        try:
            crt_time = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
            time = time or crt_time.timestamp()

            data = dict(
                time=time,
                waves=waves,
                pid=self.pid,
                ticket=self.ticket,
            )

            requests.post(self.api, json=data).close()
        except Exception:
            pass
