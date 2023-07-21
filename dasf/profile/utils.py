from typing import List
from dasf.profile.profiler import EventDatabase


from typing import List

class MultiEventDatabase:
    def __init__(self, databases: List[EventDatabase]):
        self._databases = databases

    def __iter__(self):
        for database in self._databases:
            yield from database.get_traces()