import threading
import time
from . import log
import asyncio
import uuid

logger = log.get_logger(__name__)

lock = threading.Lock()
active_profilers = dict()  # thread_id -> stats
thread_stats = dict() # key_name
func_stats = dict() # key_name@func_name
uuid_map = dict() #thread_id --> uuid

class ThreadProfiler:

    def __init__(self, thread_name):
        thread_id = threading.get_ident()
        # thread_id = asyncio.current_task().get_name()
        self.func_profilers = dict()  # thread_id: thread's list of func stats
        self.thread_name = thread_name
        self.profiler = {
            "start_time": round(time.time() * 1000),
            "thread_id": thread_id,
        }
        self.func_profilers = dict()

    def start_thread(self, thread_name):
        thread_id = threading.get_ident()
        # thread_id = asyncio.current_task().get_name()
        self.func_profilers = dict()  # thread_id: thread's list of func stats
        self.thread_name = thread_name
        self.profiler = {
            "start_time": round(time.time() * 1000),
            "thread_id": thread_id,
        }
        self.func_profilers = dict()

    def close_thread(self):
        profiler = self.profiler
        profiler['end_time'] = round(time.time() * 1000)
        profiler['process_time'] = profiler['end_time'] - profiler['start_time']
        profiler['func_profilers'] = self.func_profilers
        profiler['thread_name'] = self.thread_name
        return profiler

    def push(self, func_name):
        if func_name not in self.func_profilers:
            self.func_profilers[func_name] = []
        self.func_profilers[func_name].append({
            "start_time": round(time.time() * 1000),
            "func_name": func_name,
        })

    def pop(self, func_name):
        if len(self.func_profilers[func_name]) <= 0:
            return
        profiler = self.func_profilers[func_name][-1]
        del self.func_profilers[func_name][-1]
        profiler['end_time'] = round(time.time() * 1000)
        profiler['process_time'] = profiler['end_time'] - profiler['start_time']
        return profiler
    
class ProfilerStats:

    def __init__(self, key):
        self.key = key
        self.total = 0
        self.n_pendding = 0
        self.total_process_time = 0
        self.last_process_time = 0
        self.proc_rate = 0
        self.req_rate = 0
        self.lock = threading.Lock()

    def append(self):
        with self.lock:
            self.total += 1
            self.n_pendding += 1

    def summary(self, stats):
        with self.lock:
            self.n_pendding -= 1
            self.total_process_time += stats['process_time']
            self.last_process_time = stats['process_time']
            self.proc_rate = round(self.total / self.total_process_time / 1000 * 100) / 100 if self.total_process_time != 0 else 0
            self.req_rate = 0

def append_request_id(req_id):
    try:
        lock.acquire()
        thread_id = threading.get_ident()
        uuid_map[thread_id] = req_id
        print(uuid_map)
    finally:
        lock.release()

def create_profiler_thread(thread_name, thread_id = None):
    try:
        lock.acquire()
        if thread_id is None:
            thread_id = threading.get_ident()
        # logger.info(f'create_profiler_thread {thread_id}')
        if thread_id not in active_profilers:
            active_profilers[thread_id] = ThreadProfiler(thread_name)
        if thread_name not in thread_stats:
            thread_stats[thread_name] = ProfilerStats(thread_name)
        thread_stats[thread_name].append()
        return active_profilers[thread_id]
    finally:
        lock.release()

def close_profiler_thread(thread_id = None):
    try:
        lock.acquire()
        if thread_id is None:
            thread_id = threading.get_ident()
        if thread_id not in active_profilers:
            return
        thread_profiler = active_profilers.pop(thread_id)
        stats = thread_profiler.close_thread()
        thread_stats[stats['thread_name']].summary(stats)
    finally:
        lock.release()


def push(func_name, thread_id = None):
    try:
        lock.acquire()
        if thread_id is None:
            thread_id = threading.get_ident()
            # print('push', thread_id, uuid_map)
            # thread_id = uuid_map[thread_id] if thread_id in uuid_map else thread_id
        if thread_id not in active_profilers:
            return
        profiler = active_profilers[thread_id]
        profiler.push(func_name)
        key = profiler.thread_name + "@" + func_name
        if key not in func_stats:
            func_stats[key] = ProfilerStats(key)
        func_stats[key].append()
    finally:
        lock.release()


def pop(func_name, thread_id = None):
    try:
        lock.acquire()
        if thread_id is None:
            thread_id = threading.get_ident()
            # print('pop', thread_id, uuid_map)
            # thread_id = uuid_map[thread_id] if thread_id in uuid_map else thread_id
        if thread_id not in active_profilers:
            return
        profiler = active_profilers[thread_id]
        stats = profiler.pop(func_name)
        key = profiler.thread_name + "@" + func_name
        func_stats[key].summary(stats)
    finally:
        lock.release()
