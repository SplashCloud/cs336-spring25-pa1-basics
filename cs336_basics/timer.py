import time

class Timer:
    """Context manager for timing code blocks"""
    def __init__(self):
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        return False