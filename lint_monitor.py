#!/usr/bin/env python3
"""
Lint monitor with time-based improvement tracking
"""

import subprocess
import time
from datetime import datetime, timedelta
from collections import deque

# Configuration
LOG_FILE = "pylint_monitor.log"
INTERVAL = 60  # seconds
TIME_WINDOWS = [
    ("5m", timedelta(minutes=5)),
    ("15m", timedelta(minutes=15)),
    ("1h", timedelta(hours=1)),
    ("4h", timedelta(hours=4)),
    ("16h", timedelta(hours=16)),
]

class LintMonitor:
    def __init__(self):
        self.history = deque()
        self.last_score = None
        
    def get_pylint_score(self):
        """Run pylint and extract the score"""
        try:
            result = subprocess.run(
                ["pylint", "evoprompt/**py"],
                capture_output=True,
                text=True,
                check=True
            )
            last_line = result.stdout.strip().split('\n')[-1]
            if "rated at" in last_line:
                return float(last_line.split("rated at ")[1].split("/")[0])
        except (subprocess.CalledProcessError, ValueError):
            return None
        return None

    def calculate_improvements(self):
        """Calculate improvements for each time window"""
        improvements = {}
        current_time = datetime.now()
        
        for window_name, window_delta in TIME_WINDOWS:
            window_start = current_time - window_delta
            window_scores = [
                score for timestamp, score in self.history
                if timestamp >= window_start
            ]
            
            if window_scores:
                first = window_scores[0]
                last = window_scores[-1]
                improvement = last - first
                improvements[window_name] = improvement
            else:
                improvements[window_name] = None
                
        return improvements

    def run(self):
        """Main monitoring loop"""
        print(f"Starting lint monitor. Logging to {LOG_FILE}")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                score = self.get_pylint_score()
                if score is not None:
                    timestamp = datetime.now()
                    self.history.append((timestamp, score))
                    
                    # Keep only last 16 hours of data
                    cutoff = timestamp - TIME_WINDOWS[-1][1]
                    while self.history and self.history[0][0] < cutoff:
                        self.history.popleft()
                    
                    # Calculate improvements
                    improvements = self.calculate_improvements()
                    
                    # Format output
                    log_entry = f"{timestamp.isoformat()} - Current: {score:.2f}/10"
                    for window, improvement in improvements.items():
                        if improvement is not None:
                            log_entry += f", {window}: {improvement:+.2f}"
                    
                    # Log to file and print
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(log_entry + "\n")
                    print(log_entry)
                    
                time.sleep(INTERVAL)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor = LintMonitor()
    monitor.run()
