import pytest 
import os
import re

from training.logger import Logger
from training.observer import Observer

def normalize_whitespace(s):
    return re.sub(r'\s+', ' ', s.strip())


class MockObserver(Observer):
    def __init__(self):
        self.events = []
    
    def update(self, event_type, data):
        self.events.append((event_type, data))
    
def test_logger_writes_correctly(tmp_path):
    '''
    Test Objective:
    - Ensure that the Logger correctly writes logs 
    when notified
    '''
    log_file = tmp_path / "test_log.txt"
    logger = Logger(log_file=str(log_file), master=True)
    # Create a mock event
    event_type = "on_step_end"
    data = {
        "step": 1,
        "model": None,  # Model is not used by Logger in this test
        "loss": 0.123456,
        "lr": 0.001234
    }
    # Invoke update
    logger.update(event_type, data)
    # Read the log file and verify contents
    with open(log_file, 'r') as f:
        lines = f.readlines()
    # Perform the assertions
    assert len(lines) == 1
    expected = "Step 1\t | Loss: 0.123456\t | lr: 0.001234"
    assert normalize_whitespace(lines[0].strip()) == normalize_whitespace(expected), "Strings do not match!"

