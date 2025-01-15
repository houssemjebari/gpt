import pytest 
from   training.subject import Subject
from   training.observer import Observer

class MockObserver(Observer):
    def __init__(self):
        self.events = []
    
    def update(self, event_type, data):
        self.events.append((event_type, data))
    
class MockSubject(Subject):
    def attach(self, observer):
        super().attach(observer)
    
    def detach(self, observer):
        super().detach(observer)
    
    def notify(self, event_type, data):
        super().notify(event_type, data)
    
def test_observer_receive_notification():
    '''
    Test Objective:
    - Ensure that observers can attach and 
    detach correctly.
    '''
    subject = MockSubject()
    observer = MockObserver()

    # Initialize no observers
    assert len(subject.observers) == 0

    # Attach observer 
    subject.attach(observer)
    assert len(subject.observers) == 1

    # Detach observer 
    subject.detach(observer)
    assert len(subject.observers) == 0

def test_observer_receive_notification():
    '''
    Test Objective:
    - Ensure that observers can receive
    notifications correctly.
    '''
    subject = MockSubject()
    observer = MockObserver()
    subject.attach(observer)

    event_type = 'test_event'
    data = {'key' : 'value'}
    subject.notify(event_type, data)
    assert len(observer.events) == 1
    assert observer.events[0] == (event_type, data)
