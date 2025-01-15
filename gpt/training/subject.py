from abc import ABC, abstractmethod

class Subject(ABC):
    def __init__(self):
        self.observers = [] 

    def attach(self, observer):
        self.observers.append(observer)

    def detach(self, observer):
        self.observers.remove(observer)
    
    def notify(self, event_type, data):
        for observer in self.observers:
            observer.update(event_type, data)
    