from observer import Observer

class Logger(Observer):
    def __init__(self, log_file, master):
        self.log_file = log_file
        self.master = master

        with open(self.log_file, 'w') as f:
            pass 

    def update(self, event_type, data):
        if (event_type == 'on_step_end') and self.master:
            with open(self.log_file, 'a') as f:
                f.write(f"Step {data['step']}\t| Loss: {data['loss']}\t | lr: {data['lr']}\n")

