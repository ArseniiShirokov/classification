class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.cur = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.cur = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.cur = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
