from tensorflow.keras.optimizers.schedules import PolynomialDecay, ExponentialDecay

def getScheduler(sched, lr=1e-3):
    if sched == 'poly':
        return PolynomialDecay(lr, 3000, 0.99)
    elif sched == 'exp': 
        return ExponentialDecay(lr, 1000)
    else:
        return lr
        