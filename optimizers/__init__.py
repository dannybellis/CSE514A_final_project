from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def getOptimizer(opt, lr=1e-3):
    if opt == 1:
        return Adam(learning_rate=lr)
    elif opt == 2:
        return RMSprop(learning_rate=lr)
    elif opt == 3:
        return SGD(learning_rate=lr)
    else:
        print('Optimizer not found. Using SGD instead')
        return SGD(learning_rate=lr)