from models.u_net import uNet
from models.mobile_u_net import mobileUNet


def getModel(name, filters=32, classes=21):
    if name=='mobile_u_net':
        return mobileUNet(filters, classes)
    elif name=='u_net':
        return uNet(filters, classes)
    else:
        print('Model not found')
        return None

