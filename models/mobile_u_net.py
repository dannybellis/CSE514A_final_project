import tensorflow.keras.layers as layers
import tensorflow as tf

class depthwiseConv(layers.Layer):
  def __init__(self, filters, kernel_size=[3,3], name='depthwiseConv'):
    super(depthwiseConv, self).__init__(name=name)
    self.depthwise1 = layers.DepthwiseConv2D(kernel_size, padding='same')
    self.bitwise1 = layers.Conv2D(filters, kernel_size=[1,1], padding='same')
    self.batchnorm1 = layers.BatchNormalization()
    self.relu1 = layers.ReLU()
    self.batchnorm2 = layers.BatchNormalization()
    self.relu2 = layers.ReLU()

    self.depthwise2 = layers.DepthwiseConv2D(kernel_size, padding='same')
    self.bitwise2 = layers.Conv2D(filters, kernel_size=[1,1], padding='same')
    self.batchnorm3 = layers.BatchNormalization()
    self.relu3 = layers.ReLU()
    self.batchnorm4 = layers.BatchNormalization()
    self.relu4 = layers.ReLU()

    self.pool = layers.MaxPool2D(pool_size=(2,2))
        
  def call(self, inputs):
    with tf.device('/device:GPU:0'):
      net = self.depthwise1(inputs)
      net = self.batchnorm1(net)
      net = self.relu1(net)
      net = self.bitwise1(net)
      net = self.batchnorm2(net)
      net = self.relu2(net)

      net = self.depthwise2(net)
      net = self.batchnorm3(net)
      net = self.relu4(net)
      net = self.bitwise2(net)
      net = self.batchnorm4(net)
      net = self.relu4(net)

      pooled = self.pool(net)
      return pooled, net
    
class upConv(layers.Layer):
  def __init__(self, filters, kernel_size=[3,3], name='upConv'):
    super(upConv, self).__init__(name=name)
    self.tconv = layers.Conv2DTranspose(filters, kernel_size, strides=(2,2), padding='same')
    self.batchnorm = layers.BatchNormalization()
    self.relu = layers.ReLU()
    self.dw1 = depthwiseConv(filters)
    self.dw2 = depthwiseConv(filters)
        
  def call(self, inputs, concat):
    with tf.device('/device:GPU:0'):
      net = self.tconv(inputs)
      net = layers.concatenate([net, concat], axis=-1)
      net = self.batchnorm(net)
      net = self.relu(net)
      _, net = self.dw1(net)
      _, net = self.dw2(net)
      return net
        
class mobileUNet(tf.keras.Model):
  def __init__(self, filters=32, classes=21):
    super(mobileUNet, self).__init__()
    self.down1 = depthwiseConv(filters, name='down1')
    self.down2 = depthwiseConv(filters*2, name='down2')
    self.down3 = depthwiseConv(filters*4, name='down3')
    self.down4 = depthwiseConv(filters*8, name='down4')
    self.down5 = depthwiseConv(filters*16, name='down5')
    self.up1 = upConv(filters*8, name='up1')
    self.up2 = upConv(filters*4, name='up2')
    self.up3 = upConv(filters*2, name='up3')
    self.up4 = upConv(filters, name='up4')
    self.out = layers.Conv2D(classes, kernel_size=[1,1], padding='same', name='out')
        
  def call(self, inputs):
    with tf.device('/device:GPU:0'):
      layer1, concat1 = self.down1(inputs)
      layer2, concat2 = self.down2(layer1)
      layer3, concat3 = self.down3(layer2)
      layer4, concat4 = self.down4(layer3)
      _, layer5 = self.down5(layer4)
      layer6 = self.up1(layer5, concat4)
      layer7 = self.up2(layer6, concat3)
      layer8 = self.up3(layer7, concat2)
      layer9 = self.up4(layer8, concat1)
      layer10 = self.out(layer9)
      return layer10
