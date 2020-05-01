import tensorflow.keras.layers as layers
import tensorflow as tf

class uNet(tf.keras.Model):
  def __init__(self, filters=32, classes=21):
    super(uNet, self).__init__()

    self.down11 = layers.Conv2D(filters, kernel_size=[3,3], padding='same')
    self.relu11 = layers.ReLU()
    self.down12 = layers.Conv2D(filters, kernel_size=[3,3], padding='same')
    self.relu12 = layers.ReLU()
    self.pool1 = layers.MaxPool2D(pool_size=(2,2))

    self.down21 = layers.Conv2D(filters*2, kernel_size=[3,3], padding='same')
    self.relu21 = layers.ReLU()
    self.down22 = layers.Conv2D(filters*2, kernel_size=[3,3], padding='same')
    self.relu22 = layers.ReLU()
    self.pool2 = layers.MaxPool2D(pool_size=(2,2))

    self.down31 = layers.Conv2D(filters*4, kernel_size=[3,3], padding='same')
    self.relu31 = layers.ReLU()
    self.down32 = layers.Conv2D(filters*4, kernel_size=[3,3], padding='same')
    self.relu32 = layers.ReLU()
    self.pool3 = layers.MaxPool2D(pool_size=(2,2))

    self.down41 = layers.Conv2D(filters*8, kernel_size=[3,3], padding='same')
    self.relu41 = layers.ReLU()
    self.down42 = layers.Conv2D(filters*8, kernel_size=[3,3], padding='same')
    self.relu42 = layers.ReLU()
    self.pool4 = layers.MaxPool2D(pool_size=(2,2))

    self.down51 = layers.Conv2D(filters*16, kernel_size=[3,3], padding='same')
    self.relu51 = layers.ReLU()
    self.down52 = layers.Conv2D(filters*16, kernel_size=[3,3], padding='same')
    self.relu52 = layers.ReLU()
  
    self.upconv61 = layers.Conv2DTranspose(filters*8, kernel_size=[3,3], strides=(2,2), padding='same')
    self.upconv62 = layers.Conv2D(filters*8, kernel_size=[3,3], padding='same')
    self.relu61 = layers.ReLU()
    self.upconv63 = layers.Conv2D(filters*8, kernel_size=[3,3], padding='same')
    self.relu62 = layers.ReLU()

    self.upconv71 = layers.Conv2DTranspose(filters*4, kernel_size=[3,3], strides=(2,2), padding='same')
    self.upconv72 = layers.Conv2D(filters*4, kernel_size=[3,3], padding='same')
    self.relu71 = layers.ReLU()
    self.upconv73 = layers.Conv2D(filters*4, kernel_size=[3,3], padding='same')
    self.relu72 = layers.ReLU()

    self.upconv81 = layers.Conv2DTranspose(filters*2, kernel_size=[3,3], strides=(2,2), padding='same')
    self.upconv82 = layers.Conv2D(filters*2, kernel_size=[3,3], padding='same')
    self.relu81 = layers.ReLU()
    self.upconv83 = layers.Conv2D(filters*2, kernel_size=[3,3], padding='same')
    self.relu82 = layers.ReLU()

    self.upconv91 = layers.Conv2DTranspose(filters, kernel_size=[3,3], strides=(2,2), padding='same')
    self.upconv92 = layers.Conv2D(filters, kernel_size=[3,3], padding='same')
    self.relu91 = layers.ReLU()
    self.upconv93 = layers.Conv2D(filters, kernel_size=[3,3], padding='same')
    self.relu92 = layers.ReLU()

    self.out = layers.Conv2D(classes, kernel_size=[1,1], padding='same', name='out')
        
  def call(self, inputs, train=True):
    with tf.device('/device:GPU:0'):
      d11 = self.down11(inputs)
      r11 = self.relu11(d11)
      d12 = self.down12(r11)
      r12 = self.relu12(d12)
      p1 = self.pool1(r12)

      d21 = self.down21(p1)
      r21 = self.relu21(d21)
      d22 = self.down22(r21)
      r22 = self.relu22(d22)
      p2 = self.pool2(r22)

      d31 = self.down31(p2)
      r31 = self.relu31(d31)
      d32 = self.down32(r31)
      r32 = self.relu32(d32)
      p3 = self.pool3(r32)

      d41 = self.down41(p3)
      r41 = self.relu41(d41)
      d42 = self.down42(r41)
      r42 = self.relu42(d42)
      p4 = self.pool4(r42)

      d51 = self.down51(p4)
      r51 = self.relu51(d51)
      d52 = self.down52(r51)
      r52 = self.relu52(d52)

      u61 = self.upconv61(r52)
      c61 = layers.concatenate([u61, r42])
      u62 = self.upconv62(c61)
      r61 = self.relu61(u62)
      u63 = self.upconv63(r61)
      r62 = self.relu62(u63)
      
      u71 = self.upconv71(r62)
      c71 = layers.concatenate([u71, r32])
      u72 = self.upconv72(c71)
      r71 = self.relu71(u72)
      u73 = self.upconv73(r71)
      r72 = self.relu72(u73)

      u81 = self.upconv81(r72)
      c81 = layers.concatenate([u81, r22])
      u82 = self.upconv82(c81)
      r81 = self.relu81(u82)
      u83 = self.upconv83(r81)
      r82 = self.relu82(u83)

      u91 = self.upconv91(r82)
      c91 = layers.concatenate([u91, r12])
      u92 = self.upconv92(c91)
      r91 = self.relu91(u92)
      u93 = self.upconv93(r91)
      r92 = self.relu92(u93)

      out = self.out(r92)
      return out
