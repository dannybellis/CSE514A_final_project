import models
import optimizers
import schedulers
import utils

import tensorflow as tf
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split as ttsplit

def train(model_name, optimizer_name, scheduler_name, lr, img_path, mask_path, names_path, epochs=10):
    
    model = models.getModel(model_name)
    model.build((None, None, None, 3))
    print(model.summary())
    
    scheduler = schedulers.getScheduler(scheduler_name, lr)
    
    optimizer = optimizers.getOptimizer(optimizer_name, scheduler)

    cce = tf.keras.losses.CategoricalCrossentropy()
    
    train_loss_metric = tf.keras.metrics.Mean()
    train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
    
    test_loss_metric = tf.keras.metrics.Mean()
    test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
    
    file_list = open(names_path, 'r')
    names = file_list.read().splitlines()
    file_list.close()
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    trainset, testval = ttsplit(names, train_size=0.9)
    test, val = ttsplit(testval, train_size=0.5)
    
    
    trainset = names
    print(names)
    total_step = 0
    with tf.device('/device:GPU:0'):
        for epoch in range(epochs):
            for step_, batch in enumerate(trainset):
                total_step+=1
                print(total_step)
                img, mask = utils.genData(batch, mask_path, img_path)
                with tf.GradientTape() as tape:
                    mask_pred = model(img)
                    loss = cce(mask, mask_pred)
                    
                train_loss_metric.update_state(loss)
                train_accuracy_metric.update_state(mask, mask_pred)
                    
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                print(total_step)
                if step_%150 == 0:

                    with train_summary_writer.as_default():
                        tf.summary.scalar('Training Loss', train_loss_metric.result(), step=total_step)
                        tf.summary.scalar('Training Accuracy', train_accuracy_metric.result(), step=total_step)
                    
                    for step, batch in enumerate(val):
                        img_val, mask_val = utils.genData(batch, mask_path, img_path)
                        mask_pred_val = model(img_val)
                        loss_val = cce(mask_val, mask_pred_val)
                        print(loss_val)
                        
                        test_loss_metric.update_state(loss_val)
                        test_accuracy_metric.update_state(mask_val, mask_pred_val)
                        
                    with test_summary_writer.as_default():
                        tf.summary.scalar('Validation Loss', test_loss_metric.result(), step=total_step)
                        tf.summary.scalar('Validation Accuracy', test_accuracy_metric.result(), step=total_step)
                        
                    print('Epoch: ' + str(epoch)+ ' | Batch: ' + str(step) + ' | Training Loss: ' + str(train_loss_metric.result().numpy()) + ' | Training Accuracy: '+ str(train_accuracy_metric.result().numpy()))
                    print('Epoch: ' + str(epoch)+ ' | Batch: ' + str(step) + ' | Validation Loss: ' + str(test_loss_metric.result().numpy()) + ' | Validation Accuracy: '+ str(test_accuracy_metric.result().numpy()))
                    
                    train_loss_metric.reset_states()
                    train_accuracy_metric.reset_states()
                    
                    test_loss_metric.reset_states()
                    test_accuracy_metric.reset_states()
                        
                    
                    
                        
                
                
    
    
    