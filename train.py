from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model, Model
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pathlib import Path
import os
from tensorflow.keras.layers import Conv2D, Add, UpSampling2D,Layer,Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dot,Activation
from tensorflow.keras.layers import Lambda
import cv2

from tensorflow.keras.layers import Layer, Conv2D, Reshape, Multiply, Concatenate




import glob

jpg_files = glob.glob(f"{os.getcwd()+"/SynthText/1"}/*.jpg")
print(f"{os.getcwd()+"/Implementation-of-OCR-system-on-extracting-information-from-Vietnamese-book-cover-images/SynthText/1"}/*.jpg")
print("IMAGE LENGTH : ",jpg_files)




resnet_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(512, 512, 3))


print("resnet_model LOADED")


def Resnet50_updated(inputs):
    feature_extractor = Model(inputs=resnet_model.input, 
                              outputs=[resnet_model.get_layer('conv2_block3_out').output,
                                       resnet_model.get_layer('conv4_block6_out').output,
                                       resnet_model.get_layer('conv5_block3_out').output])

    # Get features for c2, c4, and c5 from the input image
    c2, c4, c5 = feature_extractor(inputs)

    # Upsample and project
    c4_upsampled = UpSampling2D(size=(4, 4), interpolation='bilinear')(c4)
    c5_upsampled = UpSampling2D(size=(8, 8), interpolation='bilinear')(c5)

    conv1 = Conv2D(128, (1, 1), padding='same')(c2)
    conv2 = Conv2D(128, (1, 1), padding='same')(c4_upsampled)
    conv3 = Conv2D(128, (1, 1), padding='same')(c5_upsampled)

    # c2_upsampled = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv1)
    # c4_upsampled = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv2)
    # c5_upsampled = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv3)

    fused_feature_map = Add()([conv1, conv2, conv3])
    return fused_feature_map

        
        
    

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Reshape, Multiply, Concatenate

class ContextAttentionBlock(Layer):
    def __init__(self):
        super(ContextAttentionBlock, self).__init__()
        # Define Conv2D layers but do not initialize with specific input shapes
        self.conv_theta = Conv2D(128, (1, 1), padding='same')
        self.conv_phi = Conv2D(128, (1, 1), padding='same')
        self.conv_g = Conv2D(128, (1, 1), padding='same')
        self.conv_shortcut = Conv2D(128, (1, 1), padding='same')
        self.conv_output = Conv2D(128, (1, 1), padding='same')

    def build(self, input_shape):
        # Initialize layers based on the input shape; Keras will call this automatically
        self.conv_theta.build(input_shape)
        self.conv_phi.build(input_shape)
        self.conv_g.build(input_shape)
        self.conv_shortcut.build(input_shape)
        # self.conv_output.build(input_shape)
        super(ContextAttentionBlock, self).build(input_shape)

    def call(self, inputs):
        fused_feature_map = inputs

        # Apply 1x1 Conv layers
        f_theta = self.conv_theta(fused_feature_map)
        f_phi = self.conv_phi(fused_feature_map)
        f_g = self.conv_g(fused_feature_map)

        # Reshape for attention calculation
        B, H, W, C = tf.shape(f_theta)[0], tf.shape(f_theta)[1], tf.shape(f_theta)[2], tf.shape(f_theta)[3]
        f_theta_reshaped = tf.reshape(f_theta, [B* H , W, C])
        f_phi_reshaped = tf.reshape(f_phi, [B* H , W, C])
        f_g_reshaped = tf.reshape(f_g, [B* H , W, C])

        # Calculate horizontal attention
        horizontal_attention = tf.keras.activations.sigmoid(tf.matmul(f_phi_reshaped, f_theta_reshaped, transpose_b=True))
        horizontal_attention = tf.reshape(horizontal_attention, [B, H, W, C])
        horizontal_result = Multiply()([horizontal_attention, fused_feature_map])

        # Calculate vertical attention
        # f_theta_reshaped = tf.reshape(f_theta, [B* W , H, C])
        # f_phi_reshaped = tf.reshape(f_phi, [B* W , H, C])
        # f_g_reshaped = tf.reshape(f_g, [B* W , H, C])

        vertical_attention = tf.keras.activations.sigmoid(tf.matmul(f_g_reshaped, f_phi_reshaped, transpose_a=True))
        vertical_attention = tf.reshape(vertical_attention, [B, H, W, C])
        vertical_result = Multiply()([vertical_attention, fused_feature_map])

        # Shortcut connection
        short_cut = self.conv_shortcut(fused_feature_map)

        # Adjust shapes before concatenation if needed
        if horizontal_result.shape[-1] != short_cut.shape[-1] or vertical_result.shape[-1] != short_cut.shape[-1]:
            raise ValueError("Incompatible shapes for concatenation in ContextAttentionBlock.")

        # Concatenate results
        cab_output = Concatenate()([horizontal_result, short_cut, vertical_result])
        cab_output = self.conv_output(cab_output)
        return cab_output

# Example usage in model
inputs = tf.keras.Input(shape=(512, 512, 3))
feature_map = Resnet50_updated(inputs)  # Assume this is your ResNet feature extractor

# Apply the custom CAB layer
cab_layer = ContextAttentionBlock()
attention_output = cab_layer(feature_map)
attention_output = cab_layer(attention_output)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(attention_output)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(attention_output)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
output = Conv2D(1, (1, 1), activation='relu')(x)

model = tf.keras.Model(inputs=inputs, outputs=output)




from tensorflow.keras.optimizers import Adam

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-5,
    decay_steps=1000,
    decay_rate=0.94)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule), 
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
       metrics=["accuracy"]
)




x_train = []
y_train = []
for epochs in range(50):
    for image_counter in range(len(jpg_files)):
        x_train = []
        y_train = []
        counter = image_counter - 1
        for image_path in jpg_files:
            counter += 1
            if (counter > (image_counter+1) * 500) or counter == len(jpg_files)-1 :
                break
            name = image_path
            try:
                image = cv2.imread(name)
                image = image /255
                tcl = cv2.imread(name[:-4]+"/tcl_map.jpg")
                tcl[tcl > 0] = 1
                # tco = cv2.imread(name[:-4]+"/tco_map.jpg")

                # tbo1 = cv2.imread(name[:-4]+"/tbo_map1.jpg")
                # tbo2 = cv2.imread(name[:-4]+"/tbo_map2.jpg")
                # tbo3 = cv2.imread(name[:-4]+"/tbo_map3.jpg")
                # tbo4 = cv2.imread(name[:-4]+"/tbo_map4.jpg")

                # tvo1 = cv2.imread(name[:-4]+"/tvo_map4.jpg")
                # tvo2 = cv2.imread(name[:-4]+"/tvo_map4.jpg")
                # tvo3 = cv2.imread(name[:-4]+"/tvo_map4.jpg")
                # tvo4 = cv2.imread(name[:-4]+"/tvo_map4.jpg")


                image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

                tcl_map_resized = cv2.resize(tcl, (128, 128), interpolation=cv2.INTER_LINEAR)

                # tco_map_resized = cv2.resize(tco, (128, 128), interpolation=cv2.INTER_LINEAR)


                # tbo1_map_resized = cv2.resize(tbo1, (128, 128), interpolation=cv2.INTER_LINEAR)
                # tbo2_map_resized = cv2.resize(tbo2, (128, 128), interpolation=cv2.INTER_LINEAR)
                # tbo3_map_resized = cv2.resize(tbo3, (128, 128), interpolation=cv2.INTER_LINEAR)
                # tbo4_map_resized = cv2.resize(tbo4, (128, 128), interpolation=cv2.INTER_LINEAR)



                # tvo1_map_resized = cv2.resize(tvo1, (128, 128), interpolation=cv2.INTER_LINEAR)
                # tvo2_map_resized = cv2.resize(tvo2, (128, 128), interpolation=cv2.INTER_LINEAR)
                # tvo3_map_resized = cv2.resize(tvo3, (128, 128), interpolation=cv2.INTER_LINEAR)
                # tvo4_map_resized = cv2.resize(tvo4, (128, 128), interpolation=cv2.INTER_LINEAR)

                tcl_map_resized = tcl_map_resized[:, :, 0]
                # tco_map_resized = tco_map_resized[:, :, 0]

                # tbo1_map_resized = tbo1_map_resized[:, :, 0]
                # tbo2_map_resized = tbo2_map_resized[:, :, 0]
                # tbo3_map_resized = tbo3_map_resized[:, :, 0]
                # tbo4_map_resized = tbo4_map_resized[:, :, 0]

                # tvo1_map_resized = tvo1_map_resized[:, :, 0]
                # tvo2_map_resized = tvo2_map_resized[:, :, 0]
                # tvo3_map_resized = tvo3_map_resized[:, :, 0]
                # tvo4_map_resized = tvo4_map_resized[:, :, 0]
                

                tcl_map_resized = np.expand_dims(tcl_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                # tco_map_resized = np.expand_dims(tco_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                # tbo1_map_resized = np.expand_dims(tbo1_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                # tbo2_map_resized = np.expand_dims(tbo2_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                # tbo3_map_resized = np.expand_dims(tbo3_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                # tbo4_map_resized = np.expand_dims(tbo4_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                # tvo1_map_resized = np.expand_dims(tvo1_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                # tvo2_map_resized = np.expand_dims(tvo2_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                # tvo3_map_resized = np.expand_dims(tvo3_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                # tvo4_map_resized = np.expand_dims(tvo4_map_resized, axis=-1)  # Shape becomes (1, 128, 128)
                
                # y_sample = np.concatenate([tcl_map_resized, tco_map_resized, tbo1_map_resized,tbo2_map_resized,tbo3_map_resized,tbo4_map_resized, tvo1_map_resized,tvo2_map_resized,tvo3_map_resized,tvo4_map_resized], axis=-1)  # Shape (128, 128, 10)

                # print(y_sample.shape)
                # Append y_sample to y_train list or array
                y_train.append(tcl_map_resized)  # If using a list
                x_train.append(image_resized)  # If using a list
                
            
            except:
                # print("Image index: ",i," Not Appended")
                continue

        y_train = np.stack(y_train)
        x_train = np.stack(x_train)

        model.fit(
        x=x_train,  # or a generator
        y=y_train,
        epochs=1,
        batch_size=16,
        )
