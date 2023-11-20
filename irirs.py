import numpy as np
import tensorflow as tf

# Load TFLite model
class IrisDetector():
    def __init__(self,):
        model_path = 'iris_landmark.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.input_index = self.input_details[0]['index']
        self.output_details = self.interpreter.get_output_details()
        self.output_index = self.output_details[1]['index']
        print(self.output_details)

    def __call__(self,image):
    # Get input and output details
        # Prepare input data (replace with your own data)
        image = image/255.0
        input_data = np.expand_dims(image.astype(np.float32),axis=0)

        # Set input tensor
        self.interpreter.set_tensor(self.input_index, input_data)

        # Run inference
        self.interpreter.invoke()

        # Get the output tensor
        output_data = self.interpreter.get_tensor(self.output_index)
        output_data = output_data.squeeze()
        out = []
        for j in range(0,len(output_data),3):
            # if self.sigmoid(output_data[j+2]) < 0.3:
            out.append((int(output_data[j]),int(output_data[j+1])))
        if len(out) != 5:
            return False,out
        h = out[4][1] - out[2][1] 
        w = out[1][0] - out[3][0] 

        if h < 7 or w < 7:
            return False,out
        return True,out
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))