from VGG13_Utils import *
from VGG13_Model import *
from VGG13_Training import *

if __name__=='__main__':

    content_layers = ['block5_conv2'] 

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
            'block5_conv1'
                ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    content_path = '1200px-Green_Sea_Turtle_grazing_seagrass.jpg'
    style_path = 'Vassily_Kandinsky.jpg'
    compute_and_visualize_transfer(content_path = content_path, style_path = style_path, style_layers =style_layers, content_layers =content_layers)