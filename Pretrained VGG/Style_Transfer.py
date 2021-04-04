from VGG16_Utils import *
from VGG19_Utils import *
from General_Utils import *

if __name__ == '__main__':

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
    compute_and_visualize_transfer(content_path = content_path, style_path = style_path, content_layers=content_layers, style_layers=style_layers)

    # here we simply swap content and style image to try to make the artistic image more realistic
    compute_and_visualize_transfer(content_path = 'The_Great_Wave_off_Kanagawa.jpg', style_path = 'Green_Sea_Turtle_grazing_seagrass.jpg')
    # to obtain more realistic colors, we swap the style image to an actual photograph of a wave
    compute_and_visualize_transfer(content_path = 'The_Great_Wave_off_Kanagawa.jpg', style_path = 'Welle.jpg')
    # we increase the style weight by a factor of 10
    compute_and_visualize_transfer(content_path = 'The_Great_Wave_off_Kanagawa.jpg', style_path = 'Welle 2.jpg', style_weight=0.01)
    # here we try to make a portrait painting more realistic by using a portrait photograph as style image
    compute_and_visualize_transfer(content_path = 'Van Gogh Portrait.jpg', style_path = 'Van Gogh Lookalike.jpg', style_weight=0.001)