Main Files to execute are: 
- /VGG13/VGG13_Training.py to train the network
- /VGG13/VGG13_Style_Transfer.py for transfering style <- For this to work you have to change the path to the previously exported weights accordingly in line 139
- /Pretrained VGG/Style_Transfer.py

-----Abstract----
This paper is an implementation of an artistic style transfer approach based on “A Neural Algorithm of Artistic Style” by Gatys, Ecker and Bethge (2018). 
We start by giving a short explanation of what the transfer of artistic style is and follow up by giving a brief overview of different approaches towards 
the challenge of style transfer. We continue with a more detailed explanation of the pretrained network and loss functions which were used in the original 
paper as well as by us. Subsequently, we describe our own implementation of a similar network and present the results we obtained with it. In the following 
we present and compare results we obtained by using pre trained networks and evaluate how different parameters influence the result. Additionally, we 
investigate the inverse application of making art more realistic and use our results to point out the limitations of our approach. In the end we give a 
summary of our findings and outline what additional research could be done.

- Python Version: 3.6.5
