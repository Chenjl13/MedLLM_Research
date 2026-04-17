# Welcome to the successful reimplementation of MSLoRA!

## Our model
### The problems of our model
20260405:

The model mostly uses text for reasoning, with visual information having very little impact. Some images with obvious lesions weren’t diagnosed correctly, showing the model struggles to use visual information well. Also, a few images couldn’t be loaded into the input area properly.

Like, when we input a male patient’s image but a female description in text, the model only focuses on the text and doesn’t recognize the image. The second image clearly has a lesion, but the model missed it. So basically, the visual part’s weight in reasoning is too low, and we need to fix that.

<img src="imgs/problems_20260405.png">

### [here](Original/readme.md) is the configuration of original paper.
