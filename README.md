# UM DSAI Bachelor's Thesis (2025)
## Accompanying Code
 
This repository provides the code and the necessary datasets to run the experiments conducted for the Maastricht University Data Science and Artificial Intelligence Bachelor's mandatory thesis component. 

The notebooks and scripts are organized according to the research questions to which they belong. 

**RQ1:** 
Simply run the code for each model to output scores on art and faces domains.

**RQ2:** 
To create the models "Fine-Tuned (Faces)", "Fine-Tuned (Art)", and "Fine-Tuned (Mixed)" use `rq2_finetuning.ipynb`. Scroll to the **Parameters** section and choose the dataset to use and the corresponding output directory. To test the trained models, use `rq2_testing.py` and enter the correct model name (e.g. "./sdxl-fine-tune-art") into the pipeline.

For the augmentations, you can create an augmented dataset through `rq2_augmenting.py` by changing the last line of the code with the dataset that you want to augment, the percentage of images that you wish to augment, and the output directory. You may then use the created dataset in `rq2_finetuning.ipynb` by replacing one of the dataset paths in the **Load Datasets and Processor** section of the notebook. (e.g. change the `art_dataset_path` from `archive/datasets/art_512x512` to `archive/datasets/art_aug05`)

Finally, to test any model on a randomly augmented set of images, use the `rq2_testing_aug.py` file and replace the model name in the pipeline. No other changes need to be made to this file but you can change the probabilities of applying different transformations. 

**RQ3:**
In `rq3_fewshot.ipynb`, head to the **Define Model and Data** section and choose the model to which you want to apply few-shot learning. In the same section, choose the dataset to use for the few-shot learning. Then, scroll to the **Create Few-Shot Set** section and find the line `few_shot_ds = create_few_shot_set(...)`. Here, you can change the set size, i.e. the number of images to do FSL with. Recall that a set size of 20 implies 10-shot learning. This is important if you want the results to match those of the thesis. Finally, you can change the number of epochs in the TrainingArguments() declaration under **Prepare Model For Training**.

To test the models, use `rq3_testing.py` which works the same way as the previous testing scripts.

## Datasets and Models 
Albert5913 (2025). Ai generated dogs.jpg vs real dogs.jpg. https://www.kaggle.com/datasets/albertobircoci/ai-generated-dogs-jpg-vs-real-dogs-jpg <br/>
AlDahoul, N. and Zaki, Y. Nyuad ai-generated images detector. https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector <br/>
Kannan, K. (2023). Ai and human art classification. https://www.kaggle.com/datasets/kausthubkannan/ai-and-human-art-classification <br/>
Nahrawy (2023). Ai or not. https://huggingface.co/Nahrawy/AIorNot <br/>
Organika (2024). Sdxl detector. https://huggingface.co/Organika/sdxl-detector <br/>
prithivMLmods (2023). Deepfake real class siglip2. https://huggingface.co/prithivMLmods/Deepfake-Real-Class-Siglip2 <br/>
VM7608 (2024). Real vs ai generated faces dataset. https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset. <br/>

## Licensing

**📂 Faces Dataset: Reduced Version of [Real vs AI Generated Faces Dataset](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset/data)**

This is a reduced version of the *Real vs AI Generated Faces Dataset*, originally created by Mark Otto and Andrew Fong.  
The original dataset is licensed under the MIT License.

Redistribution of this modified version is permitted under the same license.  
See `faces_512x512/LICENSE_dataset.txt` for the full license terms.

**📂 Dogs Dataset: Reduced Version of [Ai Generated Dogs.jpg VS Real Dogs.jpg](https://www.kaggle.com/datasets/albertobircoci/ai-generated-dogs-jpg-vs-real-dogs-jpg)**

This dataset is based on a public domain dataset released under the [CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/).

You are free to copy, modify, and redistribute this data for any purpose, including commercial use, without asking for permission.

Note: This does not imply any affiliation with or endorsement by the original creators.

**📂 Art Dataset: Reduced version of [AI and Human Art Classification](https://www.kaggle.com/datasets/kausthubkannan/ai-and-human-art-classification)**

This dataset is a reduced version of the original, which is licensed under the Open Database License (ODbL) v1.0.  

Accordingly, this reduced version is also released under the same license.  

See `art_512x512/LICENSE_dataset.txt` for full terms.

