# UM DSAI Bachelor's Thesis (2025)
### Accompanying Code
 
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