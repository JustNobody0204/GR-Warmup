# When Will Gradient Regularization Be Harmful?

This repo is only for review purpose.

### Training using this repo

* This repo is realized via the JAX framework. One should start by building the python environment listed in the requirements.txt file. The Python version should best be 3.8. **Be careful with the jaxlib installation. One may need manually install the correct version.**

* The **config** folder stores all the config flags and their default values. One could add extra custom flags in the files if needed. 

* The **model** folder stores the model architectures, currently including VGG, ResNet, WideResNet, PyramidNet, ViT, Swin and CaiT.
 
* The **ds_pipeline** folder is for providing dataset pipeline. Notably, when training Cifar10 or 100, one needs to sepcify the tensorflow_datasets folder. Training ImageNet and tinyimagenet dataset use the local data, not the downloaded tensorflow_datasets. One should specify the path to the local dataset folders, where the folder structure must be 
    ```
    ImageNet/TinyImageNet folder
    |
    └───n01440764
    │   │   *.JPEG
    │   
    └───n01443537
    |    │   *.JPEG
    ...
    ```

* The **optimizer** folder stores the optimizers, currently including SGD (Momentum), Adam and RMSProp.  One could add extra custom optimizers in the files if needed.

* The **recipe** folder stores the *.sh files. Each .sh file is one launching file for training a specific model. One could run it using the bash command, e.g.

    ```
    bash vit-cifar-basic.sh
    ```
    
  If one want to deploy config directly (the config flag must be in the config file), one could run with
  
    ```
    python3 -m gnp.main.main  --config=the-train-config-py-file --working_dir=your-output-dir --config.config-anything-else-here
    ```



### 3. End

Should there be any difficulties or further clarifications needed during the verification process, we encourage reviewers to reach out. We will promptly address any queries or issues that arise.

