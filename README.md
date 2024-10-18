***News***

- 11 Jan 2020: We clean up the code to make it more readable! The old version is here: [v1](https://github.com/LynnHo/AttGAN-Tensorflow/tree/v1).

#

<p align="center"> <img src="./pics/first_view.png" width="36%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="./pics/slide.png" width="54%"> </p>

<p align="center"> <img src="./pics/style.jpg" width="92%"> </p>

<hr style="height:1px" />

# <p align="center"> [AttGAN](https://ieeexplore.ieee.org/document/8718508?source=authoralert) <br> <sub><sub> [TIP Nov. 2019](https://ieeexplore.ieee.org/document/8718508?source=authoralert), [arXiv Nov. 2017](https://arxiv.org/pdf/1711.10678v1.pdf) </sub></sub> </p>

**TensorFlow** implementation of **AttGAN**: Facial Attribute Editing by Only Changing What You Want.

<p align="center"> <img src="./pics/schema.jpg" width="100%"> </p>

## Related

- Other implementations of AttGAN

    - [AttGAN-PyTorch](https://github.com/elvisyjlin/AttGAN-PyTorch) by Yu-Jing Lin

    - [AttGAN-PaddlePaddle](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) by ceci3 and zhumanyu (**AttGAN is one of the official reproduced models of [PaddlePaddle](https://github.com/PaddlePaddle?type=source)**)

- Closely related works

    - **An excellent work built upon our code - [STGAN](https://github.com/csmliu/STGAN) (CVPR 2019) by Ming Liu**

    - [Changing-the-Memorability](https://github.com/acecreamu/Changing-the-Memorability) (CVPR 2019 MBCCV Workshop) by acecreamu

    - [Fashion-AttGAN](https://github.com/ChanningPing/Fashion_Attribute_Editing) (CVPR 2019 FSS-USAD Workshop) by Qing Ping

- An unofficial [demo video](https://www.youtube.com/watch?v=gnN4ZjEWe-8) of AttGAN by 王一凡

## Exemplar Results

- See [results.md](./results.md) for more results, we try **higher resolution** and **more attributes** (all **40** attributes!!!)

- Inverting 13 attributes respectively

    from left to right: *Input, Reconstruction, Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Male, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Young*

    <img src="./pics/sample_validation.jpg" width="95%">

## Usage

- Environment

    - Python 3.6

    - TensorFlow 1.15

    - OpenCV, scikit-image, tqdm, oyaml

    - *we recommend [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers), then you can create the AttGAN environment with commands below*

        ```console
        conda create -n AttGAN python=3.6

        source activate AttGAN

        conda install opencv scikit-image tqdm tensorflow-gpu=1.15

        conda install -c conda-forge oyaml
        ```

    - *NOTICE: if you create a new conda environment, remember to activate it before any other command*

        ```console
        source activate AttGAN
        ```

- Data Preparation

    - Option 1: [CelebA](http://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf)-unaligned (higher quality than the aligned data, 10.2GB)

        - download the dataset

            - img_celeba.7z (move to **./data/img_celeba/img_celeba.7z**): [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ) or [Baidu Netdisk](https://pan.baidu.com/s/1CRxxhoQ97A5qbsKO7iaAJg) (password rp0s)

            - annotations.zip (move to **./data/img_celeba/annotations.zip**): [Google Drive](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view?usp=sharing)

        - unzip and process the data

            ```console
            7z x ./data/img_celeba/img_celeba.7z/img_celeba.7z.001 -o./data/img_celeba/

            unzip ./data/img_celeba/annotations.zip -d ./data/img_celeba/

            python ./scripts/align.py
            ```

    - Option 2: CelebA-HQ (we use the data from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), 3.2GB)

        - CelebAMask-HQ.zip (move to **./data/CelebAMask-HQ.zip**): [Google Drive](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv) or [Baidu Netdisk](https://pan.baidu.com/s/1wN1E-B1bJ7mE1mrn9loj5g)

        - unzip and process the data

            ```console
            unzip ./data/CelebAMask-HQ.zip -d ./data/

            python ./scripts/split_CelebA-HQ.py
            ```

- Run AttGAN

    - training (see [examples.md](./examples.md) for more training commands)

        ```console
        \\ for CelebA
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --load_size 143 \
        --crop_size 128 \
        --model model_128 \
        --experiment_name AttGAN_128

        \\ for CelebA-HQ
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --img_dir ./data/CelebAMask-HQ/CelebA-HQ-img \
        --train_label_path ./data/CelebAMask-HQ/train_label.txt \
        --val_label_path ./data/CelebAMask-HQ/val_label.txt \
        --load_size 128 \
        --crop_size 128 \
        --n_epochs 200 \
        --epoch_start_decay 100 \
        --model model_128 \
        --experiment_name AttGAN_128_CelebA-HQ
        ```

    - testing

        - **single** attribute editing (inversion)

            ```console
            \\ for CelebA
            CUDA_VISIBLE_DEVICES=0 \
            python test.py \
            --experiment_name AttGAN_128

            \\ for CelebA-HQ
            CUDA_VISIBLE_DEVICES=0 \
            python test.py \
            --img_dir ./data/CelebAMask-HQ/CelebA-HQ-img \
            --test_label_path ./data/CelebAMask-HQ/test_label.txt \
            --experiment_name AttGAN_128_CelebA-HQ
            ```


        - **multiple** attribute editing (inversion) example

            ```console
            \\ for CelebA
            CUDA_VISIBLE_DEVICES=0 \
            python test_multi.py \
            --test_att_names Bushy_Eyebrows Pale_Skin \
            --experiment_name AttGAN_128
            ```

        - attribute sliding example

            ```console
            \\ for CelebA
            CUDA_VISIBLE_DEVICES=0 \
            python test_slide.py \
            --test_att_name Pale_Skin \
            --test_int_min -2 \
            --test_int_max 2 \
            --test_int_step 0.5 \
            --experiment_name AttGAN_128
            ```

    - loss visualization

        ```console
        CUDA_VISIBLE_DEVICES='' \
        tensorboard \
        --logdir ./output/AttGAN_128/summaries \
        --port 6006
        ```

    - convert trained model to .pb file

        ```console
        python to_pb.py --experiment_name AttGAN_128
        ```

- Using Trained Weights

    - alternative trained weights (move to **./output/\*.zip**)

        - [AttGAN_128.zip](https://drive.google.com/file/d/1Oy4F1xtYdxj4iyiLyaEd-dkGIJ0mwo41/view?usp=sharing) (987.5MB)

            - *including G, D, and the state of the optimizer*

        - [AttGAN_128_generator_only.zip](https://drive.google.com/file/d/1lcQ-ijNrGD4919eJ5Dv-7ja5rsx5p0Tp/view?usp=sharing) (161.5MB)

            - *G only*

        - [AttGAN_384_generator_only.zip](https://drive.google.com/open?id=1scaKWcWIpTfsV0yrWCI-wg_JDmDsKKm1) (91.1MB)


    - unzip the file (AttGAN_128.zip for example)

        ```console
        unzip ./output/AttGAN_128.zip -d ./output/
        ```

    - testing (see above)


- Example for Custom Dataset

    - [AttGAN-Cartoon](https://github.com/LynnHo/AttGAN-Cartoon-Tensorflow)

## Citation

If you find [AttGAN](https://ieeexplore.ieee.org/document/8718508?source=authoralert) useful in your research work, please consider citing:

    @ARTICLE{8718508,
    author={Z. {He} and W. {Zuo} and M. {Kan} and S. {Shan} and X. {Chen}},
    journal={IEEE Transactions on Image Processing},
    title={AttGAN: Facial Attribute Editing by Only Changing What You Want},
    year={2019},
    volume={28},
    number={11},
    pages={5464-5478},
    keywords={Face;Facial features;Task analysis;Decoding;Image reconstruction;Hair;Gallium nitride;Facial attribute editing;attribute style manipulation;adversarial learning},
    doi={10.1109/TIP.2019.2916751},
    ISSN={1057-7149},
    month={Nov},}
