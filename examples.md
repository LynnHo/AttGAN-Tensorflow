# <p align="center"> [AttGAN](https://ieeexplore.ieee.org/document/8718508?source=authoralert) Usage </p>

- training

    - for 128x128 images

        ```console
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --load_size 143 \
        --crop_size 128 \
        --model model_128 \
        --experiment_name AttGAN_128
        ```

    - for 128x128 images with all **40** attributes!

        ```console
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --load_size 143 \
        --crop_size 128 \
        --model model_128 \
        --experiment_name AttGAN_128_40 \
        --att_names 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes \
                    Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry \
                    Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee \
                    Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open \
                    Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose \
                    Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair \
                    Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick \
                    Wearing_Necklace Wearing_Necktie Young
        ```

    - for 256x256 images

        ```console
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --load_size 286 \
        --crop_size 256 \
        --model model_256 \
        --experiment_name AttGAN_256
        ```

    - for 384x384 images

        ```console
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --load_size 429 \
        --crop_size 384 \
        --model model_384 \
        --experiment_name AttGAN_384
        ```