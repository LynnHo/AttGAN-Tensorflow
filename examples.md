# <p align="center"> [AttGAN](https://arxiv.org/abs/1711.10678) Usage </p>

- training

    - for 128x128 images

        ```sh
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --img_size 128 \
        --shortcut_layers 1 \
        --inject_layers 1 \
        --experiment_name 128_shortcut1_inject1_none
        ```

    - for 128x128 images with all **40** attributes!

        ```sh
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --img_size 128 \
        --shortcut_layers 1 \
        --inject_layers 1 \
        --experiment_name 128_shortcut1_inject1_none_40 \
        --atts 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes \
               Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry \
               Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee \
               Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open \
               Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose \
               Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair \
               Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick \
               Wearing_Necklace Wearing_Necktie Young
        ```

    - for 256x256 images

        ```sh
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --img_size 256 \
        --shortcut_layers 1 \
        --inject_layers 1 \
        --experiment_name 256_shortcut1_inject1_none
        ```

    - for 384x384 images

        ```sh
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --img_size 384 \
        --enc_dim 48 \
        --dec_dim 48 \
        --dis_dim 48 \
        --dis_fc_dim 512 \
        --shortcut_layers 1 \
        --inject_layers 1 \
        --n_sample 24 \
        --experiment_name 384_shortcut1_inject1_none
        ```

