cd /SSDc/Workspaces/seunghyeok_back/mask-eee-rcnn && conda activate seung
cd /SSDa/workspace/seunghyeok_back/mask-eee-rcnn && conda activate seung

python setup.py install develop
pip install setuptools==59.5.0
pip install fvcore==0.1.1.dev200512



CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_3x_bs2_nct3_we-1.yaml

CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_3x_bs2_nct3.yaml


CUDA_VISIBLE_DEVICES=6 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_scoring_rcnn_R_50_FPN_3x_bs2.yaml


CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs2_conv3_weight0.1.yaml

CUDA_VISIBLE_DEVICES=3 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs2_conv3.yaml

CUDA_VISIBLE_DEVICES=5 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs2_conv5.yaml


CUDA_VISIBLE_DEVICES=7 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_scoring_rcnn_R_50_FPN_1x_bs2.yaml