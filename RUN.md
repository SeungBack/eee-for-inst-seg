cd /SSDc/Workspaces/seunghyeok_back/mask-eee-rcnn && conda activate seung


CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_bs2.yaml

CUDA_VISIBLE_DEVICES=2 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_scoring_rcnn_R_50_FPN_1x_bs2.yaml

CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs2_nct1.yaml

CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs2_nct2.yaml

CUDA_VISIBLE_DEVICES=2 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs2_nct3.yaml

CUDA_VISIBLE_DEVICES=3 python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs2_nct3_tp_only.yaml