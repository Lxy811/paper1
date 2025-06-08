CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port 10016 --nproc_per_node=1 \
        tools/relation_train_net.py \
        --config-file "configs/e2e_relation_VGG16_1x.yaml" \
        SOLVER.PRE_VAL False \
        MODEL.ROI_RELATION_HEAD.LAMBDA_ 0.01 \
        MODEL.ROI_RELATION_HEAD.PRUNE_RATE 0.85 \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.PREDICTOR TCARPredictor \
        SOLVER.IMS_PER_BATCH 16 \
        TEST.IMS_PER_BATCH 2 \
        DTYPE "float16" \
        SOLVER.MAX_ITER 16000 \
        SOLVER.BASE_LR 0.001 \
        SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
        SOLVER.STEPS "(10000, 16000)" \
        SOLVER.VAL_PERIOD 20000 \
        SOLVER.CHECKPOINT_PERIOD 16000 \
        GAMMA 0.5 \
        GLOVE_DIR /media/n702/data1/Lxy/datasets/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT /media/n702/data1/Lxy/datasets/vg/pretrained_faster_rcnn_VGG16/model_final.pth \
        OUTPUT_DIR ./checkpoints/VGG16/TCARPredictor-VGG-sgdet