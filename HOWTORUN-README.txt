First install pycocotools.
And download pre-trained COCO weights (mask_rcnn_coco.h5) from:
  https://github.com/matterport/Mask_RCNN/releases
into './'

train:
  OMP_NUM_THREADS=56 KMP_HW_SUBSET=1T KMP_AFFINITY=compact,granularity=fine python3 coco.py train --dataset=xxx/ --model=coco --trainbs=2 --num_intra_threads=56 --num_inter_threads=1

inference:
  OMP_NUM_THREADS=56 KMP_HW_SUBSET=1T KMP_AFFINITY=compact,granularity=fine python3 coco.py evaluate --dataset=xxx/ --model=coco --infbs=1 --num_intra_threads=56 --num_inter_threads=1
