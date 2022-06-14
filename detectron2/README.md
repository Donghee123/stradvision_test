Stradvision coding test

Test3 Objectdection with ensemble

폴더 구조

└─detectron2

    ├─configs

    │  ├─Cityscapes

    │  ├─COCO-Detection

    │  ├─COCO-InstanceSegmentation

    │  ├─COCO-Keypoints

    │  ├─COCO-PanopticSegmentation

    │  ├─common

    │  │  ├─data

    │  │  └─models

    │  ├─Detectron1-Comparisons

    │  ├─LVISv0.5-InstanceSegmentation

    │  ├─LVISv1-InstanceSegmentation

    │  ├─Misc

    │  ├─new_baselines

    │  ├─PascalVOC-Detection

    │  └─quick_schedules

    ├─datasets

    ├─demo

    │  ├─images

    │  └─__pycache__

    ├─detectron2

    │  ├─checkpoint

    │  ├─config

    │  ├─data

    │  │  ├─datasets

    │  │  ├─samplers

    │  │  └─transforms

    │  ├─engine

    │  ├─evaluation

    │  ├─export

    │  ├─layers

    │  │  └─csrc

    │  │      ├─box_iou_rotated

    │  │      ├─cocoeval

    │  │      ├─deformable

    │  │      ├─nms_rotated

    │  │      └─ROIAlignRotated

    │  ├─modeling

    │  │  ├─backbone

    │  │  ├─meta_arch

    │  │  ├─proposal_generator

    │  │  └─roi_heads

    │  ├─model_zoo

    │  ├─projects

    │  ├─solver

    │  ├─structures

    │  ├─tracking

    │  └─utils

    ├─dev

    │  └─packaging

    ├─docker

    ├─docs

    │  ├─modules

    │  ├─notes

    │  ├─tutorials

    │  └─_static

    │      └─css

    ├─output

    │  └─inference

    │      └─coco_2017_val

    │          └─instances_predictions

    │              └─archive

    ├─projects

    │  ├─DeepLab

    │  │  └─deeplab

    │  ├─DensePose

    │  │  ├─densepose

    │  │  │  ├─converters

    │  │  │  ├─data

    │  │  │  │  ├─datasets

    │  │  │  │  ├─meshes

    │  │  │  │  ├─samplers

    │  │  │  │  ├─transform

    │  │  │  │  └─video

    │  │  │  ├─engine

    │  │  │  ├─evaluation

    │  │  │  ├─modeling

    │  │  │  │  ├─cse

    │  │  │  │  ├─losses

    │  │  │  │  ├─predictors

    │  │  │  │  └─roi_heads

    │  │  │  ├─structures

    │  │  │  ├─utils

    │  │  │  └─vis

    │  │  ├─dev

    │  │  ├─doc

    │  │  └─tests

    │  ├─Panoptic-DeepLab

    │  │  └─panoptic_deeplab

    │  ├─PointRend

    │  │  └─point_rend

    │  ├─PointSup

    │  │  ├─point_sup

    │  │  └─tools

    │  ├─Rethinking-BatchNorm

    │  │  └─configs

    │  ├─TensorMask

    │  │  ├─tensormask

    │  │  │  └─layers

    │  │  │      └─csrc

    │  │  │          └─SwapAlign2Nat

    │  │  └─tests

    │  └─TridentNet

    │      └─tridentnet

    ├─tests

    │  ├─config

    │  │  └─dir1

    │  ├─data

    │  ├─export

    │  ├─layers

    │  ├─modeling

    │  ├─structures

    │  └─tracking

    └─tools

        └─deploy



추가 코드 작업 장소 및 스크립트 이름


└─detectron2

    ├─demo

    │  ├─analysis_ensemble_by_F1ScoreTable.ipynb

    │  ├─analysis_ensemble_by_union.ipynb

    │  ├─analysis_retinanet_R_50_fpn_1x.ipynb

    │  ├─analysis_retinanet_R_50_fpn_3x.ipynb

    │  ├─analysis_retinanet_R_101_fpn_3x.ipynb

    │  ├─make_ensemble_union_predctions.py

    │  ├─make_f1_scroe_table.py

    │  ├─make_model_predctions.py


스크립트 설명

- analysis_ensemble_by_F1ScoreTable.ipynb
F1 Score table 기반 ensemble method을 구동시키고 평가합니다.

- analysis_ensemble_by_uninon.ipynb
uninon 기반 ensemble method을 구동시키고 평가합니다. (발표 자료에는 쓰지 않았습니다.)

- analysis_retinanet_R_50_fpn_1x.ipynb
retinanet_R_50_fpn_1x 모델을 구동시키고 평가합니다. (발표 자료에는 쓰지 않았습니다.)

- analysis_retinanet_R_50_fpn_3x.ipynb
retinanet_R_50_fpn_3x 모델을 구동시키고 평가합니다. (발표 자료에는 쓰지 않았습니다.)

- analysis_retinanet_R_101_fpn_3x.ipynb
retinanet_R_101_fpn_3x 모델을 구동시키고 평가합니다. (발표 자료에는 쓰지 않았습니다.)

- make_ensemble_union_predctions.py
uninon 기반 ensemble method의 prediction 결과를 만들고 저장 합니다.

- make_f1_scroe_table.py
각 모델들(retinanet_R_50_fpn_1x, retinanet_R_50_fpn_3x, retinanet_R_101_fpn_3x)의 predction 결과를 보고 f1 scroe table을 저장 합니다.

- make_model_predctions.py
각 모델들(retinanet_R_50_fpn_1x, retinanet_R_50_fpn_3x, retinanet_R_101_fpn_3x)의 predction 결과를 만들고 저장합니다.


스크립트 실행 순서

1. make_model_predctions.py
2. analysis_retinanet_R_50_fpn_1x.ipynb
3. analysis_retinanet_R_50_fpn_3x.ipynb
4. analysis_retinanet_R_101_fpn_3x.ipynb
5. 수기로 각 모델들의 {모델 이름}_result.txt 저장하기 (예시 retinanet_R_50_fpn_1x_result.txt, retinanet_R_50_fpn_3x_result.txt, retinanet_R_101_fpn_3x_result.txt)
6. make_f1_scroe_table.py
7. analysis_ensemble_by_F1ScoreTable.ipynb