# CF-OCC
# Installation Steps
This is the official implementation of **CF-OCC** from "One Step Closer: Creating the Future to Boost Monocular Semantic Scene Completion/itsc2025".
arxiv:https://arxiv.org/abs/2507.13801

1. Install **MaskDINO** following its README instructions: [https://github.com/open-mmlab/mmdetection/pull/9808](https://github.com/open-mmlab/mmdetection/pull/9808)
2. Install required dependencies using `requirements.txt`
3. Set dataset paths in `misc.py`, and configure other paths in `confs`
4. Run `train.py` with `conf_temp` to train spatiotemporal ssc(supports DDP training)
5. Run `train.py` with `conf_pseudo` to train pose/synth net(supports DDP training)
6. Welcome to open an issue — we’ll help resolve it.

For questions regarding the paper or the code, please send an email to [luhaoang@stu.xjtu.edu.cn](mailto:luhaoang@stu.xjtu.edu.cn).
