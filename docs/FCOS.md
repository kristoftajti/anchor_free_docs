# FCOS paper review
### official link - https://arxiv.org/pdf/1904.01355
## Downsides of anchor-based detectors
- Difficulties to deal with objects having large shape variations
- Worsens the generalization ability of the detectors
- For high recall, excessive number of boxes are needed during training in the neck, most of which are negative samples, which creates an imbalance between negative and positive samples.
- Complicated calculations are needed (IoU /w GTs)
- Many avoidable hyper-parameters (IoU threshold)
## Apporach
