# FCOS code review
### official link - [https://github.com/tianzhi0549/FCOS/tree/master/](https://github.com/tianzhi0549/FCOS/tree/master/)

#### FCOSModule - FCOS/fcos_core/modeling/rpn/fcos/fcos.py

Includes the FCOShead, loss_evaluator, and the box_selector_test, which is only used when running inference.

The **forward** call of the FCOSModule looks quite simple, predicts the incoming features from the backbone with the FCOSHead, where the features are a list of tensors, with tensors for each level we are predicting at.

````python
def forward(self, images, features, targets=None):
    box_cls, box_regression, centerness = self.head(features)
    locations = self.compute_locations(features)

    if self.training:
        return self._forward_train(
            locations, box_cls, 
            box_regression, 
            centerness, targets
        )
    else:
        return self._forward_test(
            locations, box_cls, box_regression, 
            centerness, images.image_sizes
        )
````

Then the locations are computed for all features. Meaning we transform back each $(x,y)$ feature map point to the original image (we take the middle point of the feature map location in the original image). 

````python
def compute_locations(self, features):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        locations_per_level = self.compute_locations_per_level(
            h, w, self.fpn_strides[level],
            feature.device
        )
        locations.append(locations_per_level)
    return locations
````
The **compute_locations_per_level** function uses the formula provided in the original paper. One can see an easy example here.

````python
shifts_x = torch.arange(0, 3 * 8, step=8)  # [0, 8, 16]
shifts_y = torch.arange(0, 3 * 8, step=8)  # [0, 8, 16]
shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
shift_x = shift_x.reshape(-1)  # [0, 8, 16, 0, 8, 16, 0, 8, 16]
shift_y = shift_y.reshape(-1)  # [0, 0, 0, 8, 8, 8, 16, 16, 16]
locations = torch.stack((shift_x, shift_y), dim=1) + 8 // 2
# [[4, 4], [12, 4], [20, 4], [4, 12], [12, 12], [20, 12], [4, 20], [12, 20], [20, 20]]
````

Then the **_forward_train** function calculates classification, regression and centerness loss and returns it.



### FCOSHead - FCOS/fcos_core/modeling/rpn/fcos/fcos.py

The head is instansiated in the FCOSModule, with in_channels param, which is a hyper-parameter. We have two "towers", one for classification and one for regression, both with $N$ number of $inchannel \times inchannel$  convolutions.

````python
cls_tower = []
bbox_tower = []
for i in range(cfg.MODEL.FCOS.NUM_CONVS):
    if self.use_dcn_in_tower and \
            i == cfg.MODEL.FCOS.NUM_CONVS - 1:
        conv_func = DFConv2d
    else:
        conv_func = nn.Conv2d

    cls_tower.append(
        conv_func(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
    )
    cls_tower.append(nn.GroupNorm(32, in_channels))
    cls_tower.append(nn.ReLU())
    bbox_tower.append(
        conv_func(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
    )
    bbox_tower.append(nn.GroupNorm(32, in_channels))
    bbox_tower.append(nn.ReLU())
````

The end of both towers are 2d convolutions, the cls_tower with num_classes output, the bbox_tower with 4 ($$t^*, b^*, r^*, b^*$$) and a centerness also ending with a 2d convolution of outputsize 1, as we want 1 number inbetween 0 and 1.

In the forward call, the centerness is calculated from the output of the cls_tower (optionally one can use the regression branch for it). The head returns logits **(shape of num_classes)**, bbox_reg **(shape of 4)**, centerness **(shape of 1)**

````python
def forward(self, x):
    ...
    centerness.append(self.centerness(cls_tower))
    ...
    return logits, bbox_reg, centerness
````

### FCOSLoss - FCOS/fcos_core/modeling/rpn/fcos/loss.py

The loss **call** function starts of with preparing the targets.

#### Preparetargets

Here, the we declare the size of objects each level is responsible for. This is important to look out for in case one wants to change the number of FPN levels used.

````python
def prepare_targets(self, points, targets):
    object_sizes_of_interest = [
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, INF],
    ]
    ...
````

After, we create a tensor that contains this range aforementioned for each location at all levels, called **expanded_object_sizes_of_interest**. We also store the number of locations for each level in **num_points_per_level** and all the points combined in **points_all_level**. Then we call **compute_targets_for_locations**, which creates labels and reg_targets.

Here basically we calculate $$(l^*, t^*, r^*, b^*)$$

````python
...
l = xs[:, None] - bboxes[:, 0][None]
t = ys[:, None] - bboxes[:, 1][None]
r = bboxes[:, 2][None] - xs[:, None]
b = bboxes[:, 3][None] - ys[:, None]
reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
...
````

Then filter those boxes which are not in the allowed radious of the GT boxes center point (in case **center_sampling_radius** is set, but let's get real, it needs to be set), plus filter with the range for each level that is provided, and finally in case we still have ambigous cases, we take the box with the smallest area.

````python
for im_i in range(len(targets)):
    ...
    # Filtering /w range (here we just calculate the mask, it is applied later)
    if self.center_sampling_radius > 0:
        is_in_boxes = self.get_sample_region(
            bboxes,
            self.fpn_strides,
            self.num_points_per_level,
            xs, ys,
            radius=self.center_sampling_radius
        )
    ...
    # Limit the regression range for each location (here we just calculate the mask, it is applied later)
    is_cared_in_the_level = \
    (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
    (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
    ...
    # We choose the one with minimal area
    locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
    ...
return labels, reg_targets
````

Finally, we iterate through each level of the FPN, and gather the labels and regression targets for each level, we also norm the regression target if set. 

````python
for level in range(len(points)):
    labels_level_first.append(
        torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
    )

    reg_targets_per_level = torch.cat([
        reg_targets_per_im[level]
        for reg_targets_per_im in reg_targets
    ], dim=0)
...
return labels_level_first, reg_targets_level_first
````

After creating the labels and targets, we get back to the **call** , we flatten and premute the predictions so that we can concat them for more efficient computing, and then we get the positive samples and filter the predictions to only include positive samples.

````python
pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

box_regression_flatten = box_regression_flatten[pos_inds]
reg_targets_flatten = reg_targets_flatten[pos_inds]
centerness_flatten = centerness_flatten[pos_inds]
````

The cls_loss (classification) is calculated with **SigmoidFocalLoss**, and the regression loss is **IOULoss**, while the centerness loss is **BCEWithLogitsLoss**. All is normalized with the number of positive samples.

Inference.py is not discussed here, might be in the future in case it is needed.