# FCOS code review
### official link - [https://github.com/tianzhi0549/FCOS/tree/master/](https://github.com/tianzhi0549/FCOS/tree/master/)

#### FCOSModule

Includes the FCOShead, loss_evaluator, and the box_selector_test, which is only used when running inference.

the **forward** call of the FCOSModule looks quite simple, predicts the incoming features from the backbone with the FCOSHead, where the features are a list of tensors, with tensors for each level we are predicting at.

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



#### FCOSHead

The head is instansiated in the FCOSModule, with in_channels param, which is a hyper-parameter. We do have two "towers", one for classification and one for regression, both with $N$ number of $inchannel \times inchannel$  convolutions.

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

In the forward call, the centerness is calculated from the output of the cls_tower (optionall one can use the regression branch for it, but this one is mentioned in the paper). The head returns logits **(shape of num_classes)**, bbox_reg **(shape of 4)**, centerness **(shape of 1)**

````python
def forward(self, x):
    ...
    centerness.append(self.centerness(cls_tower))
    ...
    return logits, bbox_reg, centerness
````

#### FCOSLoss