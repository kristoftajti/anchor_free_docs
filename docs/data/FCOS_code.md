# FCOS code review
### official link - [https://github.com/tianzhi0549/FCOS/tree/master/](https://github.com/tianzhi0549/FCOS/tree/master/)

#### FCOSHead

The head is instansiated in the FCOSModule (will be discussed later in detail), with in_channels param, which is a hyper-parameter. We do have two "towers", one for classification and one for regression, both with $N$ number of $inchannel \times inchannel$  convolutions.

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

The end of both towers are 2d convolutions, the cls_tower with num_classes output, the bbox_tower with 4 ($t^*, b^*, r^*, b^*$) and a centerness also ending with a 2d convolution of outputsize 1, as we want 1 number inbetween 0 and 1.

In the forward call, the centerness is calculated from the output of the cls_tower (optionall one can use the regression branch for it, but this one is mentioned in the paper). The head returns logits **(shape of num_classes)**, bbox_reg **(shape of 4)**, centerness **(shape of 1)**

````python
def forward(self, x): ...
    centerness.append(self.centerness(cls_tower))
    ...
    return logits, bbox_reg, centerness
````