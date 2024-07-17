<!-- Button to go back to the main page -->
<div style="margin-top: 20px;">
  <a href="../" style="text-decoration: none;">
    <button style="
      background-color: #4CAF50; /* Green */
      border: none;
      color: white;
      padding: 10px 20px;
      text-align: center;
      text-decoration: none;
      display: inline-block;
      font-size: 16px;
      cursor: pointer;
    ">Back to Main Page</button>
  </a>
</div>

# YoloX code review
### official link - [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)


YoloX is an anchor-free Yolo version from 2021. This review will focus on the anchor-free part of the repo.

### Forward - yolox.py

In the aforementioned file, happens the forward pass through the backbone and the head as well

````python
fpn_outs = self.backbone(x)
# type(fpn_outs) => Tuple(size=fpn_neck_lvl_nums)
# fpn_outs[N].shape = batch_size * feature_map_num * grid * grid
...
outputs = self.head(fpn_outs)
#  !in case of training, check code for inference outputs!

#type(outputs) => Dict(keys={loss, iou_loss, l1_loss, conf_loss, cls_loss, numfg (number of foreground anchors)})
````

As the backbone is pretty straight standard, I won't cover it here, in case one is interested, feel free to check it out. 

### Forward - yolo_head.py

Important to know, that stride_level is stored in the head in a list. This is to map the neck feature map back to the original image, similar to what is happening in FCOS.

````python
strides=[8, 16, 32]
````

````python
# loop through each FPN lvl
for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
    zip(self.cls_convs, self.reg_convs, self.strides, xin)
):
    ...
    cls_output = self.cls_preds[k](cls_feat)
    # cls_output shape - batch x classnum x grid x grid

    reg_output = self.reg_preds[k](reg_feat)
    # reg_output shape - batch x 4 (two offsets from left-top corner of the grid, and the height and width) x grid x grid

    obj_output = self.obj_preds[k](reg_feat)
    # obj_output shape - batch x 1 x grid x grid
    ...

    output, grid = self.get_output_and_grid(
        output, k, stride_this_level, xin[0].type()
    )
    # The output is scaled back to the original image, and grid contains grid infomation for all the outputs, showing in which grid they are in
    # output.shape => batchsize x (grid x grid) x 5 + class_num
    # grid.shape => batchsize x (grid x grid) x 2 (i,j)
    # get_output_and_grid is dicussed in detail below!
````

````python
def get_output_and_grid(self, output, k, stride, dtype):
    ...
````
