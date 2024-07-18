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
grids = [[gridxgridx2], [gridxgridx2], ...] #Each neck level has its grid coords saved
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

    x_shifts.append(grid[:, :, 0]) 
    y_shifts.append(grid[:, :, 1])
    # Each neck level, we take all the grids (i, j) or (x, y) coordinates

    if self.training:
        return self.get_losses(
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            torch.cat(outputs, 1),
            origin_preds,
            dtype=xin[0].dtype,
        )
````

````python
def get_output_and_grid(self, output, k, stride, dtype):
...
output[..., :2] = (output[..., :2] + grid) * stride 
# xy rel from topleft corner + scale to whole image
output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
# w h regression + scale to whole image
# TODO: use func with dynamic properties instead of exp that could cover the whole imagesize * 1.5 for each at each neck lvl
return output, grid
````

Now we are gonna start discussing the loss and assignment part of YoloX

````python
def get_losses(...):
    # n_anchors_all = grid * grid
    bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
    obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
    cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

    ...

    # loop through each img one by one
    for batch_idx in range(outputs.shape[0]):
        ...
        (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg_img,
        ) = self.get_assignments(
            batch_idx,
            num_gt,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            obj_preds,
        )
        ...
        cls_target = F.one_hot(
            gt_matched_classes.to(torch.int64), self.num_classes
        ) * pred_ious_this_matching.unsqueeze(-1) # weighting the one-hot with IoU to scale by the quality of the pred!
        reg_target = gt_bboxes_per_image[matched_gt_inds] # TODO ?

        ...

    ...

    # Calculate all the losses after targets are calculated for each neck lvl
    num_fg = max(num_fg, 1)
    loss_iou = (
        self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
    ).sum() / num_fg
    loss_obj = (
        self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
    ).sum() / num_fg
    loss_cls = (
        self.bcewithlog_loss(
            cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
        )
    ).sum() / num_fg
    if self.use_l1:
        loss_l1 = (
            self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
        ).sum() / num_fg
    else:
        loss_l1 = 0.0

    reg_weight = 5.0
    loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

    return (
        loss,
        reg_weight * loss_iou,
        loss_obj,
        loss_cls,
        loss_l1,
        num_fg / max(num_gts, 1),
    )
````

````python
def get_assignments(...):
    # fg_mask - checks if any gtbox is in the given "anchor" (i,j coor in fpn featuremap)
    # geometry_relation - valid anchors for each gt boxes
    fg_mask, geometry_relation = self.get_geometry_constraint(
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
    )

    bboxes_preds_per_image = bboxes_preds_per_image[fg_mask] # keep only those where a valid anchor is present
    # check get_geometry_constraint to understand what are valid anchors
    ... # Do this for all preds - cls, obj

    pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False) # iou between gt and preds (all gts and all preds)

    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8) # Why like this? idk

    pair_wise_cls_loss = F.binary_cross_entropy(
        cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
        gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
        reduction="none"
    ).sum(-1) # cls loss between all gts and anchprs

    cost = (
        pair_wise_cls_loss
        + 3.0 * pair_wise_ious_loss
        + float(1e6) * (~geometry_relation)
    ) # take only those losses into consideration where the anchor is valid (check get_geometry_constraint for further info)

    (
        num_fg,
        gt_matched_classes,
        pred_ious_this_matching,
        matched_gt_inds,
    ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

    return (
        gt_matched_classes,
        fg_mask,
        pred_ious_this_matching,
        matched_gt_inds,
        num_fg,
    )

````

````python
def get_geometry_constraint(...):
    # map center each FPN featuremap level (i,j) to the original image 
    x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0) # centers in respect to the orig img
    y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0) # centers in respect to the orig img

    # in fixed center
    # Different barriers are applied based on the FPN level, bigger area is accepted from lower FPN levels (1.5 x 32 > 1.5 x 8)
    center_radius = 1.5
    center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius # center radius for fpn level
    gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist 
    gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist # calculation of center region barrier on axis y
    gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
    gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist # calculation of center region barrier on axis x
    

    # calculating the actual center regions for boxes on the img
    c_l = x_centers_per_image - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - x_centers_per_image
    c_t = y_centers_per_image - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - y_centers_per_image

    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    # check whether the mapped centers are in the accepted center range
    is_in_centers = center_deltas.min(dim=-1).values > 0.0 
    anchor_filter = is_in_centers.sum(dim=0) > 0 # check if any gtbox is in the given "anchor" (i,j coor in fpn featuremap)
    geometry_relation = is_in_centers[:, anchor_filter] # valid anchors for each gt boxes

    return anchor_filter, geometry_relation
````

````python
def simota_matching(...):
    n_candidate_k = min(10, pair_wise_ious.size(1)) # max 10 candidate per gt
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1) # dynamically calculate the number of anchors each gt should be paired with
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(
            cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
        ) # get dynamicks number of pos_idxs based on the lowest cost 
        matching_matrix[gt_idx][pos_idx] = 1 # make the matchingmatrix 1 where posidx

    del topk_ious, dynamic_ks, pos_idx

    anchor_matching_gt = matching_matrix.sum(0)
    # deal with the case that one anchor matches multiple ground-truths - (1 anchor can only be matched to 1 gt, but 1 gt can have multiple anchors)
    # we keep the one that has lowest cost
    if anchor_matching_gt.max() > 1:
        multiple_match_mask = anchor_matching_gt > 1
        _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
        matching_matrix[:, multiple_match_mask] *= 0
        matching_matrix[cost_argmin, multiple_match_mask] = 1

    fg_mask_inboxes = anchor_matching_gt > 0
    num_fg = fg_mask_inboxes.sum().item()
    fg_mask[fg_mask.clone()] = fg_mask_inboxes

    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0) # gt indexes for each anchor
    gt_matched_classes = gt_classes[matched_gt_inds]

    # ious between anchor and gts
    pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
        fg_mask_inboxes
    ]

    # numfg - number of foreground anchors - anchros selected for gts
    # gt_matched_classes - gt classes for each anchor
    # pred_ious_this_matching - ious between anchor and gts
    # matched_gt_inds - gt indexes for each anchor
    return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds  
````