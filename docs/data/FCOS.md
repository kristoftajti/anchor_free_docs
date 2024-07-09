<!-- Button to go back to the main page -->
<div style="margin-top: 20px;">
  <a href="./index.md" style="text-decoration: none;">
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

# FCOS paper review
### official link - [https://arxiv.org/pdf/1904.01355](https://arxiv.org/pdf/1904.01355)
## Downsides of anchor-based detectors

- Difficulties to deal with objects having large shape variations
- Worsens the generalization ability of the detectors
- For high recall, excessive number of boxes are needed during training in the neck, most of which are negative samples, which creates an imbalance between negative and positive samples.
- Complicated calculations are needed (IoU /w GTs)
- Many avoidable hyper-parameters (IoU threshold)

## Approach
### Fully Convolutional One-Stage Detector
#### Feature Maps and Ground-Truth Bounding Boxes

Let $F_i ∈ R^{H \times W \times C}$ be the feature maps at layer $i$ of a backbone CNN, where $s$ is the total stride until the layer.

Ground-truth bounding boxes for an input image are defined as $\{B_i\}$, where $B_i = (x_0^{(i)}, y_0^{(i)}, x_1^{(i)}, y_1^{(i)}, c^{(i)})$ and $B_i \in \mathbb{R}^{4} \times \{1, 2, \ldots, C\}$.

- $ B_i $: Bounding box $ i $
  - $ x_0^{(i)} $: x-coordinate of the top-left corner
  - $ y_0^{(i)} $: y-coordinate of the top-left corner
  - $ x_1^{(i)} $: x-coordinate of the bottom-right corner
  - $ y_1^{(i)} $: y-coordinate of the bottom-right corner
  - $ c^{(i)} $: Class label of the bounding box

Each location $(x, y)$ on $F_i$, can be mapped onto the input image as $(\frac{x}{2}+xs;\frac{s}{2}+ys)$, and the target boundig box is directly regressed at the feature map level, with a target 4D vector 
$$t^* = (l^*, t^*, r^*, b^*)$$
which are distances from location $(x, y)$ to the four sides of the bounding box. $(x, y)$ is considered as a positive sample in case it is inside a GT box.

$ l^* = x - x_0^{(i)}, \quad t^* = y - y_0^{(i)}, \quad r^* = x_1^{(i)} - x, \quad b^* = y_1^{(i)} - y $


In case an $(x, y)$ location falls into multiple GT boxes, it is considered an ambigous example, and the GT box with the smallest area is chosen as the target. This is beacuse  a smaller bounding box is likely to provide a more precise and localized prediction. To solve the problem of dropping GTs FCOS uses multi-level prediction, but this is discussed later on.

In the network output, all targets are positive and due to this reason, all $x$ are mapped to the range $[0,∞]$ with $exp(x)$

#### Loss Function
$$L(\{p_{x,y}\}, \{t_{x,y}\}) = \frac{1}{N_{\text{pos}}} \sum_{x,y} L_{\text{cls}}(p_{x,y}, c^*_{x,y}) + \lambda \frac{1}{N_{\text{pos}}} \sum_{x,y} \mathbf{1}_{\{c^*_{x,y} > 0\}} L_{\text{reg}}(t_{x,y}, t^*_{x,y})$$


Where $L_{\text{cls}}$ is the classification loss term, being focal loss in the original paper 

$$FocalLoss(p,c^*) = -α(1−p)^γlog(p) \text{ if }  c^*=1$$

$$FocalLoss(p,c^*) = -(1-α)p^γlog(1-p) \text{ if } c^*=0$$

$p$ being the predicted class label and $c^*$ is the GT one 

And $L_{\text{reg}}$ is the regression loss, IoU loss as of rightnow

$$\text{IoULoss}(B_{\text{pred}}, B_{\text{gt}}) = 1 - \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}$$

The indicator function $$\mathbf{1}_{\{c^*_{x,y} > 0\}}$$​ ensures that the regression loss is only computed for positive samples (i.e., locations that belong to an object).


###  Multi-level Prediction with FPN

FCOS uses a 5 level FPN ${P3, P4, P5, P6\text{ and } P7}$, each of which have different resolutions and are responsible for detecting objects of different sizes. ${P3, P4 \text{ and } P5}$ are derived from the backbone and are followed by a $1x1$ convolution. $P6 \text{ and } P7$ are calculated from $P5, P6$ respectively with convolution having a stride of 2. The aforementioned way, each level has a different downsampling factor ranging from 8 to 128 and each feature level regresses targets ($$l^*, t^*, r^*, b^*$$).

A location is considered a negative sample ($(x, y)$) if:

$$max(l^*, t^*, r^*, b^*) > m_i \text{ or }$$
$$max(l^*, t^*, r^*, b^*) < m_{i-1}$$

where $m_2, m_3, m_4, m_5, m_6 \text{ and } m_7$ are set as 0, 64, 128, 256, 512 and $∞$, respectively. This greatly reduces the problem mentioned in the section above, having a location $(x, y)$ at multiple GT boxes at once, although it is important to emphasize that in case this does happen, we still take the box with the smallest area.

### Center-ness for FCOS

Even with multi-level prediction using FPN, FCOS can still produce low-quality bounding boxes, especially at locations far from the center of the object. To adress this issue, the authors have proposed a novel so called center-ness branch, which runs in paralell with the classification branch.

![Alt text](/docs/assets/images/fcos_architecture.png)

The center-ness score is designed to measure the distance of a location from the center of an object. Locations near the edges of the bounding box are given lower scores, while those near the center are given higher scores, calculated with the formula below:

$$\text{centerness}^* = \sqrt{\frac{\min(l^*, r^*)}{\max(l^*, r^*)} \times \frac{\min(t^*, b^*)}{\max(t^*, b^*)}}
$$

$sqrt$ is applied to slow down the decay of center-ness. As the center-ness ranges from 0 to 1, it is trained with binary cross entropy loss, and this is added to the aforementioned loss. It is also important to note, that during inference time, the final score for ranking boxes is computed by multiplying the predicted center-ness with the corresponding classification score, thus lower quality boxes will be ranked lower and filtered out by NMS.