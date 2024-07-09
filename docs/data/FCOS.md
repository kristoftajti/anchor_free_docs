# FCOS paper review
### official link - https://arxiv.org/pdf/1904.01355
## Downsides of anchor-based detectors

- Difficulties to deal with objects having large shape variations
- Worsens the generalization ability of the detectors
- For high recall, excessive number of boxes are needed during training in the neck, most of which are negative samples, which creates an imbalance between negative and positive samples.
- Complicated calculations are needed (IoU /w GTs)
- Many avoidable hyper-parameters (IoU threshold)

## Apporach
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

$FocalLoss(p,c^*) = -α(1−p)^γlog(p)$ if $c^*=1$

$FocalLoss(p,c^*) = -(1-α)p^γlog(1-p)$ if $c^*=0$

$p$ being the predicted class label and $c^*$ is the GT one 

And $L_{\text{reg}}$ is the regression loss, IoU loss as of rightnow

$$\text{IoULoss}(B_{\text{pred}}, B_{\text{gt}}) = 1 - \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}$$

The indicator function $$\mathbf{1}_{\{c^*_{x,y} > 0\}}$$​ ensures that the regression loss is only computed for positive samples (i.e., locations that belong to an object).



