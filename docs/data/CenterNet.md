<!-- Button to go back to the main page -->
<div style="margin-top: 20px;">
  <a href=".../index.md" style="text-decoration: none;">
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

# CenterNet paper review
### official link - [https://arxiv.org/pdf/1904.08189](https://arxiv.org/pdf/1904.08189)

## Object Detection as Keypoint Triplets

Each object is represented by a center keypoint, which we embed as a heatmap and a pair of corners (top-left and bottom-right). After predicting, the top-k bounding boxes are selected, which then are filtered. The procedure of filtering as follows:

 - Remap center keypoint to the original image, with the help of their offsets
 - Define a central region for each bbox and check whether it contains the aforementioned center keypoint
 - If it is inside a central region and the classes are the same, we preserve it, and the score of the bbox will be the averaged of the three point(top-left, center, bottom-right)

Central region in the bbox affects detection results. For example, smaller central regions lead to a low recall rate for small bounding boxes, while larger central regions lead to a low precision for large bounding boxes. A scale-aware central region is proposed, which adresses the issue mentioned before

$$
\begin{align*}
c_{tlx} &= \frac{(n + 1) tlx + (n - 1) brx}{2n} \\
c_{tly} &= \frac{(n + 1) tly + (n - 1) bry}{2n} \\
c_{brx} &= \frac{(n - 1) tlx + (n + 1) brx}{2n} \\
c_{bry} &= \frac{(n - 1) tly + (n + 1) bry}{2n}
\end{align*}
$$

Where $tl_x \text{ and } tl_y$ are the top-left coordinates, $br_x \text{ and } br_y$ are the bottom-right coordinates and $ctl_x, ctl_y, cbr_x \text { and } cbr_y$ are the central regions coordinates and $n$ is an odd number that is the scale of the central region. In CenterNet $n$ is set to be 3 and 5 for scales of bboxes less and grater than 150.

## CornerNet - Corner Pooling
As CenterNets cascade corner pooling relies heavily on CornerNets corner pooling, I feel like it is essential to discuss that first. Corner pooling ([CornerNet paper - section 3.4](https://arxiv.org/pdf/1808.01244)), for instance to determine if feature vector **f** is a top-left corner at $(i,j)$, maxpools right to $(i,j)$ and downwards from it, then sums it. 

$$
\begin{align*}
h_1(i, j) &= \max_{k \ge j} f(i, k) \\
h_2(i, j) &= \max_{k \ge i} g(k, j) \\
h(i, j) &= h_1(i, j) + h_2(i, j)
\end{align*}
$$

The same is done for the bottom-right corner, but with upward and leftward pooling and addition. This kind of pooling help  accurately localizing corners, especially in cases where local evidence is insufficient (meaning the objects edges are hard to determine).