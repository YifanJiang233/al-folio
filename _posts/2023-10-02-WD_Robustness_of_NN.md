---
layout: distill
title: Wasserstein Distributional Robustness of Neural Networks
date: 2023-10-02
description: A paper accepted for NeurIPS 2023.
tags: ["distributionally robust optimization", "adversarial attack"]
categories: ["expositions"]
bibliography: WDRobustness.bib
---

<div class="box" text="TLDR:">
Using the Wasserstein distributionally robust optimization (W-DRO) framework, we derive the first order adversarial attack, link the adversarial accuracy to the adversarial loss, and investigate the out-of-sample performance for neural networks under <em>distributional</em> threat models.
</div>

This post serves as an exposition of our recent work. Please refer to [arXiv](https://arxiv.org/abs/2306.09844) for a more detailed discussion and to [GitHub](https://github.com/JanObloj/W-DRO-Adversarial-Methods) for source codes.

<!-- **TLDR:**  -->
<!-- Using Wasserstein distributionally robust optimization (W-DRO) framework, we study the robustness of neural networks under *distributional* threat models. -->
<!-- First order adversarial attacks are derived by leveraging W-DRO sensitivity results, and FGSM is included as a special case. -->
<!-- We further link the adversarial accuracy to the adversarial loss, and investigate the out-of-sample performance. -->
<!-- Using the Wasserstein distributionally robust optimization (W-DRO) framework, we derive the first order adversarial attack, link the adversarial accuracy to the adversarial loss, and investigate the out-of-sample performance for neural networks under *distributional* threat models. -->

## Background

Deep neural networks have achieved great success in image classification tasks.
We denote the feature of an image by $x$ and its class by $y$.
A neural network $f_{\theta}$ takes $x$ as input and outputs the likelihood of each class the input could be.
We denote $P$ as the data distribution.
Then, we can write training of a neural network $f_{\theta}$ as finding the minimizer $\theta^{\star}$

$$\inf_{\theta} \E_{P}[J_{\theta}(x,y)] \leadsto  \theta^{\star},$$

where $$J_{\theta}(x,y)=L(f_{\theta}(x),y)$$ for some loss function $$L$$.

In the seminal paper <d-cite key="GSS15"></d-cite>, however, it pointed out that a well-trained neural network is vulnerable under the adversarial attack, i.e., a delicately designed perturbation on the input image.
An example is illustrated as follows: by adding an imperceptible noise to the panda image, a neural network changes its prediction from 57.7% "panda" to 99.9% "gibbon".
The middle image is normalized in order to be visible to human eyes.
We stress that the perturbation here is carefully chosen as the gradient sign direction $\mathrm{sgn}(\nabla_x J)$.

<figure>
 {% include distill_img.liquid path="assets/img/WDRobustness/panda2.png" zoomable=true%}
<figcaption class="caption">A demonstration of adversarial attack via Fast Gradient Sign Method (FGSM) <d-cite key="GSS15"></d-cite>.</figcaption>
</figure>

Classical literature on adversarial attacks, for example <d-cite key="MMS+18"></d-cite>, focus on the _pointwise_ threat model, where a uniform budget $\delta$ of perturbation is given for each image.
To generate adversarial images, we to some extent reverse the training process by maximizing the loss function over input data

$$
\E_{P}\Bigl[\sup_{\|x-x'\|\leq \delta}J_{\theta}(x',y)\Bigr]\leadsto x^{\star}.
$$

Here, $\\|\cdot\\|$ is a norm on the feature space, which could be $l_2$, $l_{\infty}$, etc.

A key observation here is that,

$$
\E_{P}\Bigl[\sup_{\|x-x'\|_s\leq \delta}J_{\theta}(x',y)\Bigr]=\sup_{\mathcal{W}_{\infty}(P,Q)\leq \delta}\E_{Q}\bigl[J_{\theta}(x,y)\bigr],
$$

where $\mathcal{W}_{\infty}$ is the $\infty$-Wasserstein distance induced by

$$
    \label{eqn-d}
    d((x_{1},y_{1}),(x_{2},y_{2}))=\|x_{1}-x_{2}\|+\infty\mathbf{1}_{\{y_{1}\neq y_{2}\}}.
$$

In this sense, finding the adversarial images is equivalent to finding the adversarial image distribution under the $\infty$-Wasserstein distance.
This motivates us to investigate the _distributional_ threat model and its associated adversarial attack, given by

$$
\sup_{\mathcal{W}_{2}(P,Q)\leq \delta}\E_{Q}\bigl[J_{\theta}(x,y)\bigr].
$$

We quickly remark one main feature of _distributional_ threat models.
In contrast to the _pointwise_ threat, the attacker has a greater flexibility and can perturb images close to the decision boundary only slightly while spending more of the attack budget on images farther away from the boundary.
This makes the _distributional_ adversarial attack more involved.

Such a W-DRO framework, while compelling theoretically, is often numerically intractable.
We leverage the W-DRO sensitivity results <d-cite key="BDOW21"></d-cite> to carry out attacks under a _distributional_ threat model.
We further give asymptotic certified bounds on the adversarial accuracy which are fast to compute and of first-order accuracy.
Finally, we utilize concentration inequality of the empirical measure and derive the out-of-sample performance of neural networks under _distribution_ threat models.

## Setup

An image is interpreted as a tuple $(x,y)$ where the feature vector $x\in \mathcal{X}=[0,1]^{n}$ encodes the graphic information and $y\in\mathcal{Y}=\\{1,\dots,m\\}$ denotes the class, or tag, of the image.
A distribution of labelled images corresponds to a probability measure $P$ on $\mathcal{X}\times\mathcal{Y}$.
Under this setting $P$ could be the empirical measure on a given dataset or an inaccessible "true" distribution on the extended image space.

Let $(p,q)$ and $(r,s)$ be two pairs of conjugate indices with $1/p+1/q=1/r+1/s=1.$
As mentioned above, we equip $\mathcal{X}$ with $l_s$ norm and consider the Wasserstein distance $\mathcal{W}_p$ generated by $d$.

We write the neural network as a map $f_{\theta}:\mathcal{X}\to \mathbb{R}^{m}$.
We denote $S$ the set of images equipped with their labels generated by $f_\theta$, i.e.,

$$
    S=\Bigl\{(x,y)\in  \mathcal{X}\times\mathcal{Y}: \arg\max_{1\leq i\leq m} f_{\theta}(x)_{i}=\{y\}\Bigr\}.
$$

We denote the clean accuracy by $$A=\E_P[\mathbf{1}_S]$$ and Wasserstein distributional adversarial accuracy by

$$A_{\delta}:=\inf_{\mathcal{W}_p(P,Q)\leq \delta}\E_Q[\mathbf{1}_S].$$

## Main Results

We always assume the following two assumptions hold.

<div class="asmp">
     We assume the map
    \(
        (x,y)\mapsto J_{\theta}(x,y)
    \)
    is \(\mathsf{L}\)-Lipschitz under \(d\), i.e., \[|J_{\theta}(x_1,y_1)-J_{\theta}(x_2,y_2)|\leq \mathsf{L} d((x_1,y_1),(x_2,y_2)).\]
</div>
<div class="asmp">
     We assume that for any \(Q\in B_{\delta}(P)\)
     <br>
       (a) $0<Q(S)<1.$
       <br>
        (b) \(
              \mathcal{W}_{p}(Q(\cdot|S),P(\cdot|S))+\mathcal{W}_{p}(Q(\cdot|S^{c}),P(\cdot|S^{c}))= o(\delta),
              \)
              where \(S^{c}=(\mathcal{X}\times\mathcal{Y})\setminus S\) and the conditional distribution is given by $Q(E|S)=Q(E\cap S)/Q(S)$.
</div>

### First Order Adversarial Attack

To propose WD adversarial attacks, we first introduce two important ingredients : sensitivity of W-DRO and rectified DLR loss.

We write

$$
V(\delta)=\sup_{\mathcal{W}_{p}(P,Q)\leq \delta}\E_{Q}\bigl[J_{\theta}(x,y)\bigr].
$$

The following theorem is adapted from <d-cite key="BDOW21"></d-cite>.

<div class="thm">
Under the above assumptions, we have the following first order approximations hold:
<br>
(a) $V(\delta)=V(0)+\delta\Upsilon+o(\delta),$
              where
              \begin{equation*}
                  \Upsilon=\Bigl(\E_{P}\|\nabla_{x}J_{\theta}(x,y)\|_{r}^{q}\Bigr)^{1/q}.
              \end{equation*}
(b)            \(V(\delta)=\E_{Q_{\delta}}[J_{\theta}(x,y)]+o(\delta),\)
              where
              \begin{equation*}
                  Q_{\delta}=\Bigl[(x,y)\mapsto \bigl(x+\delta h(\nabla_{x}J_{\theta}(x,y))\|\Upsilon^{-1}\nabla_{x}J_{\theta}(x,y)\|_{r}^{q-1},y\bigr)\Bigr]_{\#}P,
              \end{equation*}
              and \(h\) is uniquely determined by \(\langle h(x),x\rangle=\|x\|_{r}\).
</div>

The second result essentially says to maxize the loss, the perturbation given by

$$
 x\to x+\delta h(\nabla_{x}J_{\theta}(x,y))\|\Upsilon^{-1}\nabla_{x}J_{\theta}(x,y)\|_{r}^{q-1}
$$

is first-order optimal.
Particularly, if we take $p=\infty$ and $q=\infty$, we retrieve Fast Gradient Descent Method (FGSM) mentioned above for _pintwise_ threat models.

Under _pointwise_ threat models, taking loss function $L$ as a combination of Cross Entropy (CE) loss and Difference of Logits Ratio (DLR) loss has been widely shown as an effective empirical attack, see <d-cite key="CH20"></d-cite> for details.
The DLR loss is given by

$$
   \mathrm{DLR}(z,y)=\left\{\begin{aligned}
       -\frac{z_{y}-z_{(2)}}{z_{(1)}-z_{(3)}}, & \quad\text{if } z_{y}=z_{(1)}, \\
       -\frac{z_{y}-z_{(1)}}{z_{(1)}-z_{(3)}}, & \quad\text{else,}
   \end{aligned}\right.
$$

where we write $$z=(z_{1},\dots,z_{m})=f_{\theta}(x)$$ for the output of a neural network, and $$z_{(1)}\geq\dots\geq z_{(m)}$$ are the order statistics of $$z$$.
However, under _distributional_ threat models, intuitively, an effective attack should perturb more aggressively images classified far from the decision boundary and leave the misclassified images unchanged.
Consequently, neither CE loss nor DLR loss are appropriate.
Instead, we propose to use Rectified DLR (ReDLR) loss as a candidate to employ _distributional_ adversarial attacks, which is simply given by

$$
\mathop{\mathrm{ReDLR}}=-(\mathop{\mathrm{DLR}})^{-}.
$$

In the figure below, we compare the adversarial accuracy of robust networks on RobustBench <d-cite key="CAS+21"></d-cite> against pointwise threat models and distributional threat models.
We notice a significant drop of the adversarial accuracy even for those neural networks robust against pointwise threat models.

<figure>
 {% include distill_img.liquid path="assets/img/WDRobustness/acc_shortfall.jpg" zoomable=true%}
<figcaption class="caption"> 
Shortfall of WD-adversarial accuracy on CIFAR-10 with different metrics $l_{\infty}$ (left) and $l_{2}$ (right). We testify our proposed attack on all neural networks from RobustBench <d-cite key="CAS+21"></d-cite>. 
</figcaption>
</figure>

### Asymptotically Certified Bound

We write $$\mathcal{R}_{\delta}:=A_{\delta}/A$$ as a metric of robustness, and the adversarial loss condition on the misclassified images as

$$
    W(\delta)=\sup_{Q\in B_{\delta}(P)}\E_{Q}[J_{\theta}(x,y)|S^{c}].
$$

We note that an upper bound on $\mathcal{R}_\delta$ is given by any adversarial attack.
In particular,

$$\mathcal{R}_\delta \leq \mathcal{R}^u_\delta:= Q_\delta(S)/A.$$

<div class="thm">
    Under the above assumptions, we have an asymptotic lower bound as $\delta\to 0$

$$
        \mathcal{R}_\delta\geq \frac{W(0)-V(\delta)}{W(0)-V(0)} +o(\delta)=\widetilde{\mathcal{R}}_\delta^l+o(\delta)=\overline{\mathcal{R}}^l_\delta+o(\delta),
$$

    where the first order approximations are given by

$$
        \widetilde{\mathcal{R}}_\delta^l=\frac{W(0)-\E_{Q_{\delta}}[J_{\theta}(x,y)]}{W(0)-V(0)}\quad \text{and} \quad \overline{\mathcal{R}}_{\delta}^l=\frac{W(0)-V(0)-\delta\Upsilon}{W(0)-V(0)}.
$$

</div>

Consequently, $$\mathcal{R}^l_{\delta}=\min\{\widetilde{\mathcal{R}}_\delta^l, \overline{\mathcal{R}}_{\delta}^l\}$$ allows us to estimate the model robustness without performing any sophisticated adversarial attack.
We plot our proposed bounds against the reference robust metric on CIFAR-100 and ImageNet datasets as below.
For CIFAR-10 dataset, we refer to our paper.
Notably, the bounds provided here is order of magnitude faster to compute than the reference value $\mathcal{R}_{\delta}$ by using AutoAttack <d-cite key="CH20"></d-cite>.

<figure>
    {% include distill_img.liquid path="assets/img/WDRobustness/cifar100_blog.jpg" path2="assets/img/WDRobustness/imagenet_blog.jpg" zoomable=true%}
        <figcaption class="caption"> 
    $\mathcal{R}^{u}$ & $\mathcal{R}^{l}$ versus $\mathcal{R}$ on CIFAR-100 (top) and ImageNet (bottom). 
    </figcaption>
</figure>

### Out-of-Sample Performance

Our results on distributionally adversarial robustness translate into bounds for performance of the trained DNN on unseen data.
We rely on the concentration inequality of empirical measures.
Let us fix $1<p<n/2$.
We assume the training set is i.i.d sampled from the true distribution $P$ with size $N$.
Then the empirical measure of the training set $\widehat{P}$ is a random measure, and satisfies

$$
 \mathbb{P}(\mathcal{W}_p(\widehat{P},P)\geq \varepsilon)\leq K \exp(-KN\varepsilon^n),
$$

where $K$ is a constant depending on $p$ and $n$.
Thank to W-DRO framework, such an estimate naturally yields the out-of-performance of neural networks under _distributional_ threat models.

<div class="thm"> 
    Under above Assumptions, with probability at least \(1-K \exp(-KN\varepsilon^{n})\) we have
    \begin{equation*}
        V(\delta)\leq \widehat{V}(\delta) + \varepsilon\sup_{Q\in B_{\delta}^{\star}(\widehat{P})}\Bigl(\E_{Q}\|\nabla_{x}J_{\theta}(x,y)\|_{s}^{q}\Bigr)^{1/q} + o(\varepsilon)\leq \widehat{V}(\delta)+ L\varepsilon
    \end{equation*}
    where   \(B_{\delta}^{\star}(\widehat{P})=\arg\max_{Q\in B_{\delta}(\widehat{P})}\E_{Q}[J_{\theta}(x,y)]\) and constant \(K\) only depends on $p$ and $n$.
</div>

We remark that the above results are easily extended to the out-of-sample performance on the test set, via the triangle inequality.

## Conclusion

Our work contributes to the understanding of robustness of DNN classifiers.
It also offers a wider viewpoint on the question of robustness and naturally links the questions of adversarial attacks, out-of-sample performance, out-of-distribution performance and Knightian uncertainty.
By introducing a first-order Adversarial Attack (AA) algorithm, we not only encompass existing methods like FGSM and PGD but also offer new insights into distributional threat models.
We believe our research opens up many avenues for future work, especially training robust neural networks under distributional threat models.
