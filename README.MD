# Quaternion-Valued Neural Networks and Predictive Coding

## Project idea

In this Project, I want to investigates whether a quaternion-valued neural network (QVNN) can be trained competitively with predictive coding (PC) and how such a model compares to a strong quaternion backpropagation baseline on standard nonlinear function-approximation tasks.  
The project is motivated by two lines of literature: first, quaternion neural networks often outperform comparable real-valued networks while using a more compact representation; second, predictive coding offers a fundamentally different learning principle based on local prediction errors instead of standard end-to-end backpropagation.  

More concretely, the goal is to extend the benchmark setting of Bill, Cox, and Champagne by replacing quaternion backpropagation with a predictive-coding-based training procedure and comparing both approaches under the same architecture and benchmark functions. 

## Quaternions

A quaternion is a four-dimensional extension of the complex numbers,
$$
q = a + b\mathbf{i} + c\mathbf{j} + d\mathbf{k},
$$  
where a, b, c and d are real numbers and i, j and k are symbols that can be interpreted as unit vectors pointing along the three spatial axes. In practice, if any of a, b, c or d is zero, the corresponding term is omitted. If all of a, b, c and d are zero, the quaternion is the zero quaternion and is denoted by 0. If any of b, c or d equals one, the corresponding term is written as i, j or k respectively.  
A multiplicative group structure called the Hamilton product can be defined on quaternions using juxtaposition.  
* The scalar quaternion 1 is the identity element.
* The scalar quaternions commute with all other quaternions; that is, aq = qa for every quaternion q and every scalar quaternion a, which in algebraic terms means that the field of scalar quaternions is the centre of the quaternion algebra.
* The product is first defined for the basis elements and then extended to all quaternions using the distributive and centre properties of scalar quaternions. The Hamilton product is not commutative, but it is associative; therefore, the quaternions form an associative algebra over the real numbers.
* Additionally, every non-zero quaternion has an inverse with respect to the Hamilton product: 
$$ 
( a + b i + c j + d k ) − 1 = 1 a 2 + b 2 + c 2 + d 2 ( a − b i − c j − d k ) . {\displaystyle (a+b\,\mathbf {i} +c\,\mathbf {j} +d\,\mathbf {k} )^{-1}={\frac {1}{a^{2}+b^{2}+c^{2}+d^{2}}}\,(a-b\,\mathbf {i} -c\,\mathbf {j} -d\,\mathbf {k} ).} 
$$

Thus the quaternions form a division algebra.   
Importantly, quaternions are not a field because quaternion multiplication  is not commutative i.e. in general $q_1 \otimes q_2 \neq q_2 \otimes q_1$. This algebraic structure is the key reason why QVNNs are harder to train than ordinary real-valued networks.  

### QVNN
In a QVNN, inputs, weights, biases, and hidden activations are quaternion-valued rather than real-valued. This is advantageous when the data naturally contains four correlated components, because these can be represented as a single structured object instead of four loosely coupled scalars. Bill et al. explicitly cast four real inputs ($x_1,x_2,x_3,x_4$) into one quaternion $q = x_1 + x_2\mathbf{i} + x_3\mathbf    {j} + x_4\mathbf{k}$, which is the same approach adopted in the benchmark paper on quaternion backpropagation algorithms. 

Sources report that quaternion networks often achieve greater accuracy and generalisation than comparable real-valued networks. Furthermore, the 2024 loss-surface study argues that tuned quaternion MLPs can exhibit smoother loss surfaces than real-valued MLPs, which could facilitate optimisation.

###  Challenges

There are two main difficulties.

Firstly, as mentioned before, quaternion multiplication is non-commutative. This means that seemingly minor variations in gradient derivations or the order in which weights are updated can result in genuinely different learning algorithms and empirical behaviour. Bill et al. explicitly state that minor implementation differences matter in the quaternion case due to this non-commutativity. 

Secondly, there is no single, universally accepted 'default' backpropagation rule for quaternion networks. The literature contains multiple alternatives: Arena's split/pseudo-gradient method; Nitta's closely related variant; Buchholz and Sommer's Clifford Multilayer Perceptron (CMLP) update rules; and the later $G\mathbb{H}\mathbb{R}$-calculus approach, which uses a fuller quaternion derivative formalism. Additionally, split or component-wise activation functions are generally employed since directly translating standard analytic activation functions into the complex/quaternion setting encounters the classical Liouville issue. The solution is to apply a real activation function to each quaternion component individually. 

### Quaternion backpropagation Baseline

I propose as a baseline for this project to use the **CMLP backpropagation**. In the 2023 comparison paper, CMLP achieved the best average test performance on three of the four benchmark functions and was not meaningfully beaten by $G\mathbb{H}\mathbb{R}$ on the remaining one. The same paper concludes that $G\mathbb{H}\mathbb{R}$ did **not** deliver a significant overall improvement in either accuracy or runtime over CMLP. Beyond the empirical results, the authors argue that CMLP has the strongest theoretical foundation and highest generality: it is a universal approximator for non-degenerate Clifford algebras and reduces exactly to ordinary backpropagation in the real-valued case. For that reason, they explicitly recommend CMLP as the basis for future hypercomplex regression work. 

## Predictive coding

Predictive coding is interesting because, rather than using standard end-to-end gradient propagation, it employs a scheme based on top-down predictions, bottom-up prediction errors, and an alternating cycle of fast inference and slow learning. In predictive coding networks, latent states are iteratively updated to minimise prediction error and weights are updated using local quantities. The provided predictive coding (PC) source code emphasises that this produces characteristic two-timescale dynamics and weight updates that are local in a strong sense.  

### Project- Rationale

* predictive coding (PC) offers a conceptually different optimisation principle, making it scientifically interesting to compare it with quaternion backpropagation rather than simply adding another quaternion gradient variant. 

* PC weight updates are local and compatible with Hebbian-like interpretations, whereas ordinary backpropagation relies on a globally coordinated forward/backward pass. The source also highlights locality, decentralisation and parallelism as potential computational advantages.  

* PC supports variable-length inference, whereby easy samples can settle quickly, while harder samples can use more inference steps. This is relevant because the project is interested not only in the final error, but also in the optimisation behaviour, cost and speed. 

Therefore, the project's aim is not to assume that predictive coding must outperform quaternion backpropagation. Rather, the aim is to test whether PC is a viable and interesting training alternative in a quaternion setting where gradient definitions are already unusually delicate.

### Challenges
#TODO

## Methodology

### Archidecture

The benchmark paper uses a fixed **1 → 5 → 5 → 5 → 1** multilayer perceptron. The authors justify this choice by stating that it provides enough neurons per layer to learn the nonlinearities of the chosen test functions, while also being deep enough to compare runtime and scalability of different training methods. They also note that this setup was sufficient to study the first deep implementation of the GHR rules. 

That same architecture is also a sensible choice for predictive coding. PC introduces an iterative inference loop over latent states, so the cost per training example is no longer just one forward and one backward pass. Using a compact but nontrivial network therefore keeps the project manageable: the model is deep enough that predictive coding is meaningful, but still small enough that repeated latent inference remains computationally feasible and easy to debug. This is an inference from combining Bill et al.’s architecture choice with the predictive-coding algorithm, where every hidden layer participates in iterative latent-state updates before the weight step.  

### Test Setup

The experimental design will follow the benchmark paper as closely as possible.

Bill et al. use four standard nonlinear optimization test functions:

* Ackley
$$
f(\mathbf{x}) = -a \exp\left(-b \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}\right) - \exp\left(\frac{1}{d} \sum_{i=1}^{d} \cos(c x_i)\right) + a + \exp(1)
$$ 
* Griewank
$$
f(\mathbf{x}) = \sum_{i=1}^{d} \frac{x_i^2}{4000} - \prod_{i=1}^{d} \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1
$$
* Schwefel
$$
f(\mathbf{x}) = 418.9829 d - \sum_{i=1}^{d} x_i \sin\left(\sqrt{|x_i|}\right)
$$ 
* Rastrigin
$$
f(\mathbf{x}) = 10d + \sum_{i=1}^{d} \left[ x_i^2 - 10 \cos(2\pi x_i) \right]
$$ 

Each function is evaluated in a setting with four real input dimensions, which are then encoded into one quaternion input. The benchmark datasets are generated using **30,000 samples per function** obtained by **Optimal Latin Hypercube Sampling**, followed by a **70/15/15** split into training, validation, and test data. The inputs are standardized using the training-set statistics.  

This project will therefore compare models on exactly these functions and, as far as feasible, under the same data-generation and preprocessing scheme.

### Model comparisons

The core comparison is:

1. **QVNN trained with CMLP backpropagation**
2. **QVNN trained with predictive coding**

Additionally, including one classical real-valued baseline is useful.

A real-valued baseline with exactly the same visible layer counts is not a fair capacity comparison, because a single quaternion unit already contains four real components. The literature handles this in two different ways. The loss-surface paper increases the real-valued hidden-width search range specifically because a quaternion can encapsulate more information in one unit, while the predictive-control paper constructs an RVNN whose total number of connection weights matches the QVNN. Both are defensible, but they answer slightly different questions.  

So I decided to use the follwoing to use the parameter-matched RVNN as a baseline to jude of a QVNN heven brings advantages, or if is just cool mathematical shananigans.

### Evaluation criteria

The original benchmark compares models using test error, runtime, and training characteristics. That should remain the backbone of the evaluation. 

For this project, I would additionaly  evaluate:

* **Predictive performance:** MAPE and (R^2), as in Bill et al.; optionally also MSE or RMSE for readability. 
* **Runtime:** wall-clock training time, and ideally also average time per epoch or per update. Bill et al. explicitly use runtime as a comparison metric. 
* **Model size:** number of quaternion parameters, equivalent number of real scalar parameters, and memory footprint.
* **Optimization behavior:** training and validation curves, stability across seeds, and convergence behavior.
* **PC-specific cost:** number of inference steps required, since predictive coding introduces an inner inference loop before learning. 

To have statisically significant results, repeated runs with fixed random seeds and paired comparisons are the goal, escpecially since the benchmark literature also uses repeated runs and statistical testing rather than a single lucky run.  

## Research question

A compact formulation of the project question is:

> Can predictive coding train a small quaternion-valued neural network competitively compared with the CMLP backpropagation algorithm on standard nonlinear function-approximation benchmarks?

> Does such an approach even yield significant advantages over a properly scaled, but not escpecially fancy NN?