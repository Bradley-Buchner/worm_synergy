# SynergyGT: a Graph Transformer for Predicting Gene Synergy in *C. elegans* Aging

This repository contains the scripts and notebooks for SynergyGT, a model that serves as a computational tool for the Levine Lab and its research on the genetics of aging in the roundworm *C. elegans*. Although individual genes associated with aging and longevity have been well-studied, aging is a complex, emergent phenotype driven by a combination of nonlinear genetic interactions. Synergistic gene interactions offer a view into the complexity of aging, and by using deep learning to reveal such relationships in the genetic interaction network we can better understand how aging emerges. SynergyGT does this by combining knowledge of gene-aging associations and a network of mechanistic gene-gene interactions to learn features of known synergistic gene pairs that distinguish them from those that are not. With a design inspired by biology and modern LLMs, SynergyGT can predict synergistic interactions at a level significantly better than naive baselines. In practice, the model can be used to characterize the likelihood of synergy between any pair of genes, which could lead to the discovery of novel synergistic interactions and a better understanding of the genetic landscape of aging. 
<p align="center">
  <img src="https://github.com/Bradley-Buchner/worm_synergy/blob/06b4c0e873074db633e62bfa9f4ade581686c01f/figures/synergy_gt_schema_readme.jpg" width="800">
</p>

## Using SynergyGT
To use SynergyGT, there are two jupyter notebooks in the `/notebooks` directory that can easily be run in Google Colab. 
* ("model_demo.ipynb")[https://github.com/Bradley-Buchner/worm_synergy/blob/6eace04e37b57b9897bfdd2ccf93b072c59851e8/notebooks/model_demo.ipynb], walks you through the process of building and training a SynergyGT model and evaluating its performance.
* "model_exploration.ipynb", lets you interact with a trained SynergyGT model and test it on any gene or gene pair of interest. Click the "Run in Colab" button at the top of these notebooks to run them yourself. 

## Model Schema
This section outlines the conceptual blueprint of the model, specifying the information it consumes, how that information is represented, what the model is trained to predict, and the assumptions under which its learns.

### Input
The model integrates two primary data sources:

**1. Genetic interaction network (from WormBase)**

A directed, heterogeneous interaction network representing known molecular and genetic relationships in C. elegans.
   * 11,493 nodes (genes/proteins) and 90,364 edges
   * 3 types of edges (interactions): genetic, physical, and regulatory
   * Edges are directed to reflect causal relationships where applicable; non-causal interactions are represented by bidirectional edges.

**2. Double mutant lifespan assays (from SynergyAge)**

A curated collection of lifespan measurements for combinatorial genetic interventions in *C. elegans*
* 1,458 double mutant experiments, 801 unique double mutants (i.e., gene perturbation pairs)
* Each experiment is categorized as resulting in an antagonistic, additive, or synergistic effect on lifespan

Together, these inputs provide both the network context in which genes interact and empirical measurements of how pairs of genetic perturbations affect lifespan.


### Representation
Each gene pair is represented as a localized subgraph from the global interaction network.

**Pair subgraphs**
* **Node set:** The union of the one-hop neighborhoods of both perturbed genes
* **Edge set:** All edges induced from the original interaction network, preserving directionality and interaction type

**Node-level features**

Each node within a subgraph is annotated with biologically and topologically motivated attributes:
* In-degree and out-degree
* Proximity (hop distance) to each perturbed gene
* Proximity (hop distance) to the nearest aging-associated gene (zero for aging-associated genes)
* Perturbation status and perturbation type (knockdown, knockout, or overexpression)

This representation encodes local network structure while also injecting relevant biological context for the prediction task.

### Model
The model maps pair-centered subgraphs to predicted interaction outcomes using a two-stage architecture: a subgraph encoder followed by a classification head.

**Subgraph encoder**

A graph transformer is used to encode each subgraph into a fixed-dimensional vector representation.
* Nodes are treated as tokens (analogous to how LLMs tokenize sentences into words), with graph structure incorporated directly into the attention mechanism.
* Each node attribute (e.g., degree, aging proximity, perturbation status) is embedded into a small fixed-dimensional vector; these embeddings are learned and summed to produce a single representation for each node/token.
* A synthetic [CLS] node added to each subgraph learns to aggregate information from all other nodes to form another small fixed-dimensional summary representation of the subgraph.

The calculation of attention between nodes is intentionally biased to reflect:
* **Causality:** Nodes are enforced to attend only to their descendents in the directed graph.
* **Proximity:** Learnable biases based on hop distance between nodes.
* **Interaction semantics:** Separate value projections are learned for each interaction type (genetic, physical, regulatory).

This design imposes biologically motivated inductive biases while allowing the model to learn how information relevant to aging should flow through local neighborhoods and be aggregated.

**Classification head**

A multilayer perceptron with one hidden layer takes a CLS token's subgraph representation as input and outputs a 3-dimensional probability vector corresponding to antagonistic, additive, and synergistic interaction likelihoods. 

#### Learning Objective
The model is trained to minimize the Kullback–Leibler (KL) divergence between predicted and observed relative interaction-type frequencies for each gene pair. This formulation treats relative interaction-type frequencies as soft classification labels, and is an example of label distribution learning.

*Note: To account for the diverse quantity of experiments recorded for each unique double mutant and, as a result, varying evidence and confidence levels, observed interaction-type counts were smoothed via Bayesian smoothing with a maximally ignorant prior that assumed pseudocounts of 1 for each interaction type (i.e., assumes all types are equally likely).*


### Output
For any pair of gene perturbations, the model outputs predicted relative frequencies for antagonistic, additive, and synergistic lifespan effects. These predictions can be interpreted as a probability distribution over expected genetic interaction effects, and can be used to prioritize candidate gene pairs for experimental validation.
