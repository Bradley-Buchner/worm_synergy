# SynergyGT: a Graph Transformer for Predicting Gene Synergy in *C. elegans* Aging

This repository contains the scripts and notebooks for the model known as SynergyGT, which serves as a computational tool for the Levine Lab and its research on the genetics of aging in the roundworm *C. elegans*. Although individual gene perturbations associated with aging and longevity have been well-studied, understanding/solving aging requires treating it as a complex, emergent phenotype driven by nonlinear genetic interactions. Synergistic gene interactions offer a view into the complexity of aging, and by predicting/learning to identify synergy in the genetic interaction network with deep learning we can better understand how aging/longevity emerges. SynergyGT does this by combining knowledge of gene-aging associations and a network of mechanistic gene-gene interactions to learn features of known synergistic gene pairs that distinguish them from those that are not. With a design inspired by biology and modern LLMs, SynergyGT can predict synergistic interactions at a level significantly better than baseline. As a result, the model can be used to characterize the likelihood of synergy between any pair of genes, which could lead to the discovery of novel synergistic interactions and a better understanding of the genetic landscape of aging. 

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
* Pair subgraphs:
  * node set: union of each perturbed gene's/member's one-hop neighborhood
  * all edges filled in from the original graph/network
* Node-level features/attributes:
  * in-degree and out-degree
  * proximity (hop distance) from each perturbed gene
  * proximity (hop distance) to the nearest aging-associated gene (zero if associated with aging itself)
  * perturbation status and type: if perturbed, what kind of pertubation? => ("knockdown", "knockout", or "overexpression")

### Model
Subgraph encoder => classification head
* Subgraph encoder:
  * graph transformer:
    * treats nodes as tokens, encodes graph structure into the attention mechanism, learns a summary representation of the subgraph using a [CLS] token.
    * Learns 8-dim embeddings for each type of token attribute (degree, aging proximity, etc.), which are summed together to form a single 8-dim representation for each token
    * Learns how to aggregate all token representations in a subgraph
    * biases attention between nodes to reflect causality (by enforcing attention to flow only from ancestors to descendents), proximity (with a learnable bias based on hop distance), and biological interaction semantics (by learning a unique value projection for each interaction type)
* Classification head:
  * MLP with 1 hidden layer
  * output is 3-dim: probabilities of antagonistic, additive, and synergistic interactions  

### Learning Objective
Minimize the KL Divergence between observed and predicted relative interaction type frequencies, which can be thought of as soft classification labels. 

### Output
Predicted relative interaction type frequencies for any pair of gene perturbations. 

### 


TO-DO:
* make a visualization of the model
* describe how to interact with the model


Link to SynergyGT guide: []

Link to model Demo: []

Link to exploratory tool/notebook: []

