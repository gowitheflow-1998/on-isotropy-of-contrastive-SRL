# On Isotropy, Contextualization and Learning Dynamics of Contrastive-based Sentence Representation Learning
We will release the code shortly.

> _"On Isotropy, Contextualization and Learning Dynamics of Contrastive-based Sentence Representation Learning"_  
> To appear in ACL 2023



1. `metrics.py` defines our implementations on computing 1) self-similarity 2) intra-sentence/document similarity 3)anisotropy estimation
4) rouge dimension 5) embedding norm 

2. we provide some interactive scripts using these defined metrics in `analysis pipeline.ipynb` to analyze all-mpnet-base-v2 as an example. Change MODEL_NAME to analyze your model.

3. We're updating `analysis pipeline.py` to run all analysis at once on a given model.

4. We're updating `train.py` to facilitate ablation analysis.