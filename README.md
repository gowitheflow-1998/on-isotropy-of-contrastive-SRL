# On Isotropy, Contextualization and Learning Dynamics of Contrastive-based Sentence Representation Learning
We will release the code shortly.

> _"On Isotropy, Contextualization and Learning Dynamics of Contrastive-based Sentence Representation Learning"_  
> To appear in ACL 2023



1. `metrics.py` defines our implementations on computing 1) self-similarity 2) intra-sentence/document similarity 3)anisotropy estimation 4) rouge dimension 5) embedding norm 

3. run `analysis_pipeline.py` with these defined metrics to analyze your model at once. All results will be saved in a `analysis_output.txt`.

Run with customized arguments. For instance, run analysis on all-minilm-l6-v2, starting from layer 5 to the last layer:
`python analysis_pipeline.py --model_name sentence-transformers/all-MiniLM-L6-v2 --analyze_layer_start 5`

3. we provide the corresponding noteboook scripts `analysis_pipeline.ipynb` to play with these defined metrics step-by-step in to analyze all-mpnet-base-v2 as an example. Change MODEL_NAME in the notebook to analyze your model.

4. In `fun_example.ipynb` you can see some intuitive and fun examples of our paper's findings:

"NLP" in "NLP is about" and "is" in "NLP" is about are similar             (high intra-sentence similarity) \n
"NLP" in "NLP is about" and "NLP" in "What is NLP?" are similar            (high self-similarity for semantic tokens) \n
"is" in "NLP is good" and "is" in "The album is fire" are not even close   (low self-similarity for function tokens) \n

4. We're updating `train.py` to facilitate ablation analysis.