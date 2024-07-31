# ABIET
Attention-Based Importance Estimator Tool (ABIET): An Explainable Transformer for Identifying Functional Groups in Biological Active Molecules

<p align="justify"> This study focuses on the comparative analysis of various strategies for extracting token importance from attention weights in a masked-language model Transformer implementation, using canonical and stereochemical SMILES notation. We demonstrate that tokens focusing more prominent attention are associated with more critical molecular regions, such as functional groups (FGs).
Hence, we identified a valuable strategy for obtaining token importance, demonstrating that the attention directed toward FGs is statistically higher than that toward the remaining atoms of the molecule. The comparative analysis allowed us to conclude that the initial layers of the Transformer and considering the bidirectional interactions between the tokens are more informative about the individual relevance of each token. </p>


## Model Dynamics Architecture
<p align="center"><img src="/images/figure_1.jpg" width="90%" height="90%"/></p>

## Data Availability
**data_vegfr2.smi:** SMILES notation of 5,314 active compounds towards VEGFR2 target.
 

## Requirements:
- Python 3.8.12
- Tensorflow 2.3.0
- Numpy 
- Pandas
- Scikit-learn
- Itertools
- Matplotlib
- Seaborn
- Bunch
- tqdm
- rdkit 2021.03.4

## Usage 
Run main.py file to implement the described dynamics. It is possible to adjust the token extraction strategy (extraction layer, vocabulary, computation strategy, head selection) in the  argument_parser.py file.

### Running the best configuration
```
python main.py 
```
