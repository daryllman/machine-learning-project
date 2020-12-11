# Machine Learning Project 
Text Analysis with Hidden Markov Model (HMM)

**Content**
1. [Setup](#setup)
2. [Evaluation of Prediction Results](#evaluation-of-prediction-results)
3. [Training](#training)

## Setup
We are not using any externally installed libraries.

## Evaluation of Prediction Results
The results of the predictions are located in the respective language folders given by `dev.p2.out`, `dev.p3.out`, `dev.p4.out`, `dev.p5.out`         
For convenience, we have included the exact commands to run the eval.Result.py script below.     
Simply run the following commands to obtain the evaluation results.
___
### Part 2
Uses `part2_simple_system.py`
#### EN
```
python EvalScript/evalResult.py EN/dev.out EN/dev.p2.out
```
#### CN
```
python EvalScript/evalResult.py CN/dev.out CN/dev.p2.out
```
#### SG
```
python EvalScript/evalResult.py SG/dev.out SG/dev.p2.out
```
___
### Part 3
Uses `part3_viterbi.py`
#### EN
```
python EvalScript/evalResult.py EN/dev.out EN/dev.p3.out
```
#### CN
```
python EvalScript/evalResult.py CN/dev.out CN/dev.p3.out
```
#### SG
```
python EvalScript/evalResult.py SG/dev.out SG/dev.p3.out
```
___
### Part 4
Uses `part4_viterbi_modified.py`
#### EN
```
python EvalScript/evalResult.py EN/dev.out EN/dev.p4.out
```

___
### Part 5
Uses `part5_design_challenge.py`    
Implementation of our designed model is explained in greater detail in our report.
#### EN
```
python EvalScript/evalResult.py EN/dev.out EN/dev.p5.out
```

___
## Training
The scripts will train and produce prediction results in the respective folders, e.g. EN/dev.p2.out     
The following commands will produce results for EN for illustration. You will need to replace 'EN' with 'SG' or 'CN' to train for the other languages
### Part 2
```
python Models/part2_simple_system.py EN
```
___
### Part 3
```
python Models/part3_viterbi.py EN
```
___
### Part 4
```
python Models/part4_viterbi_modified.py EN
```
___
### Part 5
Using our designed algorithm - Multi Class Perceptron
```
python Models/part5_design_challenge.py EN
```


___
## Awesome Creators
- Teng Jun Yuan
- Daryll Wong
- Akmal Hakim Teo
