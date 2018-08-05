# Discourse Marker Prediction (DMP)
Code for the DMP task in "Discourse Marker Augmented Network with Reinforcement Learning for Natural Language Inference".
If you use this code as part of any published research, please acknowledge the following paper.

**"Discourse Marker Augmented Network with Reinforcement Learning for Natural Language Inference"**
Boyuan Pan, Yazheng Yang, Zhou Zhao, Yueting Zhuang, Deng Cai, Xiaofei He. _ACL (2018)_ 

```
@inproceedings{pan2018discourse,
  title={Discourse Marker Augmented Network with Reinforcement Learning for Natural Language Inference},
  author={Pan, Boyuan and Yang, Yazheng and Zhao, Zhou and Zhuang, Yueting and Cai, Deng and He, Xiaofei},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  volume={1},
  pages={989--999},
  year={2018}
}
```
## Required
* Python 3.6
* Tensorflow r1.3

## Running the Script
1. Download the dataset.
The pre-processed dataset we used for training our model is now available [here](http://www.cs.toronto.edu/~mbweb/).

If you use the BookCorpus data in your work, please also cite:

Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, Sanja Fidler.
**"Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books."** *arXiv preprint arXiv:1506.06724 (2015).*

    @article{zhu2015aligning,
        title={Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books},
        author={Zhu, Yukun and Kiros, Ryan and Zemel, Richard and Salakhutdinov, Ruslan and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
        journal={arXiv preprint arXiv:1506.06724},
        year={2015}
    }

2. Train and test model for DMP
```
python main.py --mode train/test
```

The path of the dataset can be set on your own.
