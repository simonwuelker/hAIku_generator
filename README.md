![Flake8](https://github.com/Wuelle/BW-KI-2020/workflows/Flake8/badge.svg)
![Github All Releases](https://img.shields.io/github/downloads/Wuelle/BW-KI-2020/total.svg)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)
# Haiku Generator
This Repository contains an implementation of the SeqGan Architecture proposed in [this Paper](https://arxiv.org/pdf/1609.05473.pdf) for generating Haikus. It consists, of two main branches, `charlevel`, which generates text
character by character and `Embeddings`, which uses the pytorch Embeddings to generate text.
This Branch, `Gaussian-Generator` attempts to extend the seqGAN Algorithm into a continuous Actionspace by having the Policy Gradient output
a multivariate Gaussian Distribution across the output space instead of a probability distribution
across a finite number of  actions.
This project is my submission for the [BW-KI 2020](https://bw-ki.de/) Competition.

## Functionality
The SeqGAN Algorithm was first proposed by [this Paper](https://arxiv.org/pdf/1609.05473.pdf). Like in a normal GAN, the Discrimnator Model 
rates the samples produced by the Generator on a scale from zero to one based on how realistic they seem. However, since the samples are
sequential and the Discriminator cannot judge partial Sequences, it is unable to directly provide a reward for each timestep to the Generator.
This Problem is solved by using the classic RL-Algorithm REINFORCE as a Generator Model and rolling out the sequences using Monte-Carlo.
[SeqGAN-image](https://www.researchgate.net/publication/325709720/figure/fig1/AS:636539755302912@1528774312038/An-illustration-of-SeqGAN-for-text-generation-27-Compared-to-one-step-generation-of.png)
One Downside of this Procedure is that a REINFORCE Agent can only act in a discrete environment since its output is a probability distribution over a 
finite number of actions. A somewhat working Implementation of the classic Algorithm can be found in the `charlevel` Branch. I am trying to extend the
Generator into a continuous Action Space by having the Generator output a multivariate Gaussian Distribution. Exploration can be ensured
by having the Model only output the mean but not the standard deviation for each of the `num_actions` distributions.

## Usage
For the sake of keeping it simple, I am not uploading my dataset here. However, you can just use the Haikus from
[this Repository](https://github.com/docmarionum1/haikurnn) and preprocess it using 
[this notebook](https://github.com/Wuelle/BW-KI-2020/blob/Embeddings/Dataset%20Analysis%20and%20Preprocessing.ipynb)

If you have acquired enough Haikus, place them in the data folder and run the [pretrain.py](https://github.com/Wuelle/BW-KI-2020/blob/Embeddings/pretrain.py)
 file. After thats finished, run the [main.py](https://github.com/Wuelle/BW-KI-2020/blob/Embeddings/main.py) file which will store
 the final generator model under [models/Generator.pt](https://github.com/Wuelle/BW-KI-2020/blob/Embeddings/models/Generator.pt)

## Examples
These are some handpicked samples from the generator.
>scattered rare conversant entails snatchers
>

## Future Goals
* Scrape Haikus from Twitter
* Improve charlevel branch
* Create contributing guidelines

## Credits
Though i plan to use my own data eventually, right now i am using a processed version of the dataset from
this [HaikuRNN Repository](https://github.com/docmarionum1/haikurnn) for training my model.

## License
This program is licensed under the GNU General Public License v3.0. For more Information, see
[LICENSE.txt](https://github.com/Wuelle/BW-KI-2020/blob/Embeddings/LICENSE.txt)
