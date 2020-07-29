![Flake8](https://github.com/Wuelle/BW-KI-2020/workflows/Flake8/badge.svg)
![Github All Releases](https://img.shields.io/github/downloads/Wuelle/BW-KI-2020/total.svg)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)
# Haiku Generator
This Repository contains an implementation of the SeqGan Architecture proposed in [this Paper](https://arxiv.org/pdf/1609.05473.pdf) for generating Haikus. It consists, of two main branches, `charlevel`, which generates text
character by character and `Embeddings`, which uses the pytorch Embeddings to generate text.
This project is my submission for the [BW-KI](https://bw-ki.de/) Competition

## Usage
For the sake of keeping it simple, I am not uploading my dataset here. However, you can just use the Haikus from
[this Repository](https://github.com/docmarionum1/haikurnn) and preprocess it using 
[this notebook](https://github.com/Wuelle/BW-KI-2020/blob/Embeddings/Dataset%20Analysis%20and%20Preprocessing.ipynb)

If you have acquired enough Haikus, place them in the data folder and run the [pretrain.py](https://github.com/Wuelle/BW-KI-2020/blob/Embeddings/pretrain.py)
 file. After thats finished, run the [main.py](https://github.com/Wuelle/BW-KI-2020/blob/Embeddings/main.py) file which will store
 the final generator model under [models/Generator.pt](https://github.com/Wuelle/BW-KI-2020/blob/Embeddings/models/Generator.pt)

## Examples
These are some handpicked samples from the generator.
>lotta dreams regard condoms
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
