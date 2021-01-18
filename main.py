import argparse
import os
import re
from train import train


def parse():
    parser = argparse.ArgumentParser(description='Image caption')
    parser.add_argument('-c', '--coco', help='Path of coco dataset')
    parser.add_argument('-m', '--model', help='Path of models')

    return parser.parse_args()


def latest_model_dir(model_dir):
    if not os.path.exists(model_dir):
        print("Model is not exist")
        exit(1)
    files = os.listdir(model_dir)
    files = [int(re.findall('model-(.*).pkl', file)[0]) for file in files if file.startswith("model")]
    if len(files) == 0:
        return None
    files = sorted(files)
    return os.path.join(model_dir, 'model-{}.pkl'.format(files[-1]))


if __name__ == '__main__':
    args = parse()

    batch_size = 32  # batch size
    vocab_threshold = 5  # minimum word count threshold
    vocab_from_file = True  # if True, load existing vocab file
    embed_size = 256  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder
    num_epochs = 10  # number of training epochs

    coco_loc = args.coco
    model_dir = args.model

    latest_model = None
    if model_dir:
        latest_model = latest_model_dir(model_dir)

    train(batch_size, vocab_threshold, vocab_from_file, embed_size, hidden_size, num_epochs, latest_model, coco_loc)
