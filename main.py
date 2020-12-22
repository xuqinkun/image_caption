import argparse
from train import train


def parse():
    parser = argparse.ArgumentParser(description='Image caption')
    parser.add_argument('-c', '--coco', help='Path of coco dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    batch_size = 32  # batch size
    vocab_threshold = 5  # minimum word count threshold
    vocab_from_file = True  # if True, load existing vocab file
    embed_size = 256  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder
    num_epochs = 10  # number of training epochs
    coco_loc = args.coco
    if coco_loc:
        train(batch_size, vocab_threshold, vocab_from_file, embed_size, hidden_size, num_epochs, coco_loc)
    else:
        train(batch_size, vocab_threshold, vocab_from_file, embed_size, hidden_size, num_epochs)

