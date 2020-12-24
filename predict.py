import os

import numpy as np
import torch
import argparse
from torchvision import transforms

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from utils import clean_sentence, get_prediction

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse():
    parser = argparse.ArgumentParser(description='Image caption')
    parser.add_argument('-c', '--coco', help='Path of coco dataset')
    parser.add_argument('-m', '--model', help='Path of model')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse()
    coco_dir = args.coco
    model_dir = args.model

    # Define a transform to pre-process the testing images
    transform_test = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.CenterCrop(224),  # get 224x224 crop from the center
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    # Create the data loader
    data_loader = get_loader(transform=transform_test,
                             mode='test',
                             cocoapi_loc=coco_dir)
    # Obtain sample image before and after pre-processing
    orig_image, image = next(iter(data_loader))
    # Convert image from torch.FloatTensor to numpy ndarray
    transformed_image = image.numpy()
    # Remove the first dimension which is batch_size euqal to 1
    transformed_image = np.squeeze(transformed_image)
    transformed_image = transformed_image.transpose((1, 2, 0))

    # Load the most recent checkpoint
    checkpoint = torch.load(model_dir)

    # Specify values for embed_size and hidden_size - we use the same values as in training step
    embed_size = 256
    hidden_size = 512

    # Get the vocabulary and its size
    vocab = data_loader.dataset.vocab
    vocab_size = len(vocab)

    # Initialize the encoder and decoder, and set each to inference mode
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Load the pre-trained weights
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    # Move models to GPU if CUDA is available.
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Obtain the embedded image features.
    features = encoder(image).unsqueeze(1)

    # Pass the embedded image features through the model to get a predicted caption.
    output = decoder.sample(features)
    print('example output:', output)

    assert (type(output) == list), "Output needs to be a Python list"
    assert all([type(x) == int for x in output]), "Output should be a list of integers."
    assert all([x in data_loader.dataset.vocab.idx2word for x in
                output]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

    sentence = clean_sentence(output, vocab)
    print('example sentence:', sentence)

    assert type(sentence) == str, 'Sentence needs to be a Python string!'

    get_prediction(data_loader, encoder, decoder, vocab)
