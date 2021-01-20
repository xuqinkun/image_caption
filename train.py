import math
import os
import time

import torch
import torch.nn as nn
from torchvision import transforms

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from utils import train_one, save_epoch


def train(batch_size=32, vocab_threshold=5, vocab_from_file=True,
          embed_size=256, hidden_size=512, num_epochs=10, latest_model=None, cocoapi_dir="./Coco/"):
    # Keep track of train and validation losses and validation Bleu-4 scores by epoch
    train_losses = []

    # Define a transform to pre-process the training images
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Build data loader, applying the transforms
    train_loader = get_loader(transform=transform_train,
                              mode='train',
                              batch_size=batch_size,
                              vocab_threshold=vocab_threshold,
                              vocab_from_file=vocab_from_file,
                              cocoapi_loc=cocoapi_dir)

    # The size of the vocabulary
    vocab_size = len(train_loader.dataset.vocab)

    # Initialize the encoder and decoder
    checkpoint = None
    if latest_model:
        checkpoint = torch.load(latest_model)
    start_epoch = 1
    if checkpoint:
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        start_epoch = checkpoint['epoch']
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    if checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])

    # Move models to GPU if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        encoder.cuda()
        decoder.cuda()

    # Define the loss function
    loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Specify the learnable parameters of the model
    params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

    # Define the optimizer
    optimizer = torch.optim.Adam(params=params, lr=0.001)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Set the total number of training and validation steps per epoch
    total_train_step = math.ceil(len(train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size)

    start_time = time.time()
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one(train_loader, encoder, decoder, loss, optimizer,
                           vocab_size, epoch, total_train_step)
        train_losses.append(train_loss)
        # Save the entire model anyway, regardless of being the best model so far or not
        filename = os.path.join("./models", "model-{}.pkl".format(epoch))
        save_epoch(filename, encoder, decoder, optimizer, train_losses, epoch)
        print("Epoch [%d/%d] took %ds" % (epoch, num_epochs, time.time() - start_time))
        start_time = time.time()
