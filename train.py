import math
import os
import time

import torch
import torch.nn as nn
from torchvision import transforms

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from utils import train_one, validate, save_epoch, early_stopping


def train(batch_size=32, vocab_threshold=5, vocab_from_file=True,
          embed_size=256, hidden_size=512, num_epochs=10, cocoapi_dir="/Users/xqk/Tools/Coco/"):
    # Keep track of train and validation losses and validation Bleu-4 scores by epoch
    train_losses = []
    val_losses = []
    val_bleus = []
    # Keep track of the current best validation Bleu score
    best_val_bleu = float("-INF")

    # Define a transform to pre-process the training images
    transform_train = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    # Define a transform to pre-process the validation images
    transform_val = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.CenterCrop(224),  # get 224x224 crop from the center
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    # Build data loader, applying the transforms
    train_loader = get_loader(transform=transform_train,
                              mode='train',
                              batch_size=batch_size,
                              vocab_threshold=vocab_threshold,
                              vocab_from_file=vocab_from_file,
                              cocoapi_loc=cocoapi_dir)
    val_loader = get_loader(transform=transform_val,
                            mode='val',
                            batch_size=batch_size,
                            vocab_threshold=vocab_threshold,
                            vocab_from_file=vocab_from_file,
                            cocoapi_loc=cocoapi_dir)

    # The size of the vocabulary
    vocab_size = len(train_loader.dataset.vocab)

    # Initialize the encoder and decoder
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # Move models to GPU if CUDA is available
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Define the loss function
    loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Specify the learnable parameters of the model
    params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

    # Define the optimizer
    optimizer = torch.optim.Adam(params=params, lr=0.001)

    # Set the total number of training and validation steps per epoch
    total_train_step = math.ceil(len(train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size)
    total_val_step = math.ceil(len(val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size)

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one(train_loader, encoder, decoder, loss, optimizer,
                           vocab_size, epoch, total_train_step)
        train_losses.append(train_loss)
        val_loss, val_bleu = validate(val_loader, encoder, decoder, loss,
                                      train_loader.dataset.vocab, epoch, total_val_step)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)
        if val_bleu > best_val_bleu:
            print("Validation Bleu-4 improved from {:0.4f} to {:0.4f}, saving model to best-model.pkl".
                  format(best_val_bleu, val_bleu))
            best_val_bleu = val_bleu
            filename = os.path.join("./models", "best-model.pkl")
            save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses,
                       val_bleu, val_bleus, epoch)
        else:
            print("Validation Bleu-4 did not improve, saving model to model-{}.pkl".format(epoch))
        # Save the entire model anyway, regardless of being the best model so far or not
        filename = os.path.join("./models", "model-{}.pkl".format(epoch))
        save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses,
                   val_bleu, val_bleus, epoch)
        print("Epoch [%d/%d] took %ds" % (epoch, num_epochs, time.time() - start_time))
        if epoch > 5:
            # Stop if the validation Bleu doesn't improve for 3 epochs
            if early_stopping(val_bleus, 3):
                break
        start_time = time.time()
