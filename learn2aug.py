import argparse
from cgi import test
from multiprocessing import reduction

import os
import random
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from tqdm import trange
from load_data import DataGenerator
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as T
from transformers import ViTImageProcessor, ViTFeatureExtractor, ViTModel
from transformers import AutoImageProcessor, ResNetModel
from UNet import UNet
from PIL import Image

def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim, device):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.device = device

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        B, K_add_1, N, img_size = input_images.shape
        final_labels = torch.cat(
            (input_labels[:, :-1, :, :], torch.zeros((B, 1, N, N), device=self.device)),
            dim=1,
        )
        model_input = torch.cat((input_images, final_labels), dim=-1).reshape(
            B, -1, img_size + N
        )
        x, _ = self.layer1.to(torch.float64)(model_input)
        x, _ = self.layer2.to(torch.float64)(x)
        x = x.reshape(B, K_add_1, N, N)
        return x
        #############################

    def loss_vit(self, images, augmented_images):
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        # model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        vit_features = feature_extractor(augmented_images, return_tensors="pt")
        unet_repr = Encoder()(images)
        distillation_loss = torch.nn.MSELoss(reduction="mean")(unet_repr, vit_features)
        return distillation_loss

    def loss_function(self, preds, labels, aug_img_feat, autoaug_img_feat):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################
        #### YOUR CODE GOES HERE ####
        return F.cross_entropy(
            preds[:, -1, :, :], labels[:, -1, :, :], reduction="mean"
        ) + F.mse_loss(aug_img_feat, autoaug_img_feat)
        #############################


def train_step(images, labels, model, optim, eval=False):
    B, K_add_1, N, img_size = images.shape
    
    # Extract features of UNet output and autoaug images
    aug_imgs, aug_img_feat, autoaug_img_feat = extract_features(images[:, :-1, :, :].reshape(-1, 1, 28, 28))

    # Include augmented images in support set
    aug_imgs = aug_imgs.reshape(B, K_add_1 - 1, N, img_size)
    images = torch.cat((aug_imgs, images), dim=1)
    labels = torch.cat((labels[:, :-1, :, :], labels), dim=1)

    predictions = model(images, labels)

    loss = model.loss_function(predictions, labels, aug_img_feat, autoaug_img_feat)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()

def embed_image(image, feature_extractor, vitmodel):
    inputs = feature_extractor(image.repeat(3,1,1), return_tensors="pt")

    with torch.no_grad():
        outputs = vitmodel(**inputs)

    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.view(1,-1)

def extract_features(image_batch):
    aug_img_feat = []
    autoaug_img_feat = []

    unet_model = UNet()
    augmenter = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)

    # ViT
    # feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    # img_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # ResNet
    feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    img_model = ResNetModel.from_pretrained("microsoft/resnet-50")

    aug_imgs = unet_model(image_batch)

    for idx in range(image_batch.shape[0]):
        autoaug_img = augmenter(((1.0 - image_batch[idx]) * 255.0).to(torch.uint8))
        aug_img_feat.append(embed_image(aug_imgs[idx], feature_extractor, img_model))
        autoaug_img_feat.append(embed_image(autoaug_img, feature_extractor, img_model))

    aug_img_feat = torch.stack(aug_img_feat)
    autoaug_img_feat = torch.stack(autoaug_img_feat)

    return aug_imgs, aug_img_feat, autoaug_img_feat

def main(config):
    print(config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    writer = SummaryWriter(
            f"runs/{config.num_classes}_{config.num_shot}_{config.random_seed}_{config.hidden_dim}_test"
        )

    # if config.augment_support_set:
    #     writer = SummaryWriter(
    #         f"runs/{config.num_classes}_{config.num_shot}_{config.random_seed}_{config.hidden_dim}_{config.augmenter}"
    #     )
    # else:
    #     writer = SummaryWriter(
    #         f"runs/{config.num_classes}_{config.num_shot}_{config.random_seed}_{config.hidden_dim}"
    #     )

    # Download Omniglot Dataset
    if not os.path.isdir("./omniglot_resized"):
        gdd.download_file_from_google_drive(
            file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
            dest_path="./omniglot_resized.zip",
            unzip=True,
        )
    assert os.path.isdir("./omniglot_resized")

    # Create Data Generator
    train_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="train",
        device=device,
        cache=config.image_caching,
        augment_support_set=config.augment_support_set,
        augmenter=config.augmenter,
    )
    train_loader = iter(
        torch.utils.data.DataLoader(
            train_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    test_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="test",
        device=device,
        cache=config.image_caching,
    )
    test_loader = iter(
        torch.utils.data.DataLoader(
            test_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    # Create model
    model = MANN(config.num_classes, config.num_shot + 1, config.hidden_dim, device)
    model.to(device)

    # Create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    import time

    # unet_model = UNet(enc_chs=(1, 64, 128, 256), dec_chs=(256, 128, 64),
    #              retain_dim=True, out_sz=(28,28))

    times = []
    for step in trange(config.train_steps):
        ## Sample Batch
        t0 = time.time()
        i, l = next(train_loader)
        i, l = i.to(device), l.to(device)
        t1 = time.time()

        ## Train
        _, ls = train_step(i, l, model, optim)
        t2 = time.time()
        print("Train Loss: ", ls.cpu().numpy())
        writer.add_scalar("Loss/train", ls, step)
        times.append([t1 - t0, t2 - t1])

        ## Evaluate
        if (step + 1) % config.eval_freq == 0:
            print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
            i, l = next(test_loader)
            i, l = i.to(device), l.to(device)
            pred, tls = train_step(i, l, model, optim, eval=True)
            print("Train Loss:", ls.cpu().numpy(), "Test Loss:", tls.cpu().numpy())
            writer.add_scalar("Loss/test", tls, step)
            pred = torch.reshape(
                pred, [-1, config.num_shot + 1, config.num_classes, config.num_classes]
            )
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            l = torch.argmax(l[:, -1, :, :], axis=2)
            acc = pred.eq(l).sum().item() / (
                config.meta_batch_size * config.num_classes
            )
            print("Test Accuracy", acc)
            writer.add_scalar("Accuracy/test", acc, step)

            times = np.array(times)
            print(f"Sample time {times[:, 0].mean()} Train time {times[:, 1].mean()}")
            times = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_shot", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--meta_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_steps", type=int, default=25000)
    parser.add_argument("--image_caching", type=bool, default=True)
    parser.add_argument("--augment_support_set", type=bool, default=False)
    parser.add_argument("--augmenter", type=str, default="randaug")
    main(parser.parse_args())
