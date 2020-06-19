# This script is based on tutorial https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

import torch
from imbalanced_sampler import ImbalancedDatasetSampler
from models import SimpleNet
import utils
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--momentum", type=float, required=True)
parser.add_argument("--num_epoch", type=int, required=True)
parser.add_argument("--save", type=str, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--device", type=str, required=True)


args = parser.parse_args()

train_features, train_labels, test_features, test_labels = utils.process_data()

# Create data loaders for train data

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features).float(),
                                               torch.from_numpy(train_labels))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_features).float(),
                                              torch.from_numpy(test_labels))

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=ImbalancedDatasetSampler(train_dataset),
    batch_size=args.batch_size,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    # sampler=ImbalancedDatasetSampler(test_dataset),
    batch_size=args.batch_size,
)

if __name__ == "__main__":
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    logger.info("Number of train samples = {}".format(len(train_loader) * args.batch_size))
    logger.info("Number of test samples = {}".format(len(test_loader) * args.batch_size))

    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    model = SimpleNet(train_features.shape[-1]).to(args.device)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Compute all metrics for initializations

    # Test accuracy per class
    init_test_accuracy_per_class = utils.get_accuracy_per_class(model, test_loader, args.device, 2)
    test_accuracy_per_class = {0: [init_test_accuracy_per_class[0]], 1: [init_test_accuracy_per_class[1]]}

    # Test loss per class
    cur_loss = utils.get_loss_per_class(model, torch.nn.CrossEntropyLoss(reduction="sum"),
                                        test_loader, args.device, 2)
    test_loss_per_class = {0: [cur_loss[0]], 1: [cur_loss[1]]}

    cur_train_loss = utils.get_loss_per_class(model, torch.nn.CrossEntropyLoss(reduction="sum"),
                                              train_loader, args.device, 2)
    train_loss0 = [cur_train_loss[0]]
    train_loss1 = [cur_train_loss[1]]

    torch.save(model.state_dict(), args.save + "/model_params_init")

    for epoch in range(args.num_epoch):  # loop over the dataset multiple times
        model.train()
        running_loss1 = 0.0
        running_loss0 = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            idx0 = labels == 0
            idx1 = labels == 1

            # forward + backward + optimize
            output0 = model(inputs[idx0, :].to(args.device))
            loss0 = criterion(output0, labels[idx0].to(args.device)) / args.batch_size

            output1 = model(inputs[idx1, :].to(args.device))
            loss1 = criterion(output1, labels[idx1].to(args.device)) / args.batch_size
            loss = loss0 + loss1
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss0 += loss0.item()
            running_loss1 += loss1.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                logger.info('[%d, %5d] loss0: %.3f, loss1: %.3f' %
                      (epoch + 1, i + 1, running_loss0 / 20, running_loss1 / 20))
                train_loss0.append(running_loss0 / 20)
                train_loss1.append(running_loss1 / 20)
                running_loss0 = 0.0
                running_loss1 = 0.0

        model.eval()
        cur_test_loss = utils.get_loss_per_class(model, torch.nn.CrossEntropyLoss(reduction="sum"),
                                                 test_loader, args.device, 2)

        test_loss_per_class[0].append(cur_test_loss[0])
        test_loss_per_class[1].append(cur_test_loss[1])

        cur_test_accuracy = utils.get_accuracy_per_class(model, test_loader, args.device, 2)
        test_accuracy_per_class[0].append(cur_test_accuracy[0])
        test_accuracy_per_class[1].append(cur_test_accuracy[1])

        logger.info("Test loss0 = {}, test loss1 = {}".format(test_loss_per_class[0][-1], test_loss_per_class[1][-1]))
        logger.info("Test accuracy0 = {}, test accuracy1 = {}".format(test_accuracy_per_class[0][-1],
                                                                test_accuracy_per_class[1][-1]))

        torch.save(model.state_dict(), args.save + "/model_params_{}".format(epoch))

    np.savez(args.save + "/convergence", test_accuracy_per_class=test_accuracy_per_class,
             test_loss_per_class=test_loss_per_class, train_loss0=train_loss0, train_loss1=train_loss1,
             )

