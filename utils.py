import torch
import copy
import numpy as np
from sklearn.metrics import mean_squared_error


class Utils:
    def __init__(self, train_data, val_data, test_data, model):
        self.original_model = model.cuda()
        self.best_model = model
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def train(self, max_epochs, batch_size, criterion, optimizer, debug=True):
        model = copy.deepcopy(self.original_model)
        train_loader = torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size=batch_size, num_workers=4)
        best_score = 0.0

        print("Running epochs... ")
        for epoch in range(max_epochs):
            model.train()
            losses = []

            for images, labels in train_loader:
                optimizer.zero_grad()

                images = images.cuda()
                out = model(images).cpu()
                loss = criterion(out, labels)
                loss.backward()

                optimizer.step()
                losses.append(loss.item())

            score = self.validate(model, debug)
            if debug:
                print("Epoch:", epoch, "Accuracy:", score, "Loss:", np.mean(losses))
            if score > best_score:
                best_score = score
                self.best_model = copy.deepcopy(model)
        print("Done.\n")
        return self.best_model

    def validate(self, model, debug=True):
        val_loader = torch.utils.data.DataLoader(self.val_data, shuffle=True, batch_size=1, num_workers=4)
        # toggle model to eval mode
        model.eval()

        # turn off gradients since they will not be used here
        # this is to make the inference faster
        with torch.no_grad():
            logits_predicted = np.zeros([0, 1])
            labels = np.zeros([0, 1])
            # run through several batches, does inference for each and store inference results
            # and store both target labels and inferenced scores
            for image, label in val_loader:
                image = image.cuda()
                logit_predicted = model(image)
                logit_predicted = torch.sigmoid(logit_predicted)
                logits_predicted = np.concatenate((logits_predicted, logit_predicted.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)
                # returns a list of scores, one for each of the labels
        return mean_squared_error(labels.reshape([-1]), logits_predicted.reshape([-1]))

        # model.eval()
        # correct = 0
        # incorrect = 0
        # total = 0
        # with torch.no_grad():
        #     for image, label in val_loader:
        #         total += 1
        #         image = image.cuda()
        #         label = label.float()
        #         predicted_label = model(image).cpu()
        #         predicted_label = torch.sigmoid(predicted_label)
        #         if predicted_label[0][0] > label:
        #             correct += 1
        #         else:
        #             incorrect += 1
        # if debug:
        #     print("Correct:", correct, "Incorrect:", incorrect)
        # return correct/total

    def evaluate(self):
        test_loader = torch.utils.data.DataLoader(self.test_data, shuffle=True, batch_size=1, num_workers=4)
        return

    def parameters(self):
        return self.original_model.parameters()
