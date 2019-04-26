import torch
import copy


class Utils():
    def __init__(self, train_data, val_data, test_data, model):
        self.original_model = model
        self.best_model = model
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def train(self, max_epochs, batch_size, criterion, optimizer, score_function):
        model = copy.deepcopy(self.original_model)
        train_loader = torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size=batch_size, num_workers=4)
        val_loader = torch.utils.data.DataLoader(self.val_data, shuffle=True, batch_size=batch_size, num_workers=4)

        print("Running epochs... ", end="")
        for epoch in range(max_epochs):
            # print('Epoch ' + str(epoch))
            print(epoch, end=", ")
            model.train()
            losses = []

            for images, targets in train_loader:
                optimizer.zero_grad()

                images = images.cuda()
                targets = targets.cuda()

                out = model(images)

                loss = criterion(out, targets)
                loss.backward()

                optimizer.step()

                losses.append(loss.item())

            # print('loss: ' + str(np.mean(losses)))
            auc_score_val = score_function(model, val_loader)
            auc_score_val = torch.np.mean(auc_score_val)
            # print('AUC Validation: ', str(auc_score_val))
            if auc_score_val > best_auc_score_val:
                best_auc_score_val = auc_score_val
                self.best_model = copy.deepcopy(model)
        print("Done.\n")
        return


def validate(self):
    return


def evaluate(self):
    return
