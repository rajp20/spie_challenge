import torch
import copy


class Utils:
    def __init__(self, train_data, val_data, test_data, model):
        self.original_model = model().cuda()
        self.best_model = model()
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def train(self, max_epochs, batch_size, criterion, optimizer, debug=True):
        model = copy.deepcopy(self.original_model)
        train_loader = torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size=batch_size, num_workers=4)
        best_score = 0.0

        print("Running epochs... ")
        for epoch in range(max_epochs):
            # print('Epoch ' + str(epoch))
            print(epoch, end=", ")
            model.train()
            losses = []

            for images, labels in train_loader:
                optimizer.zero_grad()

                images = images.cuda()
                out = model(images)

                loss = criterion(out.cpu(), labels.long())
                loss.backward()

                optimizer.step()
                losses.append(loss.item())

            score = self.validate(model, debug)
            if debug:
                print("Epoch:", epoch, "Accuracy:", score)
            if score > best_score:
                best_score = score
                self.best_model = copy.deepcopy(model)
        print("Done.\n")
        return self.best_model

    def validate(self, model, debug=True):
        val_loader = torch.utils.data.DataLoader(self.val_data, shuffle=True, batch_size=1, num_workers=4)
        model.eval()
        correct = 0
        incorrect = 0
        total = 0
        with torch.no_grad():
            total += 1
            for image, label in val_loader:
                image.cuda()
                predicted_label = model(image)
                if predicted_label > label:
                    correct += 1
                else:
                    incorrect += 1
        if debug:
            print("Correct:", correct, "Incorrect:", incorrect)
        return correct/total

    def evaluate(self):
        test_loader = torch.utils.data.DataLoader(self.test_data, shuffle=True, batch_size=1, num_workers=4)
        return

    def parameters(self):
        return self.original_model.parameters()
