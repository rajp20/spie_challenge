import torch
import copy
import numpy as np
import sklearn.metrics as metrics


class Utils:
    def __init__(self, train_data, val_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def train(self, model, max_epochs, batch_size, criterion, optimizer, debug=True):
        print("Training", "Max Epochs:", max_epochs, "Batch Size:", batch_size)
        model = model.cuda()
        train_loader = torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size=batch_size, num_workers=4)
        best_score = 0.0
        epoch_scores = []
        epoch_losses = []
        best_model = None

        print("Running epochs... ")
        for epoch in range(max_epochs):
            model.train()
            losses = []

            for images, labels in train_loader:
                optimizer.zero_grad()

                images = images.cuda()
                out = model(images).cpu()
                out = torch.sigmoid(out)
                loss = criterion(out, labels)
                loss.backward()

                optimizer.step()
                losses.append(loss.item())

            score = self.validate(model, debug)
            mean_loss = np.mean(losses)
            epoch_scores.append(score)
            epoch_losses.append(mean_loss)
            if debug:
                print("Epoch:", epoch, "Prediction Probability:", score, "Training Loss:", mean_loss)
            if mean_loss > best_score:
                best_score = score
                best_model = copy.deepcopy(model)
        print("Done.\n")
        return best_model, epoch_losses, epoch_scores

    def validate(self, model, debug=True):
        val_loader = torch.utils.data.DataLoader(self.val_data, shuffle=True, batch_size=1, num_workers=4)
        # toggle model to eval mode
        model.eval()

        # turn off gradients since they will not be used here
        # this is to make the inference faster
        with torch.no_grad():
            logits_predicted = []
            labels = []
            # run through several batches, does inference for each and store inference results
            # and store both target labels and inferenced scores
            for image, label in val_loader:
                image = image.cuda()
                predicted = model(image)
                predicted = torch.sigmoid(predicted)
                logits_predicted.append(predicted.cpu().detach().numpy())
                labels.append(label.cpu().detach().numpy())
                # returns a list of scores, one for each of the labels
        return self.predict_prob(labels, logits_predicted, initial_lexsort=True)

    def predict_prob(self, x, y, initial_lexsort=True):
        """
        Calculates the prediction probability. Adapted from scipy's implementation of Kendall's Tau

        Note: x should be the truth labels.

        Parameters
        ----------
        x, y : array_like
            Arrays of rankings, of the same shape. If arrays are not 1-D, they will
            be flattened to 1-D.
        initial_lexsort : bool, optional
            Whether to use lexsort or quicksort as the sorting method for the
            initial sort of the inputs. Default is lexsort (True), for which
            `predprob` is of complexity O(n log(n)). If False, the complexity is
            O(n^2), but with a smaller pre-factor (so quicksort may be faster for
            small arrays).
        Returns
        -------
        Prediction probability : float

        Notes
        -----
        The definition of prediction probability that is used is:
          p_k = (((P - Q) / (P + Q + T)) + 1)/2
        where P is the number of concordant pairs, Q the number of discordant
        pairs, and T the number of ties only in `y`.
        References
        ----------
        Smith W.D, Dutton R.C, Smith N.T. (1996) A measure of association for assessing prediction accuracy
        that is a generalization of non-parametric ROC area. Stat Med. Jun 15;15(11):1199-215
        """

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        if not x.size or not y.size:
            return (np.nan, np.nan)  # Return NaN if arrays are empty

        n = np.int64(len(x))
        temp = list(range(n))  # support structure used by mergesort

        # this closure recursively sorts sections of perm[] by comparing
        # elements of y[perm[]] using temp[] as support
        # returns the number of swaps required by an equivalent bubble sort

        def mergesort(offs, length):
            exchcnt = 0
            if length == 1:
                return 0
            if length == 2:
                if y[perm[offs]] <= y[perm[offs + 1]]:
                    return 0
                t = perm[offs]
                perm[offs] = perm[offs + 1]
                perm[offs + 1] = t
                return 1
            length0 = length // 2
            length1 = length - length0
            middle = offs + length0
            exchcnt += mergesort(offs, length0)
            exchcnt += mergesort(middle, length1)
            if y[perm[middle - 1]] < y[perm[middle]]:
                return exchcnt
            # merging
            i = j = k = 0
            while j < length0 or k < length1:
                if k >= length1 or (j < length0 and y[perm[offs + j]] <=
                                    y[perm[middle + k]]):
                    temp[i] = perm[offs + j]
                    d = i - j
                    j += 1
                else:
                    temp[i] = perm[middle + k]
                    d = (offs + i) - (middle + k)
                    k += 1
                if d > 0:
                    exchcnt += d
                i += 1
            perm[offs:offs + length] = temp[0:length]
            return exchcnt

        # initial sort on values of x and, if tied, on values of y
        if initial_lexsort:
            # sort implemented as mergesort, worst case: O(n log(n))
            perm = np.lexsort((y, x))
        else:
            # sort implemented as quicksort, 30% faster but with worst case: O(n^2)
            perm = list(range(n))
            perm.sort(key=lambda a: (x[a], y[a]))

        # compute joint ties
        first = 0
        t = 0
        for i in range(1, n):
            if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
                t += ((i - first) * (i - first - 1)) // 2
                first = i
        t += ((n - first) * (n - first - 1)) // 2

        # compute ties in x
        first = 0
        u = 0
        for i in range(1, n):
            if x[perm[first]] != x[perm[i]]:
                u += ((i - first) * (i - first - 1)) // 2
                first = i
        u += ((n - first) * (n - first - 1)) // 2

        # count exchanges
        exchanges = mergesort(0, n)
        # compute ties in y after mergesort with counting
        first = 0
        v = 0
        for i in range(1, n):
            if y[perm[first]] != y[perm[i]]:
                v += ((i - first) * (i - first - 1)) // 2
                first = i
        v += ((n - first) * (n - first - 1)) // 2

        tot = (n * (n - 1)) // 2
        if tot == u or tot == v:
            return (np.nan, np.nan)  # Special case for all ties in both ranks

        p_k = (((tot - (v + u - t)) - 2.0 * exchanges) / (tot - u) + 1) / 2

        return p_k

    def evaluate(self):
        test_loader = torch.utils.data.DataLoader(self.test_data, shuffle=True, batch_size=1, num_workers=4)
        return
