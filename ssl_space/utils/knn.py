# Portions adapted and modified for STOMP from solo-learn (MIT).
# Copyright 2021, 2023 solo-learn development team.
# See THIRD_PARTY_NOTICES.md and LICENSES/solo-learn-MIT.txt.

from typing import Tuple

import torch
import torch.nn.functional as F
from torchmetrics.metric import Metric

class WeightedKNNClassifier(Metric):
    def __init__(
        self,
        k: int = 20,
        T: float = 0.07,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        dist_sync_on_step: bool = False,
    ):
        """Implements the weighted k-NN classifier used for evaluation.

        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
            max_distance_matrix_size (int, optional): maximum number of elements in the
                distance matrix. Defaults to 5e6.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon

        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.

        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    @torch.no_grad()
    def compute(self) -> Tuple[float]:
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """


        # if compute is called without any features
        if not self.train_features or not self.test_features:
            return -1, -1

        train_features = torch.cat(self.train_features).to(torch.float32)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features).to(torch.float32)
        test_targets = torch.cat(self.test_targets)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        chunk_size = min(
            max(1, self.max_distance_matrix_size // num_train_images),
            num_test_images,
        )
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    similarities.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (
                top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
            )  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        self.reset()

        return top1, top5

class WeightedKNNClassifier_MultiValidation(Metric):
    def __init__(
        self,
        k: int = 20,
        T: float = 0.07,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        dist_sync_on_step: bool = False,
    ):
        """Implements the weighted k-NN classifier used for evaluation.

        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
            max_distance_matrix_size (int, optional): maximum number of elements in the
                distance matrix. Defaults to 5e6.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon

        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)
        # add extra states for mutliple validation passes over different validation sets
        self.add_state("test_features_corr", default=[], persistent=False)
        self.add_state("test_targets_corr", default=[], persistent=False)
        self.add_state("test_features_uncorr", default=[], persistent=False)
        self.add_state("test_targets_uncorr", default=[], persistent=False)

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
        test_features_corr: torch.Tensor = None,
        test_targets_corr: torch.Tensor = None,
        test_features_uncorr: torch.Tensor = None,
        test_targets_uncorr: torch.Tensor = None,
        
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.

        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)
        assert (test_features_corr is None) == (test_targets_corr is None)
        assert (test_features_uncorr is None) == (test_targets_uncorr is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())
        
        if test_features_corr is not None:
            assert test_features_corr.size(0) == test_targets_corr.size(0)
            self.test_features_corr.append(test_features_corr.detach())
            self.test_targets_corr.append(test_targets_corr.detach())
        
        if test_features_uncorr is not None:
            assert test_features_uncorr.size(0) == test_targets_uncorr.size(0)
            self.test_features_uncorr.append(test_features_uncorr.detach())
            self.test_targets_uncorr.append(test_targets_uncorr.detach())

    @torch.no_grad()
    def compute(self) -> Tuple[float]:
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """
        results = {"default": None, "corr": None, "uncorr": None}
        if not self.train_features:
            results["default"] = (-1, -1)
            results["corr"] = (-1, -1)
            results["uncorr"] = (-1, -1)
            return results

        all_test_features = {}
        if self.test_features:
            all_test_features["default"] = (self.test_features, self.test_targets)
        else:
            results["default"] = (-1, -1)
        if self.test_features_corr:
            all_test_features["corr"] = (self.test_features_corr, self.test_targets_corr)
        else:
            results["corr"] = (-1, -1)
        if self.test_features_uncorr:
            all_test_features["uncorr"] = (self.test_features_uncorr, self.test_targets_uncorr)
            
        else:
            results["uncorr"] = (-1, -1)
        
        train_features = torch.cat(self.train_features)
        train_targets = torch.cat(self.train_targets)
        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)

        for type_str, test_data in all_test_features.items():        
            test_features = torch.cat(test_data[0])
            test_targets = torch.cat(test_data[1])

            if self.distance_fx == "cosine":
                test_features = F.normalize(test_features)

            num_classes = torch.unique(test_targets).numel()
            num_train_images = train_targets.size(0)
            num_test_images = test_targets.size(0)
            chunk_size = min(
                max(1, self.max_distance_matrix_size // num_train_images),
                num_test_images,
            )
            k = min(self.k, num_train_images)

            top1, top5, total = 0.0, 0.0, 0
            retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
            for idx in range(0, num_test_images, chunk_size):
                # get the features for test images
                features = test_features[idx : min((idx + chunk_size), num_test_images), :]
                targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
                batch_size = targets.size(0)

                # calculate the dot product and compute top-k neighbors
                if self.distance_fx == "cosine":
                    similarities = torch.mm(features, train_features.t())
                elif self.distance_fx == "euclidean":
                    similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
                else:
                    raise NotImplementedError

                similarities, indices = similarities.topk(k, largest=True, sorted=True)
                candidates = train_targets.view(1, -1).expand(batch_size, -1)
                retrieved_neighbors = torch.gather(candidates, 1, indices)

                retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
                retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

                if self.distance_fx == "cosine":
                    similarities = similarities.clone().div_(self.T).exp_()

                probs = torch.sum(
                    torch.mul(
                        retrieval_one_hot.view(batch_size, -1, num_classes),
                        similarities.view(batch_size, -1, 1),
                    ),
                    1,
                )
                _, predictions = probs.sort(1, True)

                # find the predictions that match the target
                correct = predictions.eq(targets.data.view(-1, 1))
                top1 = top1 + correct.narrow(1, 0, 1).sum().item()
                top5 = (
                    top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
                )  # top5 does not make sense if k < 5
                total += targets.size(0)

            top1 = top1 * 100.0 / total
            top5 = top5 * 100.0 / total
            results[type_str] = (top1, top5)

        self.reset()

        return results