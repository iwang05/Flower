# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, aggregate_drop, weighted_loss_avg
from .strategy import Strategy
import numpy as np
import random

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""


class FedDrop(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.droppedWeights: Dict[str, List] = {}
        self.straggler = []

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        random_client = client_manager.sample(1)[0]
        self.initial_parameters = random_client.get_parameters().parameters
        initial_parameters = self.initial_parameters
        #self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):
            log(WARNING, DEPRECATION_WARNING_INITIAL_PARAMETERS)
            initial_parameters = weights_to_parameters(weights=initial_parameters)
        return initial_parameters

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights)
        if eval_res is None:
            return None
        loss, other = eval_res
        if isinstance(other, float):
            print(DEPRECATION_WARNING)
            metrics = {"accuracy": other}
        else:
            metrics = other
        return loss, metrics

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        config_drop = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
            config_drop = self.on_fit_config_fn(rnd,0.8)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        clientList = []
        for client in clients:
            if (client.cid in self.straggler) and rnd > 1:
                fit_ins_drop = FitIns(self.drop_rand(parameters, 0.8, [0,4], 2, client.cid), config_drop)
                clientList.append((client, fit_ins_drop))
            else:
                clientList.append((client, fit_ins))
        return clientList

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples, client.cid)
            for client, fit_res in results
        ]

        if (len(self.straggler) == 0) and rnd > 1:
            def time(elem):
                return elem[1].fit_duration

            results.sort(key=time)
            self.straggler.append(results[len(results) - 1][0].cid)
            print(self.straggler)
        return weights_to_parameters(aggregate_drop(weights_results, self.droppedWeights, parameters_to_weights(self.initial_parameters))), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {}


    def drop_rand(self, parameters: Parameters, p: float, idxList: List[int], idxConvFC: int, cid:str):
        weights = parameters_to_weights(parameters)
        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[],[]] for x in range(len(weights))]

        for idx in idxList:
            shape = weights[idx].shape
            list = random.sample(range(1, shape[0]), shape[0] - int(p * shape[0]))
            list.sort()

            self.droppedWeights[cid][idx][0] = list.copy()
            self.droppedWeights[cid][idx+1][0] = list.copy()
            self.droppedWeights[cid][idx+2][1] = list.copy()

            # remove each row/column from the back
            weights[idx] = np.delete(weights[idx], list, 0)
            weights[idx + 1] = np.delete(weights[idx + 1], list, 0)
            weights[idx + 2] = np.delete(weights[idx + 2], list, 1)


        shape = weights[idxConvFC].shape
        listFC: Tuple = random.sample(range(1, shape[0]), shape[0] - int(p * shape[0]))
        listFC.sort()
        FClist = []
        for drop in listFC:
            FClist.extend(range(drop*7*7, drop*7*7 + 7*7))

        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()

        weights[idxConvFC] = np.delete(weights[idxConvFC], listFC, 0)
        weights[idxConvFC + 1] = np.delete(weights[idxConvFC + 1], listFC, 0)
        weights[idxConvFC + 2] = np.delete(weights[idxConvFC + 2], FClist, 1)

        return weights_to_parameters(weights)

    def drop_rand_back(self, parameters: Parameters, p: float, idxList: List[int], idxConvFC: int, cid:str):
        weights = parameters_to_weights(parameters)
        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[],[]] for x in range(len(weights))]

        for idx in idxList:
            shape = weights[idx].shape

            list = [i for i in range(int(p * shape[0]), shape[0])]

            self.droppedWeights[cid][idx][0] = list.copy()
            self.droppedWeights[cid][idx+1][0] = list.copy()
            self.droppedWeights[cid][idx+2][1] = list.copy()

            # remove each row/column from the back
            weights[idx] = weights[idx][:int(p * shape[0])]
            weights[idx + 1] = weights[idx + 1][:int(p * shape[0])]
            weights[idx + 2] = weights[idx + 2][:,:int(p * shape[0])]


        shape = weights[idxConvFC].shape
        listFC: Tuple = [i for i in range(int(p * shape[0]), shape[0])]
        FClist = []
        for drop in listFC:
            FClist.extend(range(drop*7*7, drop*7*7 + 7*7))

        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()

        weights[idxConvFC] = weights[idxConvFC][:int(p * shape[0])]
        weights[idxConvFC + 1] = weights[idxConvFC +1][:int(p * shape[0])]
        weights[idxConvFC + 2] = weights[idxConvFC+2][:,:int(p * shape[0])*7*7]

        return weights_to_parameters(weights)










