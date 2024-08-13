import itertools
import math
import numpy as np
import time
import sklearn.metrics
import sys

from scipy.optimize import minimize, show_options

from data_structures import Program, Clause


def my_log(v : float):
    try:
        return math.log(v)
    except:
        return -13.8155 # math.log(0.000001)

class OptMixture():
    """
    Class to minimize the NLL of the examples given the programs,
    with scipy, not torch
    """
    def __init__(self,
            parameters_mixtures : 'list[list[float]]',
            examples : 'list[float]',
            verbosity : int = 0,
            gamma : float = 0.1,
            l1 : bool = False,
            l2 : bool = False,
            dropout : float = 0,
            return_ll : bool = False
        ) -> None:
        # each list is the prob of the fixed example in the program i
        self.n_programs = len(parameters_mixtures)
        self.par_mixtures = list(np.transpose(np.array(parameters_mixtures)))
        self.examples : 'list[float]' = examples # 0 negative, 1 positive 
        self.verbosity = verbosity
        self.gamma : float = gamma
        self.l1 : bool = False
        self.l2 : bool = False
        self.dropout : float = dropout # 0 if no dropout
        self.return_ll : bool = return_ll
        self.E = np.array([self.examples])
        self.M = np.array(self.par_mixtures)
        # iterations counter
        self.it = 0

        # print(self.par_mixtures)

    def compute_ll_roc_examples(self, weights_mixtures) -> 'tuple[float,float,float]':
        """
        Computation of the LL and ROC.
        Similar to the function to minimize but here to simplify
        that function.
        """
        normalizing_factor = sum(weights_mixtures)
        prob_examples = []
        ll_examples = []

        for e, par_mixture in zip(self.examples, self.par_mixtures):
            prob_i = 0
            for k, p in zip(weights_mixtures, par_mixture):
                prob_i += k*p
            
            # print(f"Pre: {prob_i}")
            # cross_entropy_i = -1 * e * my_log(prob_i/normalizing_factor) - (1 - e) * my_log(1 - prob_i/normalizing_factor)

            ll = my_log(prob_i/normalizing_factor) if e == 1 else my_log(1-prob_i/normalizing_factor)

            prob_examples.append(prob_i/normalizing_factor)
            ll_examples.append(ll)

        computed_roc_auc_score = sklearn.metrics.roc_auc_score(self.examples, prob_examples)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(self.examples, prob_examples)
        # Compute area under the curve
        pr_auc = sklearn.metrics.auc(recall, precision)

        return sum(ll_examples), float(computed_roc_auc_score), float(pr_auc)

    def compute_cross_entropy_error_examples_matrix(self, W) -> float:
        """
        Computation of the cross entropy error of the examples via matrix multiplication
        """
        # self.examples: 1 x N
        # weights_mixtures: 1 X M
        # self.par_mixtures: N x M

        # W = np.matrix([weights_mixtures]).transpose() # this is deprecated
        NORMALIZING_FACTOR = np.sum(W) + 10e-8 # to avoid 0

        # -1 * e * my_log(prob_i/normalizing_factor)
        MW = np.dot(self.M, W)
        D = MW / NORMALIZING_FACTOR
        # print(MW / NORMALIZING_FACTOR)
        prob_tmp = np.where(D > 0.000001, D, 0.000001)
        L1 = np.where(D > 0.000001, np.log(prob_tmp), -13.8155)
        # print(L1.shape)
        # L1E = -1 * E * L1
        L1E = np.dot(-1, self.E) @ L1
        
        # - (1 - e) * my_log(1 - prob_i/normalizing_factor) 
        # L2 = np.log(1 - D) 
        D1 = 1 - D
        prob_tmp = np.where(D1 > 0.000001, D1, 0.000001)
        L2 = np.where(D1 > 0.000001, np.log(prob_tmp), -13.8155)
        # L2E = -(1 - E) * L2
        L2E = np.dot(-1, (1- self.E)) @ L2

        R = L1E + L2E

        if self.it % 5000 == 0:
            print(f"Evaluation {self.it}, sum weights: {NORMALIZING_FACTOR}, R: {R}")
        self.it += 1

        # print(R.shape)
        return R


    def find_optimal_weights_mixtures(self):
        """
        Optimization process.
        """
        print("Starting optimization process")
        # print("Parameters mixtures")
        # print(self.par_mixtures)
        import random
        # weights_mixtures : 'list[float]' = []
        # for i in range(self.n_programs):
            # weights_mixtures.append(random.random())
        weights_mixtures = [0.5]*self.n_programs
        weights_mixtures = np.array([weights_mixtures]).reshape((self.n_programs, 1))
        # print(weights_mixtures)
        print(f"weights_mixtures.shape (W) (1 x M): {weights_mixtures.shape}")
        print(f"examples.shape (E) (1 x N): {self.E.shape}")
        print(f"parameter_mixtures.shape (M) (N x M): {self.M.shape}")

        assert weights_mixtures.shape[0] == self.M.shape[1]
        assert self.E.shape[1] == self.M.shape[0]
        
        start_time = time.time()
        # obj_and_grad = jit(value_and_grad(self.compute_cross_entropy_error_examples_matrix_jax))
        res = minimize(
            self.compute_cross_entropy_error_examples_matrix,
            # self.compute_cross_entropy_error_examples_matrix_jax,
            # obj_and_grad,
            # self.compute_negative_ll_examples,
            weights_mixtures,
            bounds=[(0,1)]*self.n_programs,
            # options={'disp': True}
            # jac=True,
            # options={'maxfun': 10e10} # with default values this may be better, less overfitting
            # method="SLSQP"
        )
        print(res)
        end_time = time.time()
        return res


class MixtureGenerator():
    """
    Class containing a mixture of programs.
    """
    def __init__(self,
            possible_atoms : 'list[str]',
            targets : 'list[str]',
            # examples : 'list[Example]',
            # device : torch.device,
            n_rules_each_program : int = 2,
            max_atoms_in_body : int = 3,
            verbosity : int = 0,
            toss_probability : int = 0
        ) -> None:
        
        self.possible_atoms = possible_atoms
        self.targets = targets # the head of the relation, list to handle arity > 1
        # self.examples = examples
        self.n_rules_each_program = n_rules_each_program
        self.max_atoms_in_body = max_atoms_in_body
        self.verbosity = verbosity
        self.toss_probability = toss_probability

        self.programs : 'list[Program]' = []

        self.generate_programs()

        if self.verbosity >= 3:
            for p in self.programs:
                print(p)

    def generate_programs(self):
        """
        Generates all the possible programs.
        """
        possible_rules : 'list[str]' = []
        # here fixed 1 cycle
        # for idx in range(self.max_atoms_in_body, self.max_atoms_in_body + 1):
        # generate all possible combinations of atoms of fixed length
        comb = list(itertools.combinations(self.possible_atoms, self.max_atoms_in_body))
        # generate all possible rules
        # print(*comb)

        prods = []
        # print(self.targets)
        for t in self.targets:
            prods.extend(list(itertools.product(t, comb)))
        # print(*prod)
        # print(prods)
        for i, p in enumerate(prods):
            # print(f"-- {p}")
            b = ','.join(p[1])
            r = f"{p[0]} :- {b}."
            possible_rules.append(r)
                
        # generate all possible programs by gluing the specified
        # number of rules - from 1 to the specified number
        # for n_rules in range(1, self.n_rules_each_program + 1):
        # exactly the specified number
        for idx, prog in enumerate(itertools.combinations(possible_rules, self.n_rules_each_program)):
            # print(f"-> {prog}")
            # if idx % 50 == 0:
            #     print(f"Adding program {idx}")
            lc = []
            for c in prog:
                lc.append(Clause(c, self.targets))
            self.programs.append(Program(lc))
