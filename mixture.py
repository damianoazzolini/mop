import itertools
# import torch

from data_structures import Program, Clause

import math
import time
import numpy as np
import random

import sys

import sklearn.metrics

from scipy.optimize import minimize

import jax.numpy as jnp
from jax import jit, value_and_grad

def inverse_sigmoid(y):
    return math.log(y / (1 - y))


def my_log(v : float):
    # print(f"v : {v}")
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
        # print(self.examples, self.par_mixtures)
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

    def compute_negative_ll_examples_matrix(self, W) -> float:
        """
        Computation of the NLL of the examples via matrix multiplication
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


    def compute_negative_ll_examples(self, weights_mixtures) -> float:
        """
        Computation of the NLL of the examples
        """
        normalizing_factor = sum(weights_mixtures)
        prob_examples = []
        ll_examples = []
        # print(len(self.examples), len(self.par_mixtures), len(weights_mixtures))
        # print(self.examples, self.par_mixtures, weights_mixtures)

        for e, par_mixture in zip(self.examples, self.par_mixtures):
            # print(e,par_mixture, weights_mixtures) # <---- TODO: check this
            # sum k_i * p_i
            prob_i = 0
            for k, p in zip(weights_mixtures, par_mixture):
                # if random.random() > self.dropout:
                prob_i += k*p
            
            # print(f"Pre: {prob_i}")
            cross_entropy_i = -1 * e * my_log(prob_i/normalizing_factor) - (1 - e) * my_log(1 - prob_i/normalizing_factor)
            # print(f"Post: {prob_i}")

            # ll = my_log(prob_i/normalizing_factor) if e == 1 else my_log(1-prob_i/normalizing_factor)
            # pe = prob_i/normalizing_factor
            # # print(f"pe: {pe}")
            # if e == 1:
            #     ll = my_log(pe)
            # else:
            #     ll = my_log(1 - pe)

            # prob_i_for_ll = prob_i/normalizing_factor if e == 1 else (1-prob_i/normalizing_factor)
            # prob_i = prob_i/normalizing_factor

            prob_examples.append(cross_entropy_i)
            # ll_examples.append(ll)

            # print(sum(prob_examples),sum(ll_examples))

        # print(prob_examples)
        # print(sum(prob_examples))
        # import sys
        # sys.exit()
        
        # print(prob_examples)
        # print(ll_examples)
        # ll_examples = sum(map(my_log, prob_examples))
        # sum_ll_examples = sum(ll_examples)
        if self.it % 500 == 0:
            print(f"Evaluation {self.it}, sum weights: {normalizing_factor}")
            if self.verbosity >= 2:
                print(f"\tweights_mixtures: {weights_mixtures}\n\tprob: {prob_examples}\n\tLL: {ll_examples}\n\tSum LL: {sum(ll_examples)}")
        self.it += 1
        # print(ll_examples)
        
        # TODO: non funziona
        l1 = 0
        l2 = 0
        if self.l1:
            l1 = normalizing_factor * self.gamma
        if self.l2:
            w = 0
            for weight in weights_mixtures:
                w = w + weight**2
            l2 = w * (self.gamma/2)

        # print((-sum_ll_examples, l1, l2))

        # import sys
        # sys.exit()

        # # return -sum_ll_examples + l1 + l2
        # if self.return_ll:
        #     return sum(ll_examples)
        
        return sum(prob_examples) + l1 + l2
    

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
        print(f"weights_mixtures.shape (W): {weights_mixtures.shape}")
        print(f"examples.shape (E): {self.E.shape}")
        print(f"parameter_mixtures.shape (M): {self.M.shape}")

        assert weights_mixtures.shape[0] == self.M.shape[1]
        assert self.E.shape[1] == self.M.shape[0]
        
        start_time = time.time()
        # obj_and_grad = jit(value_and_grad(self.compute_negative_ll_examples_matrix_jax))
        res = minimize(
            self.compute_negative_ll_examples_matrix,
            # self.compute_negative_ll_examples_matrix_jax,
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

# class Mixture(torch.nn.Module):
class Mixture():
    """
    Class containing a mixture of programs.
    """
    def __init__(self,
            possible_atoms : 'list[str]',
            target : 'list[str]',
            # examples : 'list[Example]',
            prob_rules : bool,
            # device : torch.device,
            device : None,
            n_rules_each_program : int = 2,
            max_atoms_in_body : int = 3,
            verbosity : int = 0,
            toss_probability : int = 0
        ) -> None:
        
        super(Mixture, self).__init__()
        self.possible_atoms = possible_atoms
        self.target = target # the head of the relation, list to handle arity > 1
        self.prob_rules = prob_rules # True if probabilities are also associated with rules, False otherwise
        self.dev = device
        # self.examples = examples
        self.n_rules_each_program = n_rules_each_program
        self.max_atoms_in_body = max_atoms_in_body
        self.verbosity = verbosity
        self.toss_probability = toss_probability

        self.programs : 'list[Program]' = []

        # TODO: provare ad aggiungere una mixture che serve a far sommare
        # ad 1 i valori delle rimanenti mixture. In questo modo, forse riesco
        # ad avere maggior flessibilità

        self.generate_programs()

        # if SMALL:
        #     print("Using 3 programs")
        #     self.programs = self.programs[:2] # reduced number of programs
        if self.verbosity >= 1:
            for p in self.programs:
                print(p)
        # print(*self.programs)

        # import sys
        # sys.exit()

    #     # weights for the mixtures
    #     self.in_k = [0.6225]*(len(self.programs) + self.toss_probability) # 0.6225 is sigmoid(0.5)
    #     # v = np.random.rand(len(self.programs))
    #     # self.in_k = list(map(inverse_sigmoid, v))
    #     self.k = torch.nn.Parameter(torch.tensor(self.in_k, device=self.dev))
    #     print(f"self.k.shape: {self.k.shape}")

    #     # self.slack = torch.nn.Parameter(torch.tensor([0]))

    #     # no weights on clauses
    #     self.in_a = [[-6.90]]*(len(self.programs) + self.toss_probability) # sigmoid(-6.90) ~ 0
    #     self.a = torch.nn.Parameter(torch.tensor(self.in_a, device=self.dev))
    #     print(f"self.a.shape: {self.a.shape}")

    #     # self.fix_a = self.generate_fix_column(0)
    #     # print(self.fix_a)
    #     # print(len(self.fix_a))
    #     # assert len(self.fix_a) == len(self.programs)

    #     if prob_rules:
    #         if self.n_rules_each_program >= 2:
    #             self.in_b = [[0.6225]]*len(self.programs)
    #             self.b = torch.nn.Parameter(torch.tensor(self.in_b, device=self.dev))
    #             print(f"self.b.shape: {self.b.shape}")
            
    #         if self.n_rules_each_program >= 3:
    #             self.in_c = [[0.6225]]*len(self.programs)
    #             self.c = torch.nn.Parameter(torch.tensor(self.in_c, device=self.dev))
    #             print(f"self.c.shape: {self.c.shape}")

    #     # print(f"{len(self.programs)} programs")
    #     # for p in self.programs:
    #     #     print(p)

    # def forward(self, examples):
    #     if self.prob_rules:
    #         # E = [FixA,FixB]
    #         # [FixA,FixB] are needed to avoid changing the probability
    #         # associated with a rule that actually does not contribute to
    #         # the probability
    #         # A = PA X FixA
    #         # B = PB X FixB
    #         # K X (A + B + A*B)
    #         if self.verbosity >= 3:
    #             print(f"examples.shape: {examples.shape}")
    #             print(f"examples: {examples}")

    #         if self.n_rules_each_program == 1:
    #             # print(examples)
    #             # print(examples.shape)

    #             # import sys
    #             # sys.exit()
    #             fix_a = examples.transpose(0,1)[0]
    #             A = torch.sigmoid(self.a) * fix_a.transpose(0,1)
    #             # A = torch.clamp(self.a,0,1) * fix_a.transpose(0,1)
    #             res = torch.matmul(torch.sigmoid(self.k), A) / torch.sum(torch.sigmoid(self.k))
    #             # res = torch.matmul(torch.clamp(self.k,0,1), A) / torch.sum(torch.clamp(self.k,0,1))
    #             if self.verbosity >= 3:
    #                 print(f"fix_a.shape: {fix_a.shape}")
    #                 print(f"fix_a: {fix_a}")
    #                 print("self.a")
    #                 print(self.a)
    #                 print("A (torch.sigmoid(self.a) * fix_a.transpose(0,1))")
    #                 print(A)
    #                 print("torch.sum(torch.sigmoid(self.k))")
    #                 print(torch.sum(torch.sigmoid(self.k)))
    #                 print("torch.matmul(torch.sigmoid(self.k), A)")
    #                 print(torch.matmul(torch.sigmoid(self.k), A) )
    #                 print(f"torch.matmul(torch.sigmoid(self.k), A) / torch.sum(torch.sigmoid(self.k)).shape: {res.shape}")
    #                 print(f"torch.matmul(torch.sigmoid(self.k), A) / torch.sum(torch.sigmoid(self.k))")
    #                 print(res)
    #             return res
            
    #         if self.n_rules_each_program == 2:
    #             fix_a, fix_b = examples.transpose(0,1)
    #             A = torch.sigmoid(self.a) * fix_a.transpose(0,1)
    #             # A = self.a * fix_a
    #             B = torch.sigmoid(self.b) * fix_b.transpose(0,1)
    #             # B = self.b * fix_b
    #             res = torch.matmul(self.k, (A + B - A*B)) / torch.sum(self.k)
    #             if self.verbosity >= 3:
    #                 print(f"fix_a.shape: {fix_a.shape}")
    #                 print(f"fix_a: {fix_a}")
    #                 print(f"fix_b.shape: {fix_b.shape}")
    #                 print(f"fix_b: {fix_b}")
    #                 print(f"fix_k.shape: {self.k.shape}")
    #                 print(f"fix_k: {self.k}")
    #                 print("A (torch.sigmoid(self.a) * fix_a.transpose(0,1))")
    #                 print(A)
    #                 print("B")
    #                 print(B)
    #                 print("torch.matmul(self.k, (A + B - A*B)).shape")
    #                 print(f"{torch.matmul(self.k, (A + B - A*B)).shape}")
    #                 print("torch.matmul(self.k, (A + B - A*B))")
    #                 print(f"{torch.matmul(self.k, (A + B - A*B))}")
    #                 print("torch.matmul(self.k, (A + B - A*B)) / torch.sum(self.k).shape")
    #                 print(res.shape)
    #                 print("torch.matmul(self.k, (A + B - A*B)) / torch.sum(self.k)")
    #                 print(res)
    #             # print(f"A.shape: {A.shape}")
    #             # print(f"B.shape: {B.shape}")
    #             return res
            
    #         if self.n_rules_each_program == 3:
    #             # length 3
    #             fix_a, fix_b, fix_c = examples.transpose(0,1)
    #             A = torch.sigmoid(self.a) * fix_a.transpose(0,1)
    #             # A = self.a * fix_a
    #             B = torch.sigmoid(self.b) * fix_b.transpose(0,1)
    #             # B = self.b * fix_b
    #             C = torch.sigmoid(self.c) * fix_c.transpose(0,1)
    #             # C = self.c * fix_c

    #             AB = A*B
    #             AC = A*C
    #             BC = B*C

    #             ABC = A*B*C
    #             return torch.matmul(self.k, (A + B + C - AB - AC - BC + ABC)) / torch.sum(self.k)

    #     return torch.matmul(torch.sigmoid(self.k), (examples)) / torch.sum(torch.sigmoid(self.k))
    #     # return torch.sigmoid(torch.matmul(self.k, (examples)) / torch.sum(self.k)) # valori migliori

    def generate_programs(self):
        """
        Generates all the possible programs.
        """
        possible_rules : 'list[str]' = []
        # here fixed 1 cycle
        # for idx in range(self.max_atoms_in_body, self.max_atoms_in_body + 1):
        # generate all possible combinations of atoms of fixed length
        comb = itertools.combinations(self.possible_atoms, self.max_atoms_in_body)
        # generate all possible rules
        # print(*comb)
        prod = itertools.product(self.target, comb)
        # print(*prod)
        for i, p in enumerate(prod):
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
                lc.append(Clause(c, self.target))
            self.programs.append(Program(lc))

    # def generate_fix_column(self, clause_number : int) -> 'list[list[list[float]]]':
    #     """
    #     Returns a list of length #examples where each element
    #     is 1 if the clause number clause_number cover the example i,
    #     0 otherwise
    #     """
    #     v : 'list[list[list[float]]]' = []
    #     for ex in self.examples:
    #         vt : 'list[list[float]]' = []
    #         for p in self.programs:
    #             # print(p.clauses[clause_number].idx_examples)
    #             if ex.idx in p.clauses[clause_number].idx_examples:
    #                 vt.append([1.0])
    #             else:
    #                 vt.append([0.0])
    #         v.append(vt)
    #     # print(f"v (len: {len(v)}): {v}")
    #     return v

    # def compute_neg_ll_examples(self) -> float:
    #     """
    #     Computes the probability of the examples by considering
    #     the weights of each mixture component.
    #     P(example | mixture) = sum w_i * P(example | program_i) / sum w_i
    #     """
    #     den = sum(self.k)
    #     # print(den)
    #     prob_examples : 'list[float]' = []
    #     for ex in self.examples:
    #         prob_ex = 0
    #         for idx_program, p in enumerate(self.programs):
    #             # it_models = False
    #             prob_ex += self.k[idx_program] * (1 - (1 - self.k[idx_program][0])*(1 - self.k[idx_program][1]))
             
    #         # print(prob_ex)
    #         prob_ex = prob_ex / den # normalize
    #         # print(f"prob ex {ex.idx} (pos: {ex.positive}): {prob_ex}")
    #         if ex.positive:
    #             prob_examples.append(prob_ex)
    #         else:
    #             prob_examples.append(1 - prob_ex)

    #     ll_examples = sum(map(math.log, prob_examples))

    #     return -ll_examples


    def pretty_print_results(self, final_parameters) -> None:
        """
        Pretty prints the results.
        """
        weights_mixtures, fwc, swc = [final_parameters[start::3] for start in range(3)] # split into 3 parts
        
        passed_programs = [p for i, p in enumerate(self.programs) if weights_mixtures[i] != 0]
        print(f"Mixture passed: {len(passed_programs)}/{len(weights_mixtures)}")
        