import math
import sys
import itertools

import clingo

import numpy

import time

import numpy as np
# import janus_swi as janus

# import torch
# import torch.optim

# from torcheval.metrics.aggregation.auc import AUC
# from torcheval.metrics import BinaryAUROC
# from torcheval.metrics import BinaryAUPRC

import argparse


def generate_term(
        atom : 'tuple[str, int]',
        with_prob : bool = True
    ) -> 'list[str]':

    (name, arity) = atom


    if arity == 0:
        term = name
    else:
        term = name + "(" + ','.join(["_"]*arity) + ")"

    return [term + (" : 0.5" if with_prob else "")]



def generate_permutations(
        atom : 'tuple[str, int]',
        with_id : bool,
        with_prob : bool = True,
        index : int = 1, # currently unused
        n_body_atoms : int = 0
    ) -> 'list[str]':
    """
    Generates permutations for the variables for the given atom.
    Example: atom = ("a", 2), index = 1, with_id = False
    Result: a(A1,B1), a(B1,A1)
    If with_idx is true:
    Result: a(Id,A1,B1), a(Id,B1,A1)

    """
    # print(atom)
    (name, arity) = atom
    print(atom)
    # hack - max 3 variables
    if arity > 3:
        print("Arity too big, should be less than 3")
        sys.exit()

    # if arity == 0:
    #     return [name]

    # return [name + "(" + ','.join(["_"]*arity) + ")"]

    if arity == 0:
        if with_id:
            return [name + "(Id)" + (" : 0.5" if with_prob else "")]
        else:
            return [name + (" : 0.5" if with_prob else "")]

    # letters : 'list[str]' = ["A","B","C"][:arity]
    letters : 'list[str]' = ["_","_","_"][:arity]

    atoms : 'list[str]' = []
    for perm in itertools.permutations(letters):
        b = ','.join(perm)
        if with_id:
            b = "Id," + b
        atoms.append(f"{name}({b})")
    
    # # IMPROVE: repeat the same thing but with different orders
    # # only to have more permutations
    # letters : 'list[str]' = ["B", "C", "A"][:arity]

    # for perm in itertools.permutations(letters):
    #     b = ','.join(perm)
    #     if with_id:
    #         b = "Id," + b
    #     atoms.append(f"{name}({b})")
    # # if len(atoms) > 1: 
    # #     print(atoms)

    # #     sys.exit()
    return atoms


def my_log(v : float):
    try:
        return math.log(v)
    except:
        return -13.8155 # math.log(0.000001)
    

# def my_loss(output, expected):
#     # devo scambiare 0 con 1 in prolog - va peggio che MSE
#     # e_swap = torch.ones(tuple(expected.shape)) - expected
#     abs_diff = torch.abs(output - expected)

#     return -torch.sum(torch.log(abs_diff))

from argparser import parse_args
from prolog_interface import PrologInterface
from mixture import Mixture, OptMixture

def main():

    args = parse_args()
    print(args)

    bg = "datasets/" + args.dataset + ".pl"
    # bg = "muta.pl"
    # bg = "lift.pl"
    print(f"Dataset: {bg}")

    train_set = list(map(int, args.train))

    if args.test is None:
        test_set = []
    else:
        test_set = list(map(int, args.test))

    prolog_interface = PrologInterface(bg, train_set, test_set, args.verbosity)

    # if torch.cuda.is_available():
    #     print(f"Using {torch.cuda.get_device_name(0)}")
    #     dev = torch.device("cuda")
    # else:
    #     print("Using CPU")
    #     dev = torch.device("cpu")


    # modeb is a list of atoms that can appear in the body
    # target is a list of target predicates (currently I assume there is
    # only 1)
    # exp is a list of list of 0/1 denoting that the ith example is positive
    # 1 or negative 0
    modeb, target, exp_training, exp_test = prolog_interface.get_modeb_target_and_pos_or_neg_list()   
    
    possible_atoms : 'list[str]' = []
    # possible_atoms_no_id : 'list[str]' = []

    # modeb = modeb + modeb
    for idx, el in enumerate(modeb):
        # possible_atoms.extend(generate_permutations(el, args.with_id))
        possible_atoms.extend(generate_term(el, with_prob=False))
        # possible_atoms_no_id.extend(generate_permutations(el, False))

    # print(possible_atoms)
    # # print(target)

    # sys.exit()

    if len(target) != 1:
        print("Currently only one target is supported")
        sys.exit()
    
    target = target[0]
    # sys.exit()

    # print(list(generate_permutations(target, args.with_id)))
    # print(list(generate_term(target)))

    # sys.exit()

    start_time = time.time()
    mxt_model = Mixture(
        possible_atoms,
        list(generate_term(target, with_prob=True)),
        args.prob_rules,
        # device=dev,
        device=None,
        n_rules_each_program=args.nr,
        max_atoms_in_body=args.nba,
        verbosity=args.verbosity
    )
    end_time = time.time()
    print(f"Generated mixtures in {end_time - start_time} s")
    print(f"Total number of mixtures: {len(mxt_model.programs):_}")

    start_time = time.time()
    learned_programs, probabilities_examples_train, probabilities_examples_test = prolog_interface.compute_parameters_mixtures(
        mxt_model.programs,
        args.train,
        args.test
    )
    end_time = time.time()
    print(f"Learned parameters and filtered in {end_time - start_time} s")

    print(f"Remained mixtures: {len(learned_programs)}")

    if args.verbosity >= 1:
        print("Learned Programs")
        for p in learned_programs:
            print(*p)
        if args.verbosity >= 2:
            print("Probabilities Examples Train")
            print(probabilities_examples_train)
            print("Probabilities Examples Test")
            print(probabilities_examples_test)
    assert len(learned_programs) == len(probabilities_examples_train)
    assert len(learned_programs) <= len(mxt_model.programs), f"found: {len(learned_programs)} expected <= {len(mxt_model.programs)}"
    assert len(probabilities_examples_train) <= len(mxt_model.programs)
    # assert all(len(x) == len(learned_programs[0]) for x in learned_programs)

    print(f"Examples: {len(probabilities_examples_train)}")
    print(f"Mixtures: {len(learned_programs)}")

    # LIFTCOVER può togliere regole dai programmi. In questo caso ottengo programmi uguali
    # in learned_programs, che devo prima eliminare quando faccio apprendimento,
    # per ridurre il numero di mixtures

    om = OptMixture(probabilities_examples_train, exp_training, args.verbosity)
    # loop over multiple iterations
    # for it in range(0,1):
        # print(f"Iteration {it}")
    res = om.find_optimal_weights_mixtures()
    weights = res.x
    ll = res.fun
    sum_weights = sum(weights)
    print("--- Learned Mixtures ---")
    # print(probabilities_examples)
    remaining_programs = 0
    new_parameters_examples : 'list[list[float]]' = []
    idx = 0
    for prog, w in zip(learned_programs, weights):
        if w != 0.0:
            remaining_programs += 1
            print(f"{w/sum_weights}: {prog}")
            new_parameters_examples.append(probabilities_examples_train[idx])
        idx += 1
    print(f"Remaining programs: {remaining_programs} ({remaining_programs/len(weights)})")
    print("Final Cross Entropy E. (training)")
    print(f"{ll}")

    # print("probabilities_examples_test: ")
    # print(probabilities_examples_test)
    if len(exp_test) > 0:
        print("Testing")
        # print(exp_test)
        om.examples = exp_test
        om.par_mixtures = list(np.transpose(np.array(probabilities_examples_test)))
        om.dropout = 0
        om.return_ll = True
        ll_test, roc_test, pr_test = om.compute_ll_roc_examples(weights)
        print(f"LL test: {ll_test}")
        print(f"ROC AUC test: {roc_test}")
        print(f"PR test: {pr_test}")
    

    # print(new_parameters_examples)

    # add loop part
    # if remaining_programs == 1:
    #     break
    # om = OptMixture(new_parameters_examples, exp_training, args.verbosity)



    # for el in par_mixtures:
    #     print(len(el)) # length is number of examples
    # print(len(par_mixtures))
    sys.exit()
    # find the ids of the examples that unify, with prolog
    new_rules : 'list[str]' = []
    target_name = ""
    for idx, a in enumerate(mxt_model.programs):
        for idx_cl, cl in enumerate(a.clauses):
            head, body = cl.clause.split(':-')
            # active(I) -> active(Id,I)
            name, arguments = head.split('(')
            target_name = name # this is the same for all
            if args.prob_rules:
                # probability associated with rules
                at = name + f"_({idx},{idx_cl}," + arguments
            else:
                # no probability
                at = name + f"_({idx}," + arguments
            r = at + ":-" + body

            new_rules.append(r)
    
    # print(new_rules[:10])

    # sys.exit()
    if args.prob_rules:
        examples_matrices = prolog_interface.find_examples_matrices_prob_rules(
            target_name,
            new_rules,
            len(mxt_model.programs),
            mxt_model.n_rules_each_program
        )
        # check correct length
        # the list should be of length # of examples, but I don't have this info here
        print(f"Total number of examples: {len(examples_matrices)}")
        
        # print(examples_matrices)
        # print(mxt_model.n_rules_each_program)
        # print(len(mxt_model.programs))
        # each sublist should be of length # of rules
        # for e in examples_matrices:
        #     assert len(e) == mxt_model.n_rules_each_program, f"expected {mxt_model.n_rules_each_program}, found {len(e)}"
        #     # each sub sublist must be of length # mixtures
        #     for e1 in e:
        #         assert len(e1) == len(mxt_model.programs), f"expected {len(mxt_model.programs)}, found {len(e1)}"
    else:
        examples_matrices = prolog_interface.find_examples_matrices(
            target_name,
            new_rules,
            len(mxt_model.programs)
        )
        # check correct length
        for e in examples_matrices:
            assert len(e) == len(mxt_model.programs), f"expected {len(mxt_model.programs)}, found {len(e)}"


    # expected has shape [#examples, 1]
    # in_matrix has shape [#examples, #mixtures, 1]

    expected = torch.tensor(exp, device=dev)
    if args.verbosity >= 2:
        print(f"expected (transposed for readability): {expected.transpose(0,1)}") # for better readability

    ef = torch.flatten(expected)
    pos = torch.count_nonzero(ef)
    neg = ef.shape[0]

    print(f"Positive examples: {pos}")
    print(f"Negative examples: {neg}")

    if args.prob_rules:
        # need to swap axes: 
        # I get:  [#ex, #rules, #mixtures]
        # I need: [#rules, #ex, #mixtures]
        em_swap = torch.tensor(examples_matrices, device=dev) # .swapaxes(0,1)

        # print(f"em swap shape: {em_swap.shape}")

        # E = [FixA,FixB]
        # A = PA X FixA
        # B = PB X FixB
        # K X (A + B + A*B)
        # assume that there are 2 rules foreach mixture
        fix_matrixes = em_swap
        print(f"fix_matrixes.shape: {fix_matrixes.shape}")
        if args.verbosity >= 2:
            print(f"fix_matrixes (examples): {fix_matrixes}")
        # fixA = em_swap[0]
        # fixB = em_swap[1]
        pass
    else:
        # K X E
        in_matrix = torch.tensor(examples_matrices, device=dev)
    # in_b = torch.tensor(eb)

    # print(expected.shape)
    # print(in_matrix.shape)

    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.KLDivLoss()
    # loss_fn = torch.nn.L1Loss(reduction="sum")
    # , weight_decay=0.2 for L2
    # optimizer = torch.optim.SGD(mxt_model.parameters(), lr=0.1) # non va bene
    optimizer = torch.optim.Adam(mxt_model.parameters(), lr=args.lr, weight_decay=args.l2)
    # optimizer = torch.optim.RMSprop(mxt_model.parameters(), lr=0.1)


    # print(mxt_model.k)
    loss_values : 'list[float]' = []
    ll_values : 'list[float]' = []
    auc_scores : 'list[float]' = []
    roc_scores : 'list[float]' = []
    pr_scores : 'list[float]' = []
    
    if args.scores:
        # metric_auc = AUC()
        metric_roc = BinaryAUROC(device=dev)
        metric_pr = BinaryAUPRC(device=dev)

    for ep in range(0, args.epochs):
        if args.prob_rules:
            output = mxt_model(fix_matrixes)
        else:
            output = mxt_model((in_matrix))

        if args.verbosity >= 2:
            print(f"output.shape: {output.shape}")
        loss = loss_fn(output, expected.transpose(0,1)[0])
        # loss = my_loss(output, expected.transpose(0,1)[0])

        pe = []
        for ex, p in zip(expected, output):
            if ex == 1.0:
                pe.append(p)
            else:
                pe.append(1-p)
        
        ll_values.append(sum(map(my_log, pe)))
        
        
        # if args.scores and ep % 50 == 0:
        #     ot = output #.transpose(0,1)
        #     et = expected.transpose(0,1)[0]

        #     # print(ot)
        #     # print(et)
        #     # metric_auc.update(ot, et)
        #     metric_roc.update(ot, et)
        #     metric_pr.update(ot, et)
            
        #     # auc_score = metric_auc.compute()
        #     roc_score = metric_roc.compute()
        #     pr_score = metric_pr.compute()

        #     # auc_scores.append(float(auc_score.item()))
        #     roc_scores.append(float(roc_score.item()))
        #     pr_scores.append(float(pr_score.item()))

        # s = torch.zeros(1)
        # for mp in mxt_model.parameters():
        #     s += torch.sum(torch.abs(mp))

        # l1_reg = 0.1 * s
        # print(f"Output: {output}, Loss: {loss.item()}")
        # if SMALL:
        # else:
        if ep % 100 == 0:
            if args.verbosity >= 2:
                print(f"It: {ep}, Output: {output}, Loss: {loss.item()}")
            else:
                print(f"It: {ep}, Loss: {loss.item()}")
            # print(auc_score)
        
        loss_values.append(loss.item())
        optimizer.zero_grad()

        # loss = loss + l1_reg
        loss.backward()
        optimizer.step()
    
    # if args.scores:
    #     # metric_auc.reset()
    #     metric_roc.reset()
    #     metric_pr.reset()
    # print(mxt_model.k)

    # args.verbosity = 3
    if args.verbosity >= 2:
        print("RESULT")
        print("mxt_model.k")
        print(mxt_model.k)
        print("torch.sigmoid(mxt_model.k)")
        print(torch.sigmoid(mxt_model.k))
        print("mxt_model.a.transpose(0,1)")
        print(mxt_model.a.transpose(0,1))
        print("torch.sigmoid(mxt_model.a) (transposed for readability)")
        print(torch.sigmoid(mxt_model.a.transpose(0,1)))
        if mxt_model.n_rules_each_program >= 2:
            print("torch.sigmoid(mxt_model.b) (transposed for readability)")
            print(torch.sigmoid(mxt_model.b.transpose(0,1)))
        if mxt_model.n_rules_each_program >= 3:
            print("torch.sigmoid(mxt_model.c) (transposed for readability)")
            print(torch.sigmoid(mxt_model.c.transpose(0,1)))

    # print("Final output with cut at sigmoid(v) < 0.05")
    # provo a mantenere solamente quelli con prob > 0.05
    # questo non va proprio bene, dovrei proprio togliere il valore. Come è scritto
    # ora considera 0.5 per la mixture. Metto -6 che è praticamente 0
    # mxt_model.k = torch.nn.Parameter(torch.tensor([v if torch.sigmoid(v) > 0.05 else -6 for v in mxt_model.k]))
    
    # mxt_model.verbosity = 3
    print("--- FINAL COMPUTATION --")
    if args.prob_rules:
        final_output = mxt_model(fix_matrixes)
    else:
        final_output = mxt_model((in_matrix)) 
    
    # print("final output")
    # print(final_output)

    # sys.exit()
    # # count bins
    # count_gt_005 = 0
    # count_gt_05 = 0
    # count_gt_09 = 0

    # for v in mxt_model.k:
    #     val = torch.sigmoid(v)
    #     if val >= 0.05:
    #         count_gt_005 += 1
    #         if val >= 0.5:
    #             count_gt_05 += 1
    #             if val >= 0.9:
    #                 count_gt_09 += 1

    # n_mixtures = len(mxt_model.k)
    # print(f"# >= 0.05: {count_gt_005}/{n_mixtures} = {str((count_gt_005/n_mixtures) * 100)[:5]} %")
    # print(f"# >= 0.5: {count_gt_05}/{n_mixtures} = {str((count_gt_05/n_mixtures) * 100)[:5]} %")
    # print(f"# >= 0.9: {count_gt_09}/{n_mixtures} = {str((count_gt_09/n_mixtures) * 100)[:5]} %")

    # print(ll_values)

    # if args.scores:
    #     # print(auc_scores)
    #     print(f"roc_score: {roc_score}")
    #     print(f"pr_score: {pr_score}")

    pe = []
    for ex, p in zip(expected, final_output):
        print(ex, p)
        if ex == 1.0:
            pe.append(p)
        else:
            pe.append(1-p)

    # print(pe)
    print(f"Final LL: {sum(map(my_log, pe))}")

    # if args.scores:
    #     # print(expected.transpose(0,1)[0])
    #     metric_roc.update(final_output, expected.transpose(0,1)[0])
    #     roc_score = metric_roc.compute()
    #     # print(roc_score)

    if mxt_model.n_rules_each_program == 1:
        for idx, v in enumerate(mxt_model.k):
            # print(f"Mixture {idx}")
            # print(f"\tweight: {torch.sigmoid(v).item()/torch.sum(torch.sigmoid(mxt_model.k))}")
            # print(f"\t{torch.sigmoid(mxt_model.a[idx]).item()}::{mxt_model.programs[idx].clauses[0].clause}")
            # print(torch.sigmoid(mxt_model.a[idx]).item())
            # print(torch.sigmoid(v).item())
            # print(torch.sum(torch.sigmoid(mxt_model.k)))
            num = torch.sigmoid(v).item()
            p = torch.sigmoid(mxt_model.a[idx]).item()
            den = torch.sum(torch.sigmoid(mxt_model.k)).item()
            print(f"\t{num / den} -> {p}::{mxt_model.programs[idx].clauses[0].clause}")

    # amax = torch.argmax(mxt_model.k)

    # print("Program with the highest weight")
    # print(f"weight: {torch.max(mxt_model.k)}")
    # print(f"{torch.sigmoid(mxt_model.a[amax]).item()}::{mxt_model.programs[amax].clauses[0].clause}")
    # try:
    #     print(f"{torch.sigmoid(mxt_model.b[amax]).item()}::{mxt_model.programs[amax].clauses[1].clause}")
    #     if args.nr > 2:
    #         print(f"{torch.sigmoid(mxt_model.c[amax]).item()}::{mxt_model.programs[amax].clauses[2].clause}")
    # except:
    #     pass

    

    # import matplotlib.pyplot as plt

    # plt.plot(ll_values)
    # plt.ylabel('LL')
    # plt.show()


if __name__ == "__main__":
    main()