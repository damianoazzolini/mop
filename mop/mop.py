import math
import numpy as np
import random
import sys
import time

from scipy.special import expit

from .argparser import parse_args
from .prolog_interface import PrologInterface
from .mixture import MixtureGenerator, OptMixture



def generate_term_prob(
        # atom : 'tuple[str, int]',
        atom : 'str',
        with_prob : bool = True
    ) -> 'list[str]':
    """
    Generate terms.
    """

    # (name, arity) = atom


    # if arity == 0:
    #     term = name
    # else:
    #     term = name + "(" + ','.join(["_"]*arity) + ")"

    return [atom + (" : 0.5" if with_prob else "")]


def main():
    """
    Main method.
    """
    args = parse_args()
    print(args)

    if args.dataset:
        bg = "datasets/" + args.dataset + ".pl"
    else:
        bg = args.filename
    print(f"Dataset: {bg}")

    train_set = list(map(int, args.train))

    if args.test is None:
        test_set = []
    else:
        test_set = list(map(int, args.test))

    prolog_interface = PrologInterface(bg, train_set, test_set, args.verbosity)


    # modeb is a list of atoms that can appear in the body
    # target is a list of target predicates (currently I assume there is
    # only 1)
    # exp is a list of list of 0/1 denoting that the ith example is positive
    # 1 or negative 0
    possible_atoms, target, exp_training, exp_test = prolog_interface.get_modeb_target_and_pos_or_neg_list()

    k0 = math.comb(len(possible_atoms), args.nba)
    n_mixtures = math.comb(k0 * len(target), args.nr)

    print(f"Generating {n_mixtures:_} mixtures")

    targets = []
    for t in target:
        targets.append(generate_term_prob(t, with_prob=True))

    start_time = time.time()
    mxt_model = MixtureGenerator(
        possible_atoms,
        targets,
        n_rules_each_program=args.nr,
        max_atoms_in_body=args.nba,
        verbosity=args.verbosity
    )
    end_time = time.time()
    print(f"Generated mixtures in {end_time - start_time} s")
    print(f"Total number of mixtures: {len(mxt_model.programs):_}")

    if args.nm != -1:
        # sample args.nm mixtures from the generated
        ns = min(int(args.nm), len(mxt_model.programs))
        mxt_model.programs = random.sample(mxt_model.programs, ns)
        print(f"Considering {len(mxt_model.programs)} random programs")
    else:
        print("Considering all mixtures")

    start_time = time.time()
    learned_programs, probabilities_examples_train, probabilities_examples_test = prolog_interface.compute_parameters_mixtures(
        mxt_model.programs,
        args.train,
        args.test
    )
    end_time = time.time()
    print(f"Learned parameters and filtered in {end_time - start_time} s")

    print(f"Remained mixtures: {len(learned_programs)}")
    if len(learned_programs) == 0:
        return
    

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

    cutoff_prob = math.pow(10, -args.cut)

    # LIFTCOVER may remove clauses from programs. I may get equal program.

    om = OptMixture(
        probabilities_examples_train,
        exp_training,
        args.maxfun,
        args.gamma,
        args.cut,
        args.verbosity
    )
    # loop over multiple iterations
    # for it in range(0,1):
        # print(f"Iteration {it}")
    res = om.find_optimal_weights_mixtures()
    weights = expit(res.x)
    ll = res.fun
    sum_weights = sum(weights)
    # sum_weights = sum([w for w in weights if w > cutoff_prob])
    print(f"--- Learned Mixtures (pruned below {cutoff_prob}) ---")
    # print(probabilities_examples)
    remaining_programs = 0
    # new_parameters_examples : 'list[list[float]]' = []
    # idx = 0
    for prog, w in zip(learned_programs, weights):
        if w > cutoff_prob:
            remaining_programs += 1
            if args.verbosity >= 1:
                print(f"{w}: {prog}")
        else:
            sum_weights -= w
            # new_parameters_examples.append(probabilities_examples_train[idx])
        # idx += 1
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
        ll_test, roc_test, pr_test = om.compute_ll_roc_examples(weights, sum_weights)
        print(f"LL test: {ll_test}")
        print(f"ROC AUC test: {roc_test}")
        print(f"PR test: {pr_test}")    

    # import matplotlib.pyplot as plt

    # plt.plot(ll_values)
    # plt.ylabel('LL')
    # plt.show()


if __name__ == "__main__":
    main()