import math
import numpy as np
import random
import sys
import time

from scipy.special import expit

from .argparser import parse_args
from .prolog_interface import PrologInterface
from .mixture import MixtureGenerator, OptMixture
from .data_structures import Program, Clause



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

    if args.samples == -1:
        k0 = math.comb(len(possible_atoms), args.nba)
        n_mixtures = math.comb(k0 * len(target), args.nr)
        print(f"Generating {n_mixtures:_} mixtures")
    else:
        print(f"Sampling {args.samples:_} mixtures")
        

    targets = []
    for t in target:
        targets.append(generate_term_prob(t, with_prob=True))

    mxt_model = MixtureGenerator(
        possible_atoms,
        targets,
        n_rules_each_program=args.nr,
        max_atoms_in_body=args.nba,
        samples=args.samples,
        verbosity=args.verbosity
    )

    if args.samples == -1:
        mxt_model.generate_all_programs()
    else:
        mxt_model.sample_programs()
    print(f"Total number of mixtures: {len(mxt_model.programs):_}")

    om = OptMixture(
        [],
        exp_training,
        args.gamma,
        args.l1,
        args.l2,
        args.cut,
        args.verbosity
    )

    previously_sampled : 'list[Program]' = []
    
    previous_cross_ee : float = 10e10
    current_cross_ee : float = 10e10
    it : int = 0 # number of iterations
    cee_list : 'list[float]' = []

    while (previous_cross_ee >= current_cross_ee) and it < 100:
        it += 1

        if args.samples != -1:
            print(f"Iteration {it}")
        
        considered_programs : 'list[Program]' = previously_sampled + mxt_model.programs
        previously_sampled = []
        
        start_time = time.time()
        learned_programs, probabilities_examples_train, probabilities_examples_test = prolog_interface.compute_parameters_mixtures(
            considered_programs,
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
            if args.verbosity >= 5:
                print("Probabilities Examples Train")
                print(probabilities_examples_train)
                print("Probabilities Examples Test")
                print(probabilities_examples_test)
        assert len(learned_programs) == len(probabilities_examples_train)
        if args.samples == -1:
            assert len(learned_programs) <= len(mxt_model.programs), f"found: {len(learned_programs)} expected <= {len(mxt_model.programs)}"
            assert len(probabilities_examples_train) <= len(mxt_model.programs)
        # assert all(len(x) == len(learned_programs[0]) for x in learned_programs)

        print(f"Examples: {len(probabilities_examples_train)}")
        print(f"Mixtures: {len(learned_programs)}")

        cutoff_prob = math.pow(10, -args.cut)

        # LIFTCOVER may remove clauses from programs. I may get equal program.
        
        om.n_programs = len(probabilities_examples_train)
        om.par_mixtures = list(np.transpose(np.array(probabilities_examples_train)))
        om.M = np.array(om.par_mixtures)

        res = om.find_optimal_weights_mixtures()
        weights = expit(res.x)

        previous_cross_ee = current_cross_ee
        current_cross_ee = res.fun
        cee_list.append(current_cross_ee)

        # ll = res.fun
        sum_weights = sum(weights)

        print(f"--- Learned Mixtures (pruned below {cutoff_prob}) ---")
        remaining_programs = 0
        for prog, w in zip(learned_programs, weights):
            if w > cutoff_prob:
                remaining_programs += 1
                previously_sampled.append(Program(list(map(Clause,prog))))
                if args.verbosity >= 1:
                    print(f"{w}: {prog}")
            else:
                sum_weights -= w

        print(f"Remaining programs: {remaining_programs} ({remaining_programs/len(weights)})")
        print("Final Cross Entropy E. (training)")
        print(f"{current_cross_ee}")

        # sample new programs
        if args.samples != -1:
            mxt_model.sample_programs()
        else:
            break

        # sys.exit()

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

    if args.samples:
        print("CEE iterations during training")
        for i in range(0, len(cee_list)):
            if i == 0:
                print(f"{i}: {cee_list[i]}")
            else:
                print(f"{i}: {cee_list[i]} ({cee_list[i] - cee_list[i-1]})")
    # import matplotlib.pyplot as plt

    # plt.plot(ll_values)
    # plt.ylabel('LL')
    # plt.show()


if __name__ == "__main__":
    main()