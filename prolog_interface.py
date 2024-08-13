import sys

import janus_swi as janus

from data_structures import Program

class PrologInterface:
    def __init__(
            self,
            bg : str,
            train_set : 'list[int]',
            test_set : 'list[int]',
            verbosity : int = 0
        ) -> None:
        self.verbosity = verbosity
        self.train_set = train_set
        self.test_set = test_set

        # read bg knowledge
        f = open(bg, "r")
        lines_bg = f.read()
        f.close()
        self.lines_bg = ":- style_check(-discontiguous).\n:- style_check(-singleton).\n" + lines_bg
        
        # read helper file
        f = open("helper.pl", "r")
        self.lines_helper = f.read()
        f.close()


    def get_modeb_target_and_pos_or_neg_list(self) -> 'tuple[list[str], tuple[str,int], list[float], list[float]]':
        """
        Gets the modeb, target, and a list identifying positive 
        and negative examples, from file.
        """
        
        janus.consult("bg", self.lines_bg + self.lines_helper)

        res = janus.query_once("generate_atoms(body,L)")
        if res["truth"]:
            modeb = res["L"]
        else:
            print("Error in querying background file (generate_body_atoms(body,L))")
            sys.exit()

        res = janus.query_once("generate_atoms(head,L)")
        if res["truth"]:
            target_predicate = res["L"]
        else:
            print("Error in querying background file (generate_body_atoms(head,L))")
            sys.exit()
        
        # print(target_predicate)

        # res = janus.query_once(f"get_01({target_predicate[0][0]}, {target_predicate[0][1]}, {self.train_set}, {self.test_set}, L01Train, L01Test)")
        res = janus.query_once(f"get_01({target_predicate}, {self.train_set}, {self.test_set}, L01Train, L01Test)")
        if res["truth"]:
            l01_train = res["L01Train"]
            l01_test = res["L01Test"]
        else:
            print("Error in querying background file (L01)")
            sys.exit()

        return modeb, target_predicate, l01_train, l01_test

    
    def compute_parameters_mixtures(self,
            programs : 'list[Program]',
            train_set : 'list[int]',
            test_set : 'list[int]'
        ) -> 'tuple[list[str],list[list[float]],list[list[float]]]':
        """
        Calls LIFTCOVER parameter learning on each program
        and computes probabilities.
        Returns the probabilities of the examples in each mixture
        """
        programs_in : str = ""

        for p in programs:
            programs_in += "in(["
            for cl in p.clauses:
                # print(f"adding: {cl.clause}")
                programs_in += f"({cl.clause[:-1]}),"
            
            programs_in = programs_in[:-1] +  "]).\n"
        
        if self.verbosity >= 1:
            print("Programs in")
            print(programs_in)

        # print("FILE lift.pl ** fissato **")
        # # f = open("lift.pl", "r")
        # f = open("lift.pl", "r")
        # lines = f.read()
        # f.close()
        
        janus.consult("bg", self.lines_bg + programs_in + self.lines_helper)

        # print(programs_in)

        # query = "findall(P,induce_par_lift([2],P),LP)."
        # res = janus.query_once(f"findall(P,induce_par_lift([2],P),LP)")
        try:
            training_folds = list(map(int, train_set))
        except:
            training_folds = train_set
        try:
            test_folds = list(map(int, test_set))
        except:
            test_folds = test_set

        res = janus.query_once(f"train({training_folds},{test_folds},LearnedPrograms,ProbTrain,ProbTest)")
        if res["truth"]:
            return res["LearnedPrograms"], res["ProbTrain"], res["ProbTest"]
        else:
            print("Error in learning probabilities")
            sys.exit()
