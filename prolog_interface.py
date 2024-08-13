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
        # f = open("helper.pl", "r")
        self.lines_helper = HELPER_FILE
        # f.close()


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


HELPER_FILE = """
:- style_check(-discontiguous).
:- set_prolog_flag(stack_limit, 4_294_967_296).

%% to generate body atoms from modeb declarations
reconstruct_atom(Name,IdxConst,Arity,Const,Atom):-
  length(LArgs,Arity),
  nth0(IdxConst,LArgs,Const),
  Atom =..[Name | LArgs].

find_groundings(H,GroundedAtoms):-
  H =.. [Name|Args],
  length(Args, NA),
  NA1 is NA + 1,
  length(LA,NA1),
  H1 =.. [Name|LA],
  findall(LA, H1, LAAPos),
  findall(LA, neg(H1), LAANeg),
  append(LAAPos,LAANeg,LAA),
  nth0(I,Args,-#_), !,
  I1 is I + 1,
  maplist(nth0(I1),LAA,GA),
  % writeln(GA),
  sort(GA,GAS),
  maplist(reconstruct_atom(Name,I,NA),GAS,GroundedAtoms).


generate_atom([],L,L).
generate_atom([H|T],LT,LFinal):-
  H =.. [Name|Args],
  % Idx = 0,
  % find all arguments that should be grounded
  % assume that there is only one
  ( member(-#_,Args) ->  
    % find all groundings
    find_groundings(H,Term) ;
    length(Args,N),
      length(LAnon,N),
      Term =.. [Name|LAnon]
  ),
  generate_atom(T,[Term|LT],LFinal).

generate_atoms(K,Strings):-
  ( K = body ->
    findall(A, modeb(_,A), LM) ;
    findall(A, modeh(_,A), LM)
  ),
  generate_atom(LM,[],LA),
  flatten(LA,LAtomsGround),
  sort(LAtomsGround,LAGS),
  % writeln(LAGS),
  maplist(term_string,LAGS,Strings).
  % writeln(LAtomsGround).
%%%% 

% Atom: a/n
% Id: number of the fold
% AtomWithId unifies with an atom of arity n+1 with
% the same head and as first argument the Id 
% Example
% Atom = bupa(_25996,_26002)
% Id = 277 
% AtomWithId = bupa(277,_25996,_26002)
% Ignore constants since here I need only to find whether it is
% a positive or negative example.
get_atom(Atoms, Id, AtomWithId):-
  ( term_string(Atom,Atoms) -> true ; Atom = Atoms),
  Atom =.. [Name|B],
  length(B,N),
  length(NB,N),
  append([Id],NB,BId),
  AtomWithId =.. [Name|BId].

% pos_neg(Target,Id, [1.0]):-
pos_neg(TargetAtom, Id, 1.0):-
  get_atom(TargetAtom, Id, T),
  T, !.
% pos_neg(Target,Id, [0.0]):-
pos_neg(TargetAtom, Id, 0.0):-
  get_atom(TargetAtom, Id, T),
  neg(T), !.
% pos_neg(Target,Id, [0.0]):-
pos_neg(TargetAtom, Id, []):-
  get_atom(TargetAtom, Id, T),
  writeln("Example not found"),
  writeln(T),
  false.
  % false.

% :- dynamic not_found/1.

get_fold_01(TargetAtom, Set,L01):-
  findall(S,(fold(N,S), member(N,Set)), LF),
  flatten(LF, LL),
  sort(LL,LS),
  maplist(pos_neg(TargetAtom), LS, L01).


get_01(TargetAtomsList, TrainSet, TestSet, L01Train, L01Test):-
    must_be(list, TrainSet),
    must_be(list, TestSet),
    must_be(list, TargetAtomsList),
    % here I consider only the first one: this since all the atoms
    % will be partial grounding of the same atom. So I only care about
    % the name and arity, not the arguments
    TargetAtomsList = [TargetAtom | _],
    get_fold_01(TargetAtom, TrainSet, L01Train),
    get_fold_01(TargetAtom, TestSet, L01Test),
    ( L01Train = [] ->
      writeln("Training folds not found"),
      writeln(TrainSet),
      false ;
      true
    ),
    ( L01Test = [] ->
      writeln("No test folds found"),
      true ;
      true
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%% to extract the probability with liftcover
get_probability((_ : P ; _ :- _), P).
get_probability_wrapper(L,PL):-
  maplist(get_probability,L,PL).


skim(P-(\+AID), ID-P):- !,
  AID =.. [_|Args],
  Args = [ID|_].
skim(P-AID, ID-P):-
  AID =.. [_|Args],
  Args = [ID|_].

discard_keys(_-P,P).

skim_wrap(L,S):-
  maplist(skim,L,S).

discard_keys_wrapper(L,S):-
  maplist(discard_keys,L,S).


get_prob_loop([],_,L,L,P,P).
get_prob_loop([P|PT],Folds,LLT,LLS,PT1,Probs):-
  must_be(list, P),
  test_prob_lift(P,Folds, _, _, LL, Prob),
  append(LLT,[LL],LL1),
  append(PT1,[Prob],P1),
  get_prob_loop(PT,Folds,LL1,LLS,P1,Probs).

string_to_clauses(String,Clause):-
  maplist(term_string,Clause,String).
clauses_to_strings(V, S):-
  maplist(term_string,V,S).
  % since terms have no Python representation
  % see
  % https://swi-prolog.discourse.group/t/problem-with-calling-prolog-from-python/7350
  % V = [V1],
  % V1 = (H:P ; _ :- B),
  % T = (H:P :- B),
  % term_string(T,S).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following to remove duplicated programs learnt by LIFTCOVER
% and empty programs
:- op(500,fx,:). % uncomment this for testing
list_to_conj([H],H) :- !.
list_to_conj([H|T], ','(H, Conj)) :-
    list_to_conj(T, Conj).

extract_terms(CL,[H,P,Name,Arities]):-
  CL = (H:P ; _ :- Body),
  Body =.. [N|AtomsBody],
  ( N \= ',' ->
    % only one atom in the body
    maplist(functor,[Body],Name,Arities) ;
    maplist(functor,AtomsBody,Name,Arities)
  ).

extract_terms_list(Body,L):-
    maplist(extract_terms,Body,L).

reconstruct_term(Name,Arity,Term):-
    functor(Term,Name,Arity).

reconstruct_clause([H,P,Names,Arities], CL):-
    % P1 is 1 - P,
    maplist(reconstruct_term,Names,Arities,Terms),
    list_to_conj(Terms,B),
    % CL = (H:P ; :P1 :- B).
    CL = (H:P :- B).

reconstruct_wrapper(CL,R):-
    maplist(reconstruct_clause,CL,R).

is_member(El,List):-
  El = [[Head0, _, Body0, Arities0], [Head1, _P01, Body1, Arities1], [Head2, _, Body2, Arities2]],
  member([[Head0, _, Body0, Arities0], [Head1, _, Body1, Arities1], [Head2, _, Body2, Arities2]],List).
is_member(El,List):-
  El = [[Head0, _, Body0, Arities0], [Head1, _, Body1, Arities1]],
  member([[Head0, _, Body0, Arities0], [Head1, _, Body1, Arities1]],List).
is_member(El,List):-
  El = [[Head0, _, Body0, Arities0]],
  member([[Head0, _, Body0, Arities0]],List).

remove_duplicates([], []).
remove_duplicates([Head | Tail], Result) :-
  is_member(Head, Tail), !,
  remove_duplicates(Tail, Result).

remove_duplicates([Head | Tail], [Head | Result]) :-
  remove_duplicates(Tail, Result).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

compute_probability_examples(LearnedPrograms,Folds,LProbs):-
  ( maplist(string,LearnedPrograms) ->
      maplist(string_to_clauses, LearnedPrograms, LP) ;
      LP = LearnedPrograms
  ),
  % length(LP,NP),
  % writeln(NP),
  get_prob_loop(LP, Folds, [], _, [], PsDup),
  % maplist(writeln,LP),
  Ps = PsDup,
  % maplist(sort,PsDup,Ps), % <--- this is needed if the same pos/k/neg(pos/k) is repeated multiple times
  % writeln(Ps),
  maplist(skim_wrap,Ps,PSkimmed),
  maplist(keysort,PSkimmed,PSorted),
  maplist(discard_keys_wrapper,PSorted,LProbs).

train(TrainingFolds,TestFolds,LearnedPrograms,LProbsFinalTrain,LProbsFinalTest):-
  % writeln(TrainingFolds),
  % findall(P,induce_par_lift(TrainingFolds,P),LP),
  writeln("Learning parameters"),
  findall(P,(induce_par_lift(TrainingFolds,P), P \= []), LPInit),
  writeln("Terminated learning"),
  % writeln("---"),
  % writeln(LPInit),
  % maplist(writeln,LPInit),
  % writeln("****"),
  % writeln("extract_terms_list"),
  maplist(extract_terms_list,LPInit,PO),
  % writeln("remove_duplicates"),
  remove_duplicates(PO,POS),
  % maplist(writeln,POS),
  % writeln("reconstruct_wrapper"),
  maplist(reconstruct_wrapper,POS,LP),
  % maplist(writeln,LP),
  % fail,
  % sort(LP,LSorted),
  % writeln(LSorted),
  % maplist(get_probability_wrapper,LP,LPP), % non mi serve la prob
  % writeln(LPP),
  % writeln("Induce par_lift"),
  % writeln(LP),
  maplist(clauses_to_strings,LP,LearnedPrograms),
  % writeln("HERE"),
  % writeln(LearnedPrograms),
  % writeln([H,P,B]),
  % findall(I,fold(I,_),Folds),
  % writeln(Folds),
  %%%%%%
  writeln("Computation probability training set"),
  % maplist(writeln,LP),
  % writeln(TrainingFolds),
  compute_probability_examples(LP,TrainingFolds,LProbsFinalTrain),
  % writeln(LProbsFinalTrain),
  % halt,
  % so I don't have to call it another time
  writeln("Computation probability test set"),
  compute_probability_examples(LP,TestFolds,LProbsFinalTest).
"""