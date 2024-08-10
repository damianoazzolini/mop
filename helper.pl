:- style_check(-discontiguous).
:- set_prolog_flag(stack_limit, 4_294_967_296).

% train([1,2,3,4,5,6,7,8,9]).
% test([10]).

get_modeb(L):-
    findall([N,A], (modeb(_,Atom), functor(Atom, N, A)), LD),
    sort(LD,L). % removes duplicates

get_output_predicates(L):-
  findall([N,A], output(N/A), L).


find_ids_folds(Ids):-
  findall(X, fold(_,X), IdsN),
  flatten(IdsN, Ids).

% % assert this from python so it is possible to change the target predicate
% % which here is active_
% find_ids_examples_true(IdMixture, LID):-
%   findall(Id,active_(IdMixture,Id),LIDD),
%   sort(LIDD, LID).

get_atom(TargetName, TargetArity, Id, Atom):-
  length(A1,TargetArity),
  append([Id],A1,L),
  Atom =.. [TargetName|L].

% pos_neg(Target,Id, [1.0]):-
pos_neg(TargetName, TargetArity, Id, 1.0):-
  get_atom(TargetName, TargetArity, Id, T),
  T, !.
% pos_neg(Target,Id, [0.0]):-
pos_neg(TargetName, TargetArity, Id, 0.0):-
  get_atom(TargetName, TargetArity, Id, T),
  call(neg(T)), !.
% pos_neg(Target,Id, [0.0]):-
pos_neg(TargetName, TargetArity, Id, 0.0):-
  get_atom(TargetName, TargetArity, Id, T),
  writeln("Example not found"),
  writeln(T),
  false.

% :- dynamic not_found/1.

get_fold_01(TargetName, TargetArity,Set,L01):-
  % writeln("set"),
  % writeln(Set),
  % findall(F,fold(F,_),LFFF),
  % writeln(LFFF),
  findall(S,(fold(N,S), member(N,Set)), LF),
  flatten(LF, LL),
  sort(LL,LS),
  maplist(pos_neg(TargetName, TargetArity), LS, L01).

get_01(TargetName, TargetArity, TrainSet, TestSet, L01Train, L01Test):-
    must_be(list, TrainSet),
    must_be(list, TestSet),
    get_fold_01(TargetName, TargetArity, TrainSet, L01Train),
    get_fold_01(TargetName, TargetArity, TestSet, L01Test),
    % findall(S,(fold(N,S), member(N,TrainSet)), LFTrain),
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
    % % findall(L,fold(_,L),LF),
    % % writeln("ok"),
    % flatten(LFTrain, LL),
    % sort(LL,LS),
    % % writeln(Target),
    % % writeln(LS),
    % maplist(pos_neg(Target), LS, L01).

example_modelled(ExampleId,LModelled,R):-
    ( member(ExampleId,LModelled) -> R = [1.0] ; R = [0.0]).

does_it_model(MixtureLists,CurrentId, LO):-
    % writeln(IdsModelled),
    maplist(example_modelled(CurrentId), MixtureLists, LO).

get_examples_matrices(NMixtures, LModels):-
    % writeln("OK0"),
    find_ids_foreach_mixture(NMixtures, L),
    % writeln("OK1"),
    find_ids_folds(Ids),
    % writeln("OK2"),
    maplist(does_it_model(L),Ids,LModels).


:- dynamic find_ids_examples_true/2.

% IdExamples is a list of lists. The list at position i contains the ids for the examples
% true in the mixture i
find_ids_foreach_mixture(NMixtures,IdExamples):-
  N1 is NMixtures - 1,
  findall(X, between(0, N1, X), Idx),
  maplist(find_ids_examples_true, Idx, IdExamples).



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
  % writeln("Computing prob"),
  % writeln(Folds),
  % writeln(V),
  test_prob_lift(P,Folds, _, _, LL, Prob),
  % writeln((P,Prob)),
  % writeln((P,Prob)),
  % writeln(TrainingFolds),
  % maplist(writeln,Prob),
  % writeln((LL,Prob)),
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
    Body =.. [_|AtomsBody],
    % writeln(Body),
    % writeln((AtomsBody,Name,Arities)),
    maplist(functor,AtomsBody,Name,Arities).

extract_terms_list(Body,L):-
    maplist(extract_terms,Body,L).

not_empty(L):- L \= [].

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

% for predsort to remove duplicates: here I consider a duplicate a clause
% that only differs in the probability associated in the head 
% does not work, due to the my_compare/3 with > and <, it still depends on the order
my_compare(=,E0,E1):-
    E0 = [[Head0, _P00, Body0, Arities0], [Head1, _P01, Body1, Arities1], [Head2, _P02, Body2, Arities2]],
    E1 = [[Head0, _P10, Body0, Arities0], [Head1, _P11, Body1, Arities1], [Head2, _P12, Body2, Arities2]], !.
my_compare(=,E0,E1):-
    E0 = [[Head0, _P00, Body0, Arities0], [Head1, _P01, Body1, Arities1]],
    E1 = [[Head0, _P10, Body0, Arities0], [Head1, _P11, Body1, Arities1]], !.
my_compare(=,E0,E1):-
    E0 = [[Head0, _P00, Body0, Arities0]],
    E1 = [[Head0, _P10, Body0, Arities0]], !.
my_compare(>,_,_):- !.
my_compare(<,_,_):- !.


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
  get_prob_loop(LP, Folds, [], _, [], Ps),
  % maplist(writeln,LP),
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
  % maplist(writeln,LPInit),
  % writeln("Removing duplicates and empty lists"),
  maplist(extract_terms_list,LPInit,PO),
  % predsort(my_compare,PO,POS),
  remove_duplicates(PO,POS),
  % maplist(writeln,POS),
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
  compute_probability_examples(LP,TrainingFolds,LProbsFinalTrain),
  % halt,
  % so I don't have to call it another time
  writeln("Computation probability test set"),
  compute_probability_examples(LP,TestFolds,LProbsFinalTest).

  % get_prob_loop(LP,TrainingFolds,[],_,[],Ps),
  % maplist(skim_wrap,Ps,PSkimmed),
  % maplist(keysort,PSkimmed,PSorted), % CRUCIAL: CHECK THIS IS OK AND THE ORDER IS THE SAME AS IN helper.pl
  % maplist(discard_keys_wrapper,PSorted,LProbsFinal).



%%%%%%%%%%%%%%%%% BELOW: only for programs where there is probability associated with rules

% id esempio fissato, indice regola fissato, indice mixture fissato:
% restituisco 1 se la regola di indice fissato nella mixture fissata modella
% l'esempio fissato, 0 altrimenti
% *** asserted from python since the head changes ***
% get_list_example_fixed_index_rule_fixed_index_mixture_fixed(IDExample, IdxRule, IdxMixture, Models):-
%     ( r(IdxMixture, IdxRule, IDExample) -> Models = 1 ; Models = 0).

% id esempio fissato, indice regola fissato: restituisco una lista
% lunga NMixtures con 0 o 1 se la regola di indice fissato modella
% esempio fissato
get_list_example_fixed_index_rule_fixed(IDExample, LMixturesIds, IdxRule, L):-
    % trovo gli esempi modellati da tutte le regole di indice IdxRule in tutte
    % le mixtures -> loop su tutte le mixtures
    maplist(get_list_example_fixed_index_rule_fixed_index_mixture_fixed(IDExample, IdxRule), LMixturesIds, L).

% id esempio fissato: restituisco una lista con NRules liste
get_list_example_fixed(LIdxRules, LMixturesIds, IdExample, L):-
    maplist(get_list_example_fixed_index_rule_fixed(IdExample, LMixturesIds), LIdxRules, L).

% restituisce una lista lunga numero di esempi
% ciascuna lista è lunga numero di regole e contiene una lista con numero mixture elementi
% CALLED from python
get_matrices(NMixtures, NRules, Matrix):-
    find_ids_folds(ExamplesIds),
    NM is NMixtures - 1,
    % NM is NMixtures,
    findall(X, between(0, NM, X), LMixturesIds),
    NR is NRules - 1,
    findall(X, between(0, NR, X), LIdxRules),
    maplist(get_list_example_fixed(LIdxRules, LMixturesIds), ExamplesIds, Matrix).

% devo trovare gli ID degli esempi modellati da ciascuna regola in ciascuna mixture.
% se le regole sono 2 (A e B), devo restituire una lista per ogni esempio.
% in ciascuna lista ho una lista per A ed una per B. Nella lista A ho un 1 se 
% l'esempio corrente è modellato dalla regola A nella mixture alla posizione corrente.

% 3 esempi, 1, 2, 3
% per esempio 1 ottengo
% [1, 0], [0, 0]
% per esempio 2 ottengo
% [0, 1], [1, 1]
% per esempio 3 ottengo
% [0, 1], [1, 1]

  
% % assert this from python
% %%%
% % active is the head predicate
% % the first argument is the index of the mixture
% % the second argument will be unified with the index of the examples
% active_(0,Id) :- ring_size_6(Id,A),benzene(Id,B),phenanthrene(Id,C).
% active_(0,Id) :- benzene(Id,A),ball3(Id,B).

% active_(1,Id) :- bond(Id,A,B,7),ball3(Id,C).
% active_(1,Id) :- bond(Id,A,B,7),phenanthrene(Id,C).
% %%%