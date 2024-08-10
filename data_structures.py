class Clause:
    """
    Wrapper class to keep clauses organized.
    clause: str representation of the clause.
    idx_examples: list containing the indexes of the examples
        where the clause is true. Needed for the computation of the probability.
    """
    def __init__(self,
            clause : str,
            # examples : 'list[Example]',
            target : 'list[str]'
        ) -> None:
        self.clause = clause
        self.target = target
        self.idx_examples : 'list[int]' = [] 
        # self.compute_idx_examples(examples)

    def __str__(self) -> str:
        return self.clause # + f" -> {self.idx_examples}"
    def __repr__(self) -> str:
        return self.__str__()

class Program:
    """
    Class containing a program composed by a set of clauses.
    """
    def __init__(self,
            clauses : 'list[Clause]'
        ) -> None:
        self.clauses = clauses
    
    def __str__(self) -> str:
        return f"{self.clauses}"
    def __repr__(self) -> str:
        return self.__str__()


# class Example:
#     """
#     Class containing the examples.
#     """
#     def __init__(self,
#             atoms : 'list[str]',
#             idx : int,
#             positive : bool
#         ) -> None:
#         self.atoms = atoms
#         self.idx = idx
#         self.positive = positive
    
#     def __str__(self) -> str:
#         return f"int. {self.idx}: " + ','.join(self.atoms)
#     def __repr__(self) -> str:
#         return self.__str__()
