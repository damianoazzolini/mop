class Clause:
    """
    Wrapper class to keep clauses organized.
    """
    def __init__(self,
            clause : str,
        ) -> None:
        self.clause = clause

    def __str__(self) -> str:
        return self.clause
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
