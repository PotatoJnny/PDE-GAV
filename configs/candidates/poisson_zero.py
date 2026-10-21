"""Minimal executable starting point, with no numerical-method assumptions."""
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind
class Prepared:
    def evaluate(self, query):
        return AnswerView(QueryKind.POINTS, {'u': (0.,)*len(query.points)},
                          points=query.points, shape=(len(query.points),))
def prepare_solution(case, solve_budget=None):
    return Prepared()
