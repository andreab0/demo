import cvxpy as cvx
import numpy as np
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

OPERATOR_MAP = {
    "==": lambda x, y: x == y,
    "<=": lambda x, y: x <= y,
    ">=": lambda x, y: x >= y,
}


class CvxpyWrapper:
    """
    A higher-level wrapper around CVXPY to store variables, objectives,
    and constraints in a structured manner.
    """

    def __init__(self):
        self.objectives = []
        self.constraints = []
        self.var_map = {}
        self.var_scope = "global"

    @contextmanager
    def variable_scope(self, scope: str):
        """
        Context manager to temporarily switch the "scope" for variable naming.
        Example usage:
            with wrapper.variable_scope("my_scope"):
                x = wrapper.get_variable("weights", shape=10)
        """
        pre_scope = self.var_scope
        self.var_scope = scope
        try:
            yield
        finally:
            self.var_scope = pre_scope

    def get_variable(self, var_name, shape, **kwargs):
        """
        Returns (and caches) a cvx.Variable. If it doesn't exist, creates it.
        """
        scoped_name = f"{self.var_scope}_{var_name}"
        if scoped_name in self.var_map:
            return self.var_map[scoped_name]
        else:
            var = cvx.Variable(shape, name=scoped_name, **kwargs)
            self.var_map[scoped_name] = var
            return var

    ########################################
    # Objective Adders
    ########################################

    def add_quad_objective(self, x, H, weight=1.0):
        """
        Add a quadratic form x^T H x (multiplied by 'weight').
        """
        H = cvx.psd_wrap(H)
        self.objectives.append(weight * cvx.quad_form(x, H))

    def add_sum_of_squares_objective(self, x, h=None, weight=1.0):
        """
        Adds sum of squares objective: sum((x_i*h_i)^2).
        If h is None, it's sum of squares of x.
        """
        if h is not None:
            self.objectives.append(weight * cvx.sum_squares(cvx.multiply(x, h)))
        else:
            self.objectives.append(weight * cvx.sum_squares(x))

    def add_linear_objective(self, x, h, weight=1.0):
        """
        Adds a linear objective: weight * (x dot h).
        """
        self.objectives.append(weight * (x @ h))

    def add_norm_objective(self, xs, weight=1.0):
        """
        Adds a single L2 norm objective over the horizontal stack of 'xs'.
        """
        if not isinstance(xs, list):
            xs = [xs]
        vect = cvx.hstack(xs)
        self.objectives.append(weight * cvx.norm(vect, 2))

    ########################################
    # Constraint Adders
    ########################################

    def add_linear_constraints(self, x, A, operator, rhs, vectorize=True):
        """
        Adds a linear constraint of the form: (x * A) operator rhs
        If 'vectorize=True', uses elementwise multiply,
        else uses matrix multiply (x dot A).

        :param x: CVXPY variable
        :param A: numpy array or vector
        :param operator: one of '==', '<=', '>='
        :param rhs: numeric scalar, array, or CVXPY expression
        :param vectorize: bool
        """
        if operator not in OPERATOR_MAP:
            raise ValueError(f"Unknown operator {operator}")
        op_func = OPERATOR_MAP[operator]

        if vectorize:
            # elementwise multiply
            lhs = cvx.multiply(x, A)
        else:
            # matrix multiply
            lhs = x @ A

        new_constraint = op_func(lhs, rhs)
        self.constraints.append(new_constraint)

    def add_sum_of_squares_constraint(self, x, h, rhs):
        """
        Adds sum of squares constraint: sum((x_i*h_i)^2) <= rhs
        """
        self.constraints.append(cvx.sum_squares(cvx.multiply(x, h)) <= rhs)

    ########################################
    # Solve
    ########################################

    def solve(self, solver="ECOS", verbose=False, check_feasibility=False, **kwargs):
        """
        Solve the problem using the specified solver.
        :param check_feasibility: if True, will log constraints post-solve.
        """
        total_obj = sum(self.objectives) if len(self.objectives) > 0 else 0.0
        problem = cvx.Problem(cvx.Minimize(total_obj), self.constraints)

        logger.debug(
            "CVXPY solve => solver=%s, verbose=%s, #vars=%d, #constraints=%d",
            solver,
            verbose,
            len(problem.variables()),
            len(self.constraints),
        )

        problem.solve(solver=solver, verbose=verbose, **kwargs)

        # Optionally, we can do a quick feasibility check
        if check_feasibility and problem.status in ["optimal", "optimal_inaccurate"]:
            # Evaluate constraints
            for i, cst in enumerate(self.constraints):
                val = cst.violation()
                if val > 1e-7:
                    logger.warning(
                        "Constraint %d has violation=%.6g => %s", i, val, str(cst)
                    )
        else:
            if problem.status not in ["optimal"]:
                logger.warning("Problem ended with status=%s", problem.status)

        # Collect variable solutions
        var_values = {}
        for var in problem.variables():
            var_values[var.name()] = var.value

        return problem.value, var_values
