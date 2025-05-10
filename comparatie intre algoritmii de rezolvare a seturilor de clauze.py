import tracemalloc
import random
import time
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from pysat.solvers import Solver
from pysat.formula import CNF
import multiprocessing
from multiprocessing import Process, Queue

Literal = int
Clause = List[Literal]
Formula = List[Clause]
Assignment = Dict[int, bool]

# Timeout settings
TIMEOUT_SECONDS = 60


# ================== Formula Generation ==================
def generate_large_formula(num_clauses: int, max_literals_per_clause: int, num_variables: int) -> Formula:
    """Generate random CNF formula"""
    formula = []
    for _ in range(num_clauses):
        clause_size = random.randint(1, max_literals_per_clause)
        clause_vars = random.sample(range(1, num_variables + 1), clause_size)
        clause = [var if random.random() < 0.5 else -var for var in clause_vars]
        formula.append(clause)
    return formula


# ================== Input Handling ==================
def read_formula_from_input() -> Formula:
    """Read CNF formula from user input"""
    print("\nEnter clauses one by line. End each clause with 0.")
    print("Example: 1 -2 3 0 (this means x1 ∨ ¬x2 ∨ x3)")

    formula = []
    while True:
        while True:
            clause_str = input("Enter clause (or 'done' to finish): ").strip()
            if clause_str.lower() == 'done':
                return formula

            try:
                literals = list(map(int, clause_str.split()))
                if literals[-1] != 0:
                    print("Error: Clause must end with 0")
                    continue
                clause = literals[:-1]
                if not all(lit != 0 for lit in clause):
                    print("Error: 0 can only appear at end of clause")
                    continue
                formula.append(clause)
                break
            except ValueError:
                print("Error: Please enter integers only")

        print(f"Current formula: {len(formula)} clauses")


# ================== Solver Algorithms ==================
def resolution_solver(formula: Formula) -> bool:
    clauses = [set(clause) for clause in formula]
    seen = {frozenset(c) for c in clauses}

    while True:
        new_clauses = set()
        n = len(clauses)

        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = clauses[i], clauses[j]

                # Find all possible resolvents
                for lit in ci:
                    if -lit in cj:
                        resolvent = (ci | cj) - {lit, -lit}

                        # Skip tautologies
                        if any(abs(l) == abs(m) for l in resolvent for m in resolvent if l != m):
                            continue

                        if not resolvent:
                            return False  # Empty clause

                        f_resolvent = frozenset(resolvent)
                        if f_resolvent not in seen:
                            new_clauses.add(f_resolvent)

        if not new_clauses:
            return True  # No new clauses can be derived

        seen.update(new_clauses)
        clauses.extend([set(c) for c in new_clauses])


def davis_putnam_solver(formula: Formula) -> bool:
    clauses = [set(clause) for clause in formula]
    variables = {abs(lit) for clause in clauses for lit in clause}

    while variables:
        var = variables.pop()

        # Split clauses
        pos_clauses = [c for c in clauses if var in c]
        neg_clauses = [c for c in clauses if -var in c]
        remaining_clauses = [c for c in clauses if var not in c and -var not in c]

        # Perform resolution
        new_clauses = []
        for pc in pos_clauses:
            for nc in neg_clauses:
                resolvent = (pc - {var}) | (nc - {-var})
                if any(lit in resolvent and -lit in resolvent for lit in resolvent):
                    continue  # Skip tautologies
                if not resolvent:
                    return False  # Empty clause means unsatisfiable
                new_clauses.append(resolvent)

        # Add only non-redundant new clauses
        unique_new = []
        for nc in new_clauses:
            if not any(nc.issuperset(existing) for existing in remaining_clauses + unique_new):
                unique_new.append(nc)

        clauses = remaining_clauses + unique_new
        variables = {abs(lit) for clause in clauses for lit in clause}

    return True  # No empty clause found


def dpll_optimized(formula: Formula, assignment: Optional[Assignment] = None) -> List[Assignment]:
    """Optimized DPLL implementation"""
    if assignment is None:
        assignment = {}

    # Unit propagation
    def unit_propagate(f: Formula, a: Assignment):
        changed = True
        while changed:
            changed = False
            unit_clauses = [c for c in f if len(c) == 1]
            for clause in unit_clauses:
                lit = clause[0]
                var = abs(lit)
                val = lit > 0

                if var in a:
                    if a[var] != val:
                        return None, a
                    continue

                a[var] = val
                changed = True
                new_f = []
                for c in f:
                    if lit in c:
                        continue
                    new_c = [l for l in c if l != -lit]
                    if not new_c:
                        return None, a
                    new_f.append(new_c)
                f = new_f
        return f, a

    formula, assignment = unit_propagate(formula, assignment)
    if formula is None:
        return []
    if not formula:
        return [assignment]

    # Pure literal elimination
    literal_sign = defaultdict(set)
    for clause in formula:
        for lit in clause:
            var = abs(lit)
            if var not in assignment:
                literal_sign[var].add(lit > 0)

    pure_literals = []
    for var, signs in literal_sign.items():
        if len(signs) == 1:
            pure_literals.append(var if True in signs else -var)

    if pure_literals:
        new_assignment = assignment.copy()
        for lit in pure_literals:
            new_assignment[abs(lit)] = lit > 0
        new_formula = []
        for clause in formula:
            if not any(lit in clause for lit in pure_literals):
                new_clause = [lit for lit in clause if -lit not in pure_literals]
                new_formula.append(new_clause)
        return dpll_optimized(new_formula, new_assignment)

    # Variable selection
    var_counts = defaultdict(int)
    for clause in formula:
        for lit in clause:
            var = abs(lit)
            if var not in assignment:
                var_counts[var] += 1

    if not var_counts:
        return [assignment]

    var = max(var_counts.items(), key=lambda x: x[1])[0]
    solutions = []
    for val in [True, False]:
        new_assignment = assignment.copy()
        new_assignment[var] = val
        solutions.extend(dpll_optimized(formula, new_assignment))
    return solutions


class CDCLSolver:
    """Complete CDCL implementation"""

    def __init__(self, formula: Formula):
        self.formula = formula
        self.assignment = {}
        self.level = 0
        self.decisions = []
        self.antecedents = {}
        self.levels = {}
        self.watch_list = defaultdict(set)
        self.activity = defaultdict(float)
        self.var_inc = 1.0
        self.var_decay = 0.95
        self.setup_watch_list()

    def setup_watch_list(self):
        for i, clause in enumerate(self.formula):
            if len(clause) > 1:
                self.watch_list[clause[0]].add(i)
                self.watch_list[clause[1]].add(i)
            elif clause:
                lit = clause[0]
                var = abs(lit)
                if var not in self.assignment:
                    self.assignment[var] = lit > 0
                    self.levels[var] = 0
                    self.antecedents[var] = i

    def solve(self) -> Tuple[bool, Optional[Assignment]]:
        while True:
            conflict = self.propagate()
            if conflict is not None:
                if self.level == 0:
                    return False, None

                learned_clause, bt_level = self.analyze_conflict(conflict)
                self.learn_clause(learned_clause)
                self.backtrack(bt_level)
            else:
                if all(abs(l) in self.assignment for clause in self.formula for l in clause):
                    return True, self.assignment.copy()

                var = self.select_variable()
                if var is None:
                    return True, self.assignment.copy()

                self.level += 1
                self.decisions.append(var)
                self.assignment[var] = True
                self.levels[var] = self.level

    def propagate(self) -> Optional[Clause]:
        while True:
            unit = None
            for lit in list(self.watch_list):
                var = abs(lit)
                if var in self.assignment:
                    if (lit > 0) != self.assignment[var]:
                        for clause_idx in list(self.watch_list[lit]):
                            clause = self.formula[clause_idx]
                            found = False
                            for other_lit in clause:
                                if other_lit != lit:
                                    other_var = abs(other_lit)
                                    if other_var not in self.assignment or (other_lit > 0) == self.assignment[
                                        other_var]:
                                        self.watch_list[lit].remove(clause_idx)
                                        self.watch_list[other_lit].add(clause_idx)
                                        found = True
                                        break

                            if not found:
                                if len(clause) == 1:
                                    unit = clause[0]
                                else:
                                    return clause
            if unit is None:
                return None

            var = abs(unit)
            if var in self.assignment:
                if (unit > 0) != self.assignment[var]:
                    return [unit]
            else:
                self.assignment[var] = unit > 0
                self.levels[var] = self.level
                self.antecedents[var] = -1

    def analyze_conflict(self, conflict: Clause) -> Tuple[Clause, int]:
        literals = []
        levels = set()

        for lit in conflict:
            var = abs(lit)
            levels.add(self.levels[var])
            literals.append(lit)

        while len(levels) > 1:
            last_lit = None
            max_level = max(levels)
            for lit in literals:
                if self.levels[abs(lit)] == max_level:
                    last_lit = lit
                    break

            if last_lit is None:
                break

            var = abs(last_lit)
            antecedent_idx = self.antecedents.get(var)
            if antecedent_idx is None or antecedent_idx == -1:
                break

            antecedent = self.formula[antecedent_idx]
            new_literals = []
            for lit in literals:
                if lit != last_lit and -lit not in antecedent:
                    new_literals.append(lit)

            for lit in antecedent:
                if lit != -last_lit and lit not in new_literals:
                    new_literals.append(lit)

            literals = new_literals
            levels = {self.levels[abs(l)] for l in literals}

        bt_level = max(levels - {max(levels)}) if len(levels) > 1 else 0
        return literals, bt_level

    def learn_clause(self, clause: Clause):
        if clause:
            clause_idx = len(self.formula)
            self.formula.append(clause)
            if len(clause) > 1:
                self.watch_list[clause[0]].add(clause_idx)
                self.watch_list[clause[1]].add(clause_idx)

            self.var_inc *= 1 / self.var_decay
            for lit in clause:
                self.activity[abs(lit)] += self.var_inc

    def backtrack(self, level: int):
        to_remove = [var for var in self.assignment if self.levels[var] > level]
        for var in to_remove:
            del self.assignment[var]
            del self.levels[var]
            if var in self.antecedents:
                del self.antecedents[var]

        self.decisions = [d for d in self.decisions if self.levels.get(abs(d), 0) <= level]
        self.level = level

    def select_variable(self) -> Optional[int]:
        unassigned = [var for var in
                      range(1, max(abs(l) for clause in self.formula for l in clause) + 1)
                      if var not in self.assignment]

        if not unassigned:
            return None

        self.var_inc *= self.var_decay
        return max(unassigned, key=lambda v: self.activity.get(v, 0))


def cdcl_solve(formula: Formula) -> Tuple[bool, Optional[Assignment]]:
    solver = CDCLSolver(formula)
    return solver.solve()


def pysat_solver(formula: Formula) -> List[Assignment]:
    """Solve using PySAT's Glucose3 solver"""
    cnf = CNF()
    for clause in formula:
        cnf.append(clause)

    with Solver(name='glucose3', bootstrap_with=cnf) as solver:
        if solver.solve():
            model = solver.get_model()
            return [{abs(lit): lit > 0 for lit in model}]
        return []


def hybrid_solver(formula: Formula, threshold=1000) -> List[Assignment]:
    """Choose solver based on problem size"""
    if len(formula) > threshold:
        return pysat_solver(formula)
    return dpll_optimized(formula)


# ================== Timeout Wrappers ==================
def run_solver_with_timeout(solver_func, formula: Formula, queue: Queue):
    """Run a solver function with timeout using multiprocessing"""
    try:
        result = solver_func(formula)
        queue.put(('result', result))
    except Exception as e:
        queue.put(('error', str(e)))


def execute_with_timeout(solver_func, formula: Formula, timeout: int) -> Tuple[Optional[any], Optional[str]]:
    """Execute a solver function with a timeout"""
    queue = Queue()
    process = Process(target=run_solver_with_timeout, args=(solver_func, formula, queue))

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return None, f"Timeout after {timeout} seconds"

    if not queue.empty():
        msg_type, content = queue.get()
        if msg_type == 'result':
            return content, None
        elif msg_type == 'error':
            return None, content

    return None, "Unknown error occurred"


# ================== Results Handling ==================
def save_results_to_file(formula: Formula, results: dict, filename: str = "results.txt"):
    """Save comparison results with detailed timing information"""
    try:
        with open(filename, 'a') as f:
            # Header with formula info
            variables = {abs(l) for clause in formula for l in clause}
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"SAT SOLVER COMPARISON RESULTS\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nFORMULA STATISTICS:\n")
            f.write(f"- Clauses: {len(formula)}\n")
            f.write(f"- Variables: {len(variables)}\n")
            f.write(f"- Avg clause length: {sum(len(c) for c in formula) / len(formula):.2f}\n")

            # Timing results section (REMOVED THE CLAUSE PRINTING SECTION)
            f.write("\nSOLVER PERFORMANCE:\n")
            f.write("{:<15} {:<12} {:<15} {:<30}\n".format(
                "Solver", "Time (s)", "Memory (KB)", "Result"))
            f.write("-" * 80 + "\n")

            for name, res in results.items():
                time_str = f"{res['time']:.6f}" if res['time'] >= 0 else "Error"
                mem_str = f"{res['memory']:.2f}" if res['memory'] >= 0 else "Error"
                result = res['output']
                if len(result) > 50:
                    result = result[:50] + "..."
                f.write("{:<15} {:<12} {:<15} {:<30}\n".format(
                    name, time_str, mem_str, result))

            # Detailed timing comparison
            f.write("\nTIMING COMPARISON:\n")
            valid_times = [(name, res['time']) for name, res in results.items() if res['time'] >= 0]
            if valid_times:
                fastest = min(valid_times, key=lambda x: x[1])
                slowest = max(valid_times, key=lambda x: x[1])
                f.write(f"- Fastest solver: {fastest[0]} ({fastest[1]:.6f}s)\n")
                f.write(f"- Slowest solver: {slowest[0]} ({slowest[1]:.6f}s)\n")

                if len(valid_times) > 1:
                    ratio = slowest[1] / fastest[1]
                    f.write(f"- Speed difference: {ratio:.2f}x\n")

            f.write("=" * 80 + "\n")

        print(f"\nResults saved to {filename} (without clause details)")
    except IOError as e:
        print(f"Error saving file: {e}")


# ================== Main Menu ==================
def main_menu():
    print("SAT Solver Comparison Tool")
    print("=" * 40)
    print(f"Note: All solvers will timeout after {TIMEOUT_SECONDS} seconds")

    while True:
        print("\nMain Menu:")
        print("1. Enter formula manually")
        print("2. Generate random formula")
        print("3. Exit")
        choice = input("Choose option: ").strip()

        if choice == '3':
            print("Exiting program.")
            return

        if choice == '1':
            formula = read_formula_from_input()
        elif choice == '2':
            try:
                num_clauses = int(input("Number of clauses: "))
                max_literals = int(input("Max literals per clause: "))
                num_vars = int(input("Number of variables: "))
                formula = generate_large_formula(num_clauses, max_literals, num_vars)
                print(f"Generated formula with {len(formula)} clauses")
                print("\nSample of generated clauses:")
                for clause in formula[:5]:
                    print(" ".join(map(str, clause)) + " 0")
                if len(formula) > 5:
                    print(f"... and {len(formula) - 5} more clauses")
            except ValueError:
                print("Invalid input! Please enter integers.")
                continue
        else:
            print("Invalid choice")
            continue

        # Display formula statistics
        variables = {abs(l) for clause in formula for l in clause}
        print("\nFormula statistics:")
        print(f"- Clauses: {len(formula)}")
        print(f"- Variables: {len(variables)}")
        print(f"- Avg clause length: {sum(len(c) for c in formula) / len(formula):.2f}")

        # Solver selection
        solvers = {
            '1': ('Resolution', resolution_solver),
            '2': ('Davis-Putnam', davis_putnam_solver),
            '3': ('DPLL', dpll_optimized),
            '4': ('CDCL', cdcl_solve),
            '5': ('PySAT', pysat_solver),
            '6': ('Hybrid', hybrid_solver)
        }

        print("\nSelect solvers to compare:")
        for num, (name, _) in solvers.items():
            print(f"{num}. {name}")
        print("7. All solvers")
        print("8. Back to main menu")

        while True:
            solver_choice = input("\nEnter choices (comma separated, or 7 for all): ").strip()

            if solver_choice == '8':
                break

            selected = solver_choice.split(',')
            if '7' in selected:
                to_run = list(solvers.values())
            else:
                to_run = [solvers[num] for num in selected if num in solvers]

            if not to_run:
                print("No valid solvers selected")
                continue

            # Run selected solvers
            results = {}
            for name, solver in to_run:
                print(f"\nRunning {name}...")
                tracemalloc.start()
                start = time.time()

                try:
                    # Run with timeout
                    result, error = execute_with_timeout(solver, formula.copy(), TIMEOUT_SECONDS)
                    end = time.time()
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    if error:
                        output = error
                        results[name] = {
                            'time': TIMEOUT_SECONDS if "Timeout" in error else -1,
                            'memory': peak / 1024 if peak >= 0 else -1,
                            'output': output
                        }
                        print(output)
                        continue

                    if name in ['Resolution', 'Davis-Putnam']:
                        output = f"Formula is {'satisfiable' if result else 'unsatisfiable'}"
                    elif name == 'CDCL':
                        sat, assignment = result
                        output = f"Formula is {'satisfiable' if sat else 'unsatisfiable'}"
                        if sat:
                            output += f"\nAssignment sample: {dict(list(assignment.items())[:5])}..."
                    else:
                        output = f"Found {len(result)} solution(s)" if result else "No solutions found"
                        if result:
                            output += f"\nFirst assignment sample: {dict(list(result[0].items())[:5])}..."

                    results[name] = {
                        'time': end - start,
                        'memory': peak / 1024,
                        'output': output
                    }

                    print(output)
                    print(f"Time: {end - start:.6f} seconds")
                    print(f"Peak memory: {peak / 1024:.2f} KB")

                except Exception as e:
                    tracemalloc.stop()
                    print(f"Error in {name}: {str(e)}")
                    results[name] = {
                        'time': -1,
                        'memory': -1,
                        'output': f"Error: {str(e)}"
                    }

            # Save results option
            save = input("\nSave these results to file? (y/n): ").lower()
            if save == 'y':
                filename = "rezultat.txt"
                save_results_to_file(formula, results, filename)

            print("\n1. Run more solvers on same formula")
            print("2. Back to main menu")
            next_choice = input("Choose option: ").strip()
            if next_choice == '2':
                break


if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows compatibility
    main_menu()