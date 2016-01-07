from enum import Enum
from copy import deepcopy
import random

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

CTR = 0
IT = 0

class Type(Enum):
    linear = 1
    encime = 2
    active = 3

D_TYPES = (Type.linear, Type.linear, Type.linear, Type.linear, Type.encime, Type.encime, Type.active)
ARP_TYPES = (Type.linear, Type.linear, Type.encime)

# Change factor
NO_CHG_TYPE = 0.9
CHG_FACTOR = 0.5
N_PROTEINS = 3
ADD_RM_PROTEIN_EXPRESSION = 0.05
ADD_RM_PROTEIN = 0.02
PROTEIN_MUTATION_DEFAULT = 2
GENERATION_MUTATION_RATE = 2

N_NETS_BEFORE = 20
N_NETS_AFTER = 4

FREQUENCY = 50
OUTER_DIFF = 3
DT = 5

ON_VAL = 10
OFF_VAL = 0
WORST_EVAL = 1e10


class Expression:
    def __init__(self):
        self._typ = Type.linear
        self._factor = 1
        self._limit = 0
        self._other = None

    def set_other(self, other):
        self._other = other

    def get_other(self):
        return self._other

    def mutate(self, other, new_type):
        if random.random() < NO_CHG_TYPE:
            if CHG_FACTOR or self._typ != Type.encime:
                self._factor *= random.random() * 1.5
            else:
                self._limit += (random.random() * 2.5 - 1)
                if self._limit < 0:
                    self._limit = 0
        else:
            self._typ = new_type
            self._other = other
            self._factor = 1
            self._limit = 0

    def get_random_typ(self, types):
        new_type = random.choice(types)
        while new_type == self._typ:
            new_type = random.choice(types)

        return new_type


class ActRep(Expression):
    def calc_diff_out(self, prot):
        diff = 0
        if self._typ == Type.linear:
            diff = self._factor
        elif self._typ == Type.encime:
            ra = 1 if self._other.val - prot.val < 0 else 0
            ra = ra if self._factor > 0 else 1 - ra
            diff = self._factor * ra

        return diff

    def mutate(self, other, new_type=None):
        new_type = super(ActRep, self).get_random_typ(ARP_TYPES)
        super(ActRep, self).mutate(other, new_type)


class Activator(ActRep):
    def calc_diff(self, protein, _):
        protein.diff += self.calc_diff_out(protein)


class Repressor(ActRep):
    def calc_diff(self, protein, _):
        protein.diff -= self.calc_diff_out(protein)


class Degradation(Expression):
    def calc_diff(self, prot, _):
        if self._typ == Type.linear:
            diff = prot.val * self._factor
        elif self._typ == Type.encime:
            diff = self._factor * prot.val / (self._limit + prot.val)
        else:
            diff = self._factor * prot.val * self._other.val
        prot.diff -= diff

    def mutate(self, other, new_type=None):
        new_type = super(Degradation, self).get_random_typ(D_TYPES)
        super(Degradation, self).mutate(other, new_type)


class ProteinModification(Expression):
    def calc_diff(self, protein, _):
        diff = 0
        if self._typ == Type.linear:
            diff = self._factor * protein.val
        elif self._typ == Type.encime:
            diff = self._factor * protein.val / (self._limit + protein.val)

        protein.diff -= diff
        self._other.diff += diff

    def mutate(self, other, new_type=None):
        new_type = super(ProteinModification, self).get_random_typ(ARP_TYPES)
        super(ProteinModification, self).mutate(other, new_type)

###############################################
# End of expressions, now Protein and Network #
###############################################


class Protein:
    def __init__(self):
        self.val = 1
        self.diff = 0
        self.expressions = {
            "ACT": Activator(),
            "REP": Repressor(),
            "DEG": Degradation(),
        }

    # A mutation that adds ProteinMutation expression to protein
    def add_expression_pm(self, proteins):
        for prot in random.sample(proteins, len(proteins)):
            if prot not in self.expressions and prot != self:
                new_expr = ProteinModification()
                new_expr.set_other(prot)
                self.expressions[prot] = new_expr
                break

    # A mutation that removes ProteinMutation expression to protein
    def rm_expression_pm(self):
        for k in random.sample(list(self.expressions.keys()), len(self.expressions)):
            if k not in ("ACT", "REP", "DEG") and self.expressions[k] is not None:
                self.expressions.pop(k)
                break

    # Extually mutating this protein
    def mutate(self, proteins, k=None):
        prot = random.choice(proteins)
        while prot == self:
            prot = random.choice(proteins)

        # Do add/remove protein/protein expression
        if random.random() < ADD_RM_PROTEIN_EXPRESSION:
            if random.random() > 0.4:
                self.add_expression_pm(proteins)
            else:
                self.rm_expression_pm()
        # modify ACT/RET/DEG expression
        else:
            if k is None:
                k = random.choice(list(self.expressions.keys()))
            self.expressions[k].mutate(prot)

    def remove_protein(self, protein, proteins):
        chk = {"ACT": Activator(), "DEG": Degradation(), "REP": Repressor}
        for k, e in chk.items():
            if self.expressions[k].get_other == protein:
                self.expressions[k] = e
                self.expressions[k].mutate(proteins)

        if protein in self.expressions:
            self.expressions.pop(protein)


class Network:
    """
    Contains a couple of proteins that are connected with
    some expressions (degradation, activation,...).
    Also includes utilities to simulate the network and analysis.
    TODO: add analysis and mutation
    """
    def __init__(self):
        self.proteins = [Protein() for _ in range(N_PROTEINS)]
        self.mutation_rate = GENERATION_MUTATION_RATE
        self.graph = None

        self.time = np.linspace(0, 200, num=16384)
        self.ideal = [OFF_VAL if (t % FREQUENCY) < FREQUENCY / 2 else ON_VAL for t in self.time]

    def mutate(self):
        if random.random() < ADD_RM_PROTEIN:
            self.add_rm_protein()

        # Select random protein and mutate "mutation_rate" number of times
        for i in range(self.mutation_rate):
            p = random.choice(self.proteins)
            p.mutate(self.proteins)

    def add_rm_protein(self):
        if random.random() > 0.5 or len(self.proteins) <= 3:
            self.proteins.append(Protein())
            for i in range(PROTEIN_MUTATION_DEFAULT):
                self.proteins[-1].mutate(self.proteins)
        else:
            rmp = random.choice(self.proteins)
            for p in self.proteins:
                if p != rmp:
                    p.remove_protein(rmp, self.proteins)

    def diff_closure(self):
        first_in = FREQUENCY / 2
        second_in = FREQUENCY
        k = 4 * OUTER_DIFF / (DT * DT)

        # Defining calc diff closure function
        # for use in odeint. For every protein
        def calc_diff(x, t):
            # Copy values into protein
            for i, p in enumerate(self.proteins):
                p.val = x[i]
                p.diff = 0

            # Calculate diffs from all expressions in all proteins
            for p in self.proteins:
                for e in p.expressions.values():
                    e.calc_diff(p, self.proteins)
            df = [p.diff for p in self.proteins]

            # Handle output signal with triangle DIFF
            # This has to be done with "triangle" in order for ODE
            # to converge. DIFF function looks like this:
            """
            |       Integral is OUTER_DIFF = total change in protein in this time
            |       k is steepness of this function
            |   /\  DT is how much time the protein will be changing in value.
            |  /  \
            | /    \
            |/      \
            +--------DT---->
            """
            nonlocal first_in, second_in, k
            if first_in < t < first_in + DT:
                lt = (t - first_in)
                diff = lt * k if lt < DT / 2 else -lt * k + k * DT
                df[0] += diff
            elif t > first_in + DT:
                first_in += FREQUENCY

            if second_in < t < second_in + DT:
                lt = (t - second_in)
                diff = lt * k if lt < DT / 2 else -lt * k + k * DT
                df[1] += diff
            elif t > second_in + DT:
                second_in += FREQUENCY

            return df

        # return the closure
        return calc_diff

    def run_simulation(self):
        Network.current = self
        init = [p.val for p in self.proteins]
        global IT
        IT += 1

        try:
            self.graph, info = odeint(self.diff_closure(), init, self.time, hmin=0.0001, hmax=1, printmessg=False, full_output=True)
            if info['message'] != 'Integration successful.':
                return WORST_EVAL + 1

            # HERE should go the evaluation of the result
            return self.evaluate_me()

        except Exception:
            return WORST_EVAL + 2

    def evaluate_me(self):
        eval = WORST_EVAL
        for p in range(2, len(self.proteins)):
            eval = min(eval, sum((self.graph[i, p] - self.ideal[i]) ** 2 for i in range(len(self.time))))

        # TODO!
        return eval

    def plot_me(self):
        plt.close()

        for i, _ in enumerate(self.proteins):
            g = self.graph[:, i]
            plt.plot(self.time, g, label='Protein ' + str(i + 1))

        plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3, ncol=2, mode="expand")
        global CTR
        plt.title('Plot Nr: ' + str(CTR))
        plt.savefig('npmp_' + str(CTR // 100)  + str((CTR // 10) % 10) + str(CTR % 10) + '.png', bbox_inches='tight')
        CTR += 1

def mutation_stuff():
    networks = []
    for _ in range(N_NETS_BEFORE):
        n = Network()
        n.mutate()
        networks.append(n)

    good_old_analysis = []

    while True:
        # Preserve previous best
        analysis = good_old_analysis

        # Do analysis
        for n in networks:
            analysis.append((n, n.run_simulation()))

        # Sort analysis
        analysis = sorted(analysis, key=lambda x: x[1])

        # Print current state
        best_analysis = analysis[0]
        avg_analysis = sum(a[1] for a in analysis[:N_NETS_AFTER]) / N_NETS_AFTER
        print("Generation %d: BEST: %d, AVERAGE: %d" % (CTR, best_analysis[1], avg_analysis))
        best_analysis[0].plot_me()

        # Generate new mutations
        networks = []
        good_old_analysis = analysis[:N_NETS_AFTER]
        for (ntw, _) in analysis[:N_NETS_AFTER]:
            multiply = N_NETS_BEFORE / N_NETS_AFTER
            assert(multiply == round(multiply))

            for i in range(int(multiply)):
                n = deepcopy(ntw)
                n.mutate()
                networks.append(n)

if __name__ == "__main__":
    filelist = [f for f in os.listdir(".") if f.endswith(".png")]
    for f in filelist:
        os.remove(f)
    mutation_stuff()
