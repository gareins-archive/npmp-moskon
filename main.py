from enum import Enum
from copy import deepcopy
import random

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


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
ADD_RM_PROTEIN_EXPRESSION = 0.1
ADD_RM_PROTEIN = 0.01
PROTEIN_MUTATION_DEFAULT = 4

N_NETS_BEFORE = 20
N_NETS_AFTER = 4

FREQUENCY = 50
OUTER_DIFF = 3
DT = 5


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
                self._factor *= random.random() * 2
            else:
                self._limit += (random.random() * 3 - 1)
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
                del self.expressions[k]
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

        del self.expressions[protein]


class Network:
    """
    Contains a couple of proteins that are connected with
    some expressions (degradation, activation,...).
    Also includes utilities to simulate the network and analysis.
    TODO: add analysis and mutation
    """
    def __init__(self):
        self.proteins = [Protein() for _ in range(N_PROTEINS)]
        self.mutation_rate = 5
        self.graph = None
        self.time = None

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
            df = []
            for p in self.proteins:
                for e in p.expressions.values():
                    e.calc_diff(p, self.proteins)
                df.append(p.diff)

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

    def analyze_me(self):
        Network.current = self
        init = [p.val for p in self.proteins]
        self.time = np.linspace(0, 200, num=16384)
        try:
            self.graph, info = odeint(self.diff_closure(), init, self.time, hmin=0.0001, hmax=1, printmessg=False, full_output=True)
            return int(info['message'] != 'Integration successful.')
            # HERE should go the evaluation of the result
        except Exception as e:
            return 100000

    def plot_me(self):
        for i, _ in enumerate(self.proteins):
            g = self.graph[:, i]
            plt.plot(self.time, g)

        plt.show()

def mutation_stuff():
    networks = []
    for _ in range(N_NETS_BEFORE):
        n = Network()
        n.mutate()
        networks.append(n)

    while True:
        analysis = sorted([(n, n.analyze_me()) for n in networks], key=lambda x: x[1])

        networks = []
        for (ntw, _) in analysis[:N_NETS_AFTER]:
            ntw.plot_me()
            multiply = N_NETS_BEFORE / N_NETS_AFTER
            assert(multiply == round(multiply))

            for i in range(int(multiply)):
                n = deepcopy(ntw)
                n.mutate()
                networks.append(n)

if __name__ == "__main__":
    mutation_stuff()
