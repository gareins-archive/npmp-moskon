from enum import Enum
import random

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Change factor
NO_CHG_TYPE = 0.9
CHG_FACTOR = 0.5

class Type(Enum):
    linear = 1
    encime = 2
    active = 3

D_TYPES = (Type.linear, Type.linear, Type.linear, Type.linear, Type.encime, Type.encime, Type.active)
ARP_TYPES = (Type.linear, Type.linear, Type.encime)

class Expression:
    def __init__(self):
        self._typ = Type.linear
        self._factor = 1
        self._limit = 0
        self._othr = None

    def set_othr(self, othr):
        self._othr = othr

    def mutate(self, othr, new_type):
        if random.random() < NO_CHG_TYPE:
            if CHG_FACTOR or self.Type != self.encime:
                self._factor *= random.random() * 2
            else:
                self._limit += (random.random() * 3 - 1)
                if self._limit < 0:
                    self._limit = 0
        else:
            self._typ = new_type
            self._othr = othr
            self._factor = 1
            self._limit = 0

    def get_random_typ(self, types):
        new_type = random.choice(types)
        while new_type == self._typ:
            new_type = random.choice(types)

        return new_type

class ActRep(Expression):
    def calc_diff_out(self, prot, proteins):
        diff = 0
        if self._typ == Type.linear:
            diff = self._factor
        elif self._typ == Type.encime:
            ra = 1 if proteins[self._othr].val - prot.val < 0 else 0
            ra = ra if self._factor > 0 else 1 - ra
            diff = self._factor * ra
        else:
            print("BAD_332")
        return diff

    def mutate(self, othr):
        new_type = super(ActRep, self).get_random_typ(ARP_TYPES)
        super(ActRep, self).mutate(othr, new_type)

class Activator(ActRep):
    def calc_diff(self, prot, proteins):
        prot.diff += self.calc_diff_out(prot, proteins)
class Repressor(ActRep):
    def calc_diff(self, prot, proteins):
        prot.diff -= self.calc_diff_out(prot, proteins)

class Degradation(Expression):
    def calc_diff(self, prot, proteins):
        if self._typ == Type.linear:
            diff = prot.val * self._factor
        elif self._typ == Type.encime:
            diff = self._factor * prot.val / (self._limit + prot.val)
        else:
            diff = self._factor * prot.val * proteins[self._othr].val
        prot.diff -= diff

    def mutate(self, othr):
        new_type = super(ActRep, self).get_random_typ(D_TYPES)
        super(ActRep, self).mutate(othr, new_type)

class ProteinModification(Expression):
    def calc_diff(self, prot, proteins):
        if self._typ == Type.linear:
            diff = self._factor * prot.val
        elif self._typ == Type.encime:
            diff = self._factor * prot.val / (self._limit + prot.val)
        else:
            print("BAD_123")
        prot.diff -= diff
        proteins[self._othr].diff += diff

    def mutate(self, othr):
        new_type = super(ActRep, self).get_random_typ(ARP_TYPES)
        super(ActRep, self).mutate(othr, new_type)

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

    def add_expression_pm(self, proteins):
        for prot in random.sample(proteins, len(proteins)):
            if prot not in self.expressions and prot != self:
                new_expr = ProteinModification()
                new_expr.set_othr(prot)
                self.expressions[prot] = new_expr
                break

    def rm_expression_pm(self):
        if len(self.expressions == 3):
            return

        for k in random.sample(self.expressions.keys(), len(self.expressions)):
            if k not in ("AR1", "AR2", "DEG"):
                self.expressions[k] = None

    def mutate(self, proteins, k=None):
        if k is None:
            k = random.choice(self.expressions.keys())
        exp = self.expressions[k]

        if k not in ("AR1", "AR2", "DEG"):



    def remove_protein(self, prot, proteins):
        chk = {"ACT": Activator(), "DEG": Degradation(), "REP": Repressor}
        for k, e in chk.items():
            if self.expressions[k]._othr == protein:
                self.expressions[k] = e
                self.expressions[k].mutate(proteins)

        self.expressions[protein] = None

class Network:
    def __init__(self):
        self.proteins = [Protein() for _ in range(3)]
        self.mutation_rate = 5

    def mutate(self):
        for i in range self.mutation_rate():
            p = random.choice(self.proteins)
            self.proteins



if __name__ == "__main__":
    def eq(X, t):
        x, y = X
        return [x - y - np.exp(t), x + y + 2 * np.exp(t)]

    init = [-1.0, -1.0]
    t = np.linspace(0, 4, num=50)
    X = odeint(eq, init, t)

    x = X[:, 0]
    y = X[:, 1]

    plt.plot(t, x, 'k--')
    plt.plot(t, y, 'k:')

    plt.show()
