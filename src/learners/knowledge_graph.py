import heapq
import itertools

from src.envs.knowledge_graph import KnowledgeGraphModel


class KnowledgeGraphTranslator:
    def __init__(self, x_to_y):
        self.x_to_y = x_to_y
        self.score = 0
        self.nodewise_error = None
        self.edgewise_error = None

    def __call__(self, x: int) -> int:
        return self.x_to_y[x]

    def __lt__(self, other):
        return self.score < other.score


class KnowledgeGraphLearner:
    def __init__(self, env: KnowledgeGraphModel, n_top_translators: int):
        self.env = env
        self.translators = [
            KnowledgeGraphTranslator(permutation)
            for permutation in itertools.permutations(range(self.env.n_target), r=self.env.n_source)
        ]
        self.n_top_translators = n_top_translators
        self.validation_errors = None

    def update_validation_errors(self):
        if self.n_top_translators > 1:
            top_translators = heapq.nlargest(self.n_top_translators, self.translators)
        else:
            top_translators = [max(self.translators)]
        self.validation_errors = {'node': [], 'edge': []}
        for translator in top_translators:
            if translator.nodewise_error is None:
                assert translator.edgewise_error is None
                translator.nodewise_error = self.env.nodewise_error(translator)
                translator.edgewise_error = self.env.edgewise_error(translator)
            self.validation_errors['node'].append(translator.nodewise_error)
            self.validation_errors['edge'].append(translator.edgewise_error)

    def train_iter(self, x1: int, x2: int):
        for translator in self.translators:
            if self.env.P[translator(x1), translator(x2)]:
                translator.score += 1
        self.update_validation_errors()
