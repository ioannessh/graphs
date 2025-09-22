from graphblas import Vector, Matrix, binary, semiring
from graphblas import dtypes
import sys


class MultiSourceBFS:
    def __init__(self):
        self.A = None
        self.result = None
        self.parent = None

    def load_graph(self, filename):
        edges = []
        max_vertex = 0

        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            for line in lines:
                src, dst = map(int, line.strip().split())
                edges.append((src, dst))
                max_vertex = max(max_vertex, src, dst)

        n = max_vertex + 1
        self.A = Matrix(dtypes.BOOL, n, n)
        for src, dst in edges:
            self.A[src, dst] = True

    def run_bfs(self, sources):
        n = self.A.nrows
        sources_number = len(sources)

        front = Matrix(bool, sources_number, n)
        parent = Matrix(int, sources_number, n)
        not_visited = Matrix(int, sources_number, n)


        for i, source in enumerate(sources):
            front[i, source] = True
            parent[i, source] = source
            for j in range(n):
                not_visited[i, j] = True
            not_visited[i, source] = False

        step = 0

        while front.nvals > 0 and step < 5:
            next_parents = front.mxm(self.A, semiring.ss.any_secondi)

            next_parents.ewise_mult(not_visited)

            if next_parents.nvals == 0:
                break

            not_visited(mask=next_parents.S) << False
            parent(mask=next_parents.S) << next_parents

            front = Matrix(bool, sources_number, n)
            front(mask=next_parents.S) << True

            step += 1
        self.parent = parent


if __name__ == "__main__":
    bfs = MultiSourceBFS()

    bfs.load_graph(filename=sys.argv[1])

    sources = list(map(int, sys.argv[2].split(",")))

    bfs.run_bfs(sources)

    print(bfs.parent.to_coo())
