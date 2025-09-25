from graphblas import Vector, Matrix, binary, semiring
from graphblas import dtypes
import sys


class MultiSourceBFS:
    def __init__(self):
        self.A = None
        self.result = None
        self.parent = None

    def stop(self):
        pass

    def load(self, filename):
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

    def run(self, sources):
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

        iteration = 0

        while front.nvals > 0:
            next_parents = front.mxm(self.A, semiring.ss.any_secondi)

            next_parents = next_parents.ewise_mult(not_visited)
            nonzero_mask = next_parents.apply(binary.ne, 0)
            cleaned = Matrix(next_parents.dtype, next_parents.nrows, next_parents.ncols)
            cleaned(mask=nonzero_mask) << next_parents
            iteration += 1

            if cleaned.nvals == 0:
                break

            not_visited(mask=cleaned.S) << False
            parent(mask=cleaned.S) << cleaned

            front = Matrix(bool, sources_number, n)
            front(mask=cleaned.S) << True

        print(f"Iterations: {iteration}")
        self.parent = parent
        return (self.parent, iteration)


if __name__ == "__main__":
    bfs = MultiSourceBFS()

    bfs.load(filename=sys.argv[1])

    sources = list(map(int, sys.argv[2].split(",")))

    bfs.run(sources)

    print(bfs.parent.to_coo())
