import sys
from graphblas import Matrix, Vector, binary, semiring, monoid
from graphblas import dtypes
import numpy as np

class DijkstraSSSP:
    def __init__(self):
        self.A = None

    def stop(self):
        pass

    def load(self, file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        sources, targets, weights = [], [], []
        max_vertex = 0

        for line in lines:
            parts = line.split()
            s, t = int(parts[0]), int(parts[1])
            w = float(parts[2]) if len(parts) >= 3 else 1.0

            sources.append(s)
            targets.append(t)
            weights.append(w)

            if s > max_vertex:
                max_vertex = s
            if t > max_vertex:
                max_vertex = t

        n = max_vertex + 1
        A = Matrix(dtypes.FP64, n, n)
        A.build(sources, targets, weights)

        self.A = A

    def run(self, source_vertex):
        n = self.A.nrows

        distances = Vector(dtypes.FP64, n)
        distances[:] = float('inf')
        distances[source_vertex] = 0.0

        visited = Vector(dtypes.BOOL, n)
        it = 0
        for iteration in range(n):
            it = iteration
            unvisited_distances = Vector(dtypes.FP64, n)
            unvisited_distances(~visited.S) << distances

            min_distance = unvisited_distances.reduce(monoid.min)

            print(f"Iteration: {iteration}/{n}")

            if min_distance == float('inf'):
                break

            min_candidates = unvisited_distances.apply(binary.eq, min_distance)

            current_vertex = None
            for i in min_candidates:
                if min_candidates.get(i, False):
                    current_vertex = i
                    break

            if current_vertex is None:
                break

            visited[current_vertex] = True

            update_vector = Vector(dtypes.FP64, n)
            update_vector[current_vertex] = distances[current_vertex]

            new_distances = semiring.min_plus[dtypes.FP64](self.A.T @ update_vector).new()

            for i in new_distances:
                if not visited.get(i, False):
                    new_dist = new_distances.get(i, float('inf'))
                    if new_dist < float('inf'):
                        old_dist = distances.get(i, float('inf'))
                        if new_dist < old_dist:
                            distances[i] = new_dist
        return (distances, it)


def main():
    path = sys.argv[1]
    source = int(sys.argv[2])

    dijkstra = DijkstraSSSP()

    dijkstra.load(path)
    result = dijkstra.run(source)

    print(result.to_coo())


if __name__ == "__main__":
    main()
