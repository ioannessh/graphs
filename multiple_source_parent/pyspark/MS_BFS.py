import sys
from functools import cache
from subprocess import check_output

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, split
from pyspark.storagelevel import StorageLevel
from pyspark.sql.types import StructType, StructField, IntegerType


class MultiSourceBFS:
    def __init__(self):
        self.spark = SparkSession.builder \
            .master("local[*]") \
            .appName("MultiSourceBFS") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()

        self.spark.sparkContext.setCheckpointDir("/tmp/checkpoint")
        self.use_checkpoint = True
        self.num_edges = 1
        self.edges = None

    def load(self, file_path):
        edges = self.spark.read.text(file_path) \
            .select(split(col("value"), r"\s+").alias("cols")) \
            .filter(size("cols") == 2) \
            .select(
                col("cols").getItem(0).cast("int").alias("src"),
                col("cols").getItem(1).cast("int").alias("dst")
            )

        reverse_edges = edges.select(
            col("dst").alias("src"),
            col("src").alias("dst")
        )
        self.edges = edges.union(reverse_edges).distinct().repartition(200, "src").cache()
        self.num_edges = self.edges.count()

    def run(self, source_vertices):
        num_partitions = max(1, min(200, (self.num_edges + 999) // 5000))
        schema = StructType([
            StructField("vertex", IntegerType(), nullable=False),
            StructField("dist", IntegerType(), nullable=False),
            StructField("parent", IntegerType(), nullable=True)
        ])
        frontier = self.spark.createDataFrame([(int(source), 0,) for source in source_vertices],
                                              ["vertex", "dist"])#.repartition(num_partitions, "vertex")
        visited = self.spark.createDataFrame([(int(source), 0, None,) for source in source_vertices],
                                             schema=schema)#.repartition(num_partitions, "vertex")

        # if self.use_checkpoint:
        #     frontier = frontier.checkpoint(eager=True)
        #     visited = visited.checkpoint(eager=True)

        iteration = 0
        while True:
            candidates = frontier.alias("f") \
                .join(self.edges.alias("e"), col("f.vertex") == col("e.src"), "inner") \
                .select(
                    col("e.dst").alias("vertex"),
                    col("f.vertex").alias("parent"),
                    (col("f.dist") + 1).alias("dist")
                )

            new_vertices = candidates.alias("c") \
                .join(visited.alias("v"), col("c.vertex") == col("v.vertex"), "left_anti") \
                .dropDuplicates(["vertex"]) \
                .repartition(num_partitions, "vertex") \
                .persist(StorageLevel.MEMORY_AND_DISK)

            if new_vertices.rdd.isEmpty():
                break

            new_visited = new_vertices \
                .select("vertex", "dist", "parent") \
                .union(visited) \
                .dropDuplicates(["vertex"]) \
                .repartition(num_partitions, "vertex") \
                .persist(StorageLevel.MEMORY_AND_DISK)

            if visited.is_cached:
                visited.unpersist()
            if frontier.is_cached:
                frontier.unpersist()

            visited = new_visited
            frontier = new_vertices.select("vertex", "dist") \
                 .dropDuplicates(["vertex"]) \
                 .repartition(num_partitions, "vertex") \
                 .persist(StorageLevel.MEMORY_AND_DISK)

            if new_vertices.is_cached:
                new_vertices.unpersist()

            if iteration % 3 == 0:
                if self.use_checkpoint:
                    visited = visited.checkpoint(eager=True)
                    frontier = frontier.checkpoint(eager=True)

            iteration += 1

        print(f"Iterations: {iteration}")
        if visited.is_cached:
            visited.unpersist()
        if frontier.is_cached:
            frontier.unpersist()
        return (visited, iteration)

    def stop(self):
        if self.edges is not None:
            self.edges.unpersist()


def main():
    path = sys.argv[1]
    source = sys.argv[2].split(",")

    bfs = MultiSourceBFS()
    bfs.load(path)
    result = bfs.run(source)

    result.orderBy("dist").show(20, truncate=False)

    bfs.stop()


if __name__ == "__main__":
    main()
