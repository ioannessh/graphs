import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, split
from pyspark.storagelevel import StorageLevel
from pyspark.sql.types import StructType, StructField, IntegerType

class MultiSourceBFS:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("MultiSourceBFS") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()

        self.spark.sparkContext.setCheckpointDir("/tmp/checkpoint")

        self.edges = None

    def load_edges(self, file_path):
        edges = self.spark.read.text(file_path) \
            .select(split(col("value"), r"\s+").alias("cols")) \
            .filter(size("cols") == 2) \
            .select(
                col("cols").getItem(0).cast("long").alias("src"),
                col("cols").getItem(1).cast("long").alias("dst")
            )

        reverse_edges = edges.select(
            col("dst").alias("src"),
            col("src").alias("dst")
        )
        self.edges = edges.union(reverse_edges).distinct().repartition("src").persist(StorageLevel.MEMORY_ONLY)

    def run_bfs(self, source_vertexs):
        schema = StructType([
            StructField("vertex", IntegerType(), nullable=False),
            StructField("dist", IntegerType(), nullable=False),
            StructField("parent", IntegerType(), nullable=True)
        ])
        frontier = self.spark.createDataFrame([(int(source),0,) for source in source_vertexs], ["vertex","dist"]).persist(StorageLevel.MEMORY_ONLY)
        visited = self.spark.createDataFrame([(int(source),0,None,) for source in source_vertexs], schema=schema).persist(StorageLevel.MEMORY_ONLY)

        frontier = frontier.checkpoint()
        visited = visited.checkpoint()

        num_partitions = max(1, self.spark.sparkContext.defaultParallelism)

        while True:
            candidates = frontier.alias("f") \
                .join(self.edges.alias("e"), col("f.vertex") == col("e.src"), "inner") \
                .select(
                    col("e.dst").alias("vertex"),
                    col("f.vertex").alias("parent"),
                    (col("f.dist") + 1).alias("dist")
                ) \
                .distinct()

            new_vertices = candidates.alias("c") \
                .join(visited.alias("v"), col("c.vertex") == col("v.vertex"), "left_anti") \
                .persist(StorageLevel.MEMORY_ONLY)

            if new_vertices.count() == 0:
                break

            new_visited = new_vertices.select("vertex", "dist", "parent")
            visited = visited.union(new_visited).distinct().repartition(num_partitions, "vertex").persist(StorageLevel.MEMORY_ONLY)
            frontier = new_vertices.select("vertex", "dist").repartition(num_partitions, "vertex").persist(StorageLevel.MEMORY_ONLY)

        return visited

    def stop(self):
        self.spark.stop()


def main():
    path = sys.argv[1]
    source = sys.argv[2].split(",")

    bfs = MultiSourceBFS()
    bfs.load_edges(path)
    result = bfs.run_bfs(source)

    result.orderBy("dist").show(20, truncate=False)

    bfs.stop()

if __name__ == "__main__":
    main()

# $env:PATH = "C:\Users\iashurenkov\Documents\Java\jdk-17.0.12\bin;" + $env:PATH
# $env:PATH = "C:\hadoop;" + $env:PATH
# $env:HADOOP_HOME = "C:\hadoop"
# $env:JAVA_HOME = "C:\Users\iashurenkov\Documents\Java\jdk-17.0.12"
# $env:PATH = "C:\Users\iashurenkov\AppData\Local\Programs\Python\Python312\python3.exe;" + $env:PATH