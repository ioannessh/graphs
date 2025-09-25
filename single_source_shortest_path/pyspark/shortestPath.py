import sys
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, LongType


class DijkstraSSSP:
    def __init__(self, app_name="Dijkstra-SSSP"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()
            # .config("spark.sql.autoBroadcastJoinThreshold", "-1") \

        self.spark.sparkContext.setLogLevel("ERROR")
        self.spark.sparkContext.setCheckpointDir("/tmp/checkpoint")
        self.INF = float('inf')

    def stop(self):
        if self.edges_df is not None:
            self.edges_df.unpersist()
        # self.spark.stop()

    def load(self, path):
        self.edges_df = self.spark.read.text(path) \
            .select(F.split(F.col("value"), r"\s+").alias("cols")) \
            .filter(F.size("cols") >= 2) \
            .select(
                F.col("cols").getItem(0).cast("long").alias("src"),
                F.col("cols").getItem(1).cast("long").alias("dst"),
                F.when(F.size("cols") >= 3, F.col("cols").getItem(2).cast("float")).otherwise(1.0).alias("weight")
            )\
            .filter(F.col("weight") >= 0) \
            .persist(StorageLevel.MEMORY_ONLY)

    def run(self, source):
        edges = self.edges_df \
            .select(
                F.col("src").alias("from_vertex"),
                F.col("dst").alias("to_vertex"),
                "weight"
            )\
            .persist(StorageLevel.MEMORY_AND_DISK)

        vertices = edges.select("from_vertex").union(edges.select("to_vertex")).distinct()
        total_vertices = vertices.count()

        distances = vertices.select(
            F.col("from_vertex").alias("vertex"),
            F.when(F.col("from_vertex") == source, 0.0).otherwise(self.INF).alias("distance")
        ).persist(StorageLevel.MEMORY_AND_DISK)

        visited = self.spark.createDataFrame([], StructType([
            StructField("vertex", LongType(), True)
        ]))
        it = 0
        for iteration in range(min(40, total_vertices)):
            it = iteration
            unvisited_distances = (distances.filter(F.col("distance") < self.INF)) \
                .join(visited, "vertex", "left_anti")

            print(f"Iteration: {iteration}/{total_vertices}")

            if unvisited_distances.isEmpty():
                break

            min_vertex_row = min(unvisited_distances.collect(), key=lambda r: r["distance"])
            current_vertex = min_vertex_row["vertex"]

            new_visited = visited \
                .union(self.spark.createDataFrame([(current_vertex,)], ["vertex"])) \
                .repartition(200, "vertex") \
                .persist(StorageLevel.MEMORY_AND_DISK)
            visited.unpersist()
            visited = new_visited
            # if iteration % 5 == 0:
            visited = visited.checkpoint(eager=True)

            outgoing_edges = edges.filter(F.col("from_vertex") == current_vertex).persist(StorageLevel.MEMORY_AND_DISK)

            if not outgoing_edges.isEmpty():
                updated_distances = outgoing_edges \
                    .join(
                        distances.withColumnRenamed("distance", "current_distance"),
                        outgoing_edges["from_vertex"] == distances["vertex"]) \
                    .select(
                        F.col("to_vertex").alias("vertex"),
                        (F.col("current_distance") + F.col("weight")).alias("new_distance")
                    ) \
                    .persist(StorageLevel.MEMORY_AND_DISK)

                combined_distances = distances.alias("distances") \
                    .join(updated_distances.alias("updated_distances"), "vertex", "left") \
                    .select(
                        F.coalesce(F.col("distances.vertex"), F.col("updated_distances.vertex")).alias("vertex"),
                        F.coalesce(F.col("distance"), F.lit(self.INF)).alias("old_distance"),
                        F.coalesce(F.col("new_distance"), F.lit(self.INF)).alias("new_distance")) \
                    .select(
                        "vertex",
                        F.least(F.col("old_distance"), F.col("new_distance")).alias("distance")
                    )

                distances.unpersist()
                updated_distances.unpersist()
                distances = combined_distances.persist(StorageLevel.MEMORY_AND_DISK).checkpoint(eager=True)
            outgoing_edges.unpersist()

        edges.unpersist()
        visited.unpersist()
        distances.unpersist()
        return (distances.orderBy("vertex"), it)

def main():
    path = sys.argv[1]
    source = int(sys.argv[2])

    dijkstra = DijkstraSSSP()

    dijkstra.load(path)
    result = dijkstra.run(source)

    result.show(20, truncate=False)

    dijkstra.stop()


if __name__ == "__main__":
    main()
