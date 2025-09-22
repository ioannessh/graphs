import sys
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, LongType


class DijkstraSSSP:
    def __init__(self, app_name="Dijkstra-SSSP", master="local[*]"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("ERROR")
        self.INF = float('inf')

    def stop(self):
        self.spark.stop()

    def load_edges(self, path):
        return self.spark.read.text(path) \
            .select(F.split(F.col("value"), r"\s+").alias("cols")) \
            .filter(F.size("cols") >= 2) \
            .select(
                F.col("cols").getItem(0).cast("long").alias("src"),
                F.col("cols").getItem(1).cast("long").alias("dst"),
                F.when(F.size("cols") >= 3, F.col("cols").getItem(2).cast("float")).otherwise(1.0).alias("weight")
            )\
            .filter(F.col("weight") >= 0) \
            .persist(StorageLevel.MEMORY_ONLY)


    def run_dijkstra(self, edges_df, source):
        edges = edges_df \
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

        for iteration in range(total_vertices):
            unvisited_distances = distances \
                .join(visited, "vertex", "left_anti") \
                .filter(F.col("distance") < self.INF)

            if unvisited_distances.isEmpty():
                break

            min_vertex_row = unvisited_distances.orderBy("distance").first()
            current_vertex = min_vertex_row["vertex"]

            new_visited = self.spark.createDataFrame([(current_vertex,)], ["vertex"])
            visited = visited.union(new_visited).distinct().persist(StorageLevel.MEMORY_AND_DISK)

            outgoing_edges = edges.filter(F.col("from_vertex") == current_vertex)

            if not outgoing_edges.isEmpty():
                updated_distances = outgoing_edges \
                    .join(
                        distances.withColumnRenamed("distance", "current_distance"),
                        outgoing_edges["from_vertex"] == distances["vertex"]) \
                    .select(
                        F.col("to_vertex").alias("vertex"),
                        (F.col("current_distance") + F.col("weight")).alias("new_distance")
                    )

                combined_distances = (distances.alias("distances")
                    .join(updated_distances.alias("updated_distances"), "vertex", "full_outer")
                    .select(
                        F.coalesce(F.col("distances.vertex"), F.col("updated_distances.vertex")).alias("vertex"),
                        F.coalesce(F.col("distance"), F.lit(self.INF)).alias("old_distance"),
                        F.coalesce(F.col("new_distance"), F.lit(self.INF)).alias("new_distance"))
                    .select(
                        "vertex",
                        F.least(F.col("old_distance"), F.col("new_distance")).alias("distance")
                    )
                )

                distances.unpersist()
                distances = combined_distances.persist(StorageLevel.MEMORY_AND_DISK)

        edges.unpersist()
        visited.unpersist()
        return distances.orderBy("vertex")

def main():
    path = sys.argv[1]
    source = int(sys.argv[2])

    dijkstra = DijkstraSSSP()

    edges = dijkstra.load_edges(path)
    result = dijkstra.run_dijkstra(edges, source)

    result.show(20, truncate=False)

    dijkstra.stop()


if __name__ == "__main__":
    main()
