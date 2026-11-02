
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, to_date, sum as spark_sum,
    count, avg, when, lit
)
from pyspark.sql.window import Window


def analyze_transactions(csv_path):

    spark = (
        SparkSession.builder
        .appName("BankingTransactionAnalytics")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    try:
        # 1. Read CSV using PySpark
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(csv_path)
        )

        # 2. Clean the dataset
        df = df.dropna(subset=[
            "transaction_id",
            "customer_id",
            "transaction_date",
            "amount",
            "transaction_type",
            "status"
        ])

        # 3. Convert date and amount columns
        df = (
            df.withColumn(
                "transaction_date",
                to_timestamp(col("transaction_date"))
            )
            .withColumn("amount", col("amount").cast("double"))
            .dropna(subset=["transaction_date", "amount"])
            .filter(col("amount") >= 0)
            .withColumn(
                "transaction_day",
                to_date(col("transaction_date"))
            )
        )

        # 4. Count daily transactions per customer
        window_spec = Window.partitionBy(
            "customer_id", "transaction_day"
        )

        df = df.withColumn(
            "daily_transaction_count",
            count("transaction_id").over(window_spec)
        )

        # 5. Flag suspicious transactions using simple rules
        df = df.withColumn(
            "suspicious_reason",
            when(
                col("amount") >= 50000,
                lit("High Value Transaction")
            ).when(
                col("daily_transaction_count") >= 5,
                lit("Repeated Daily Transactions")
            ).otherwise(
                lit("Normal")
            )
        )

        df = df.withColumn(
            "is_suspicious",
            when(col("suspicious_reason") != "Normal", 1)
            .otherwise(0)
        )

        # 6. Aggregate by transaction type
        type_summary = (
            df.groupBy("transaction_type")
            .agg(
                count("*").alias("total_transactions"),
                spark_sum("amount").alias("total_amount"),
                avg("amount").alias("average_amount")
            )
            .orderBy("transaction_type")
            .toPandas()
        )

        # 7. Aggregate daily transactions
        daily_summary = (
            df.groupBy("transaction_day")
            .agg(
                count("*").alias("total_transactions"),
                spark_sum("amount").alias("total_amount")
            )
            .orderBy("transaction_day")
            .toPandas()
        )

        # 8. Customer-wise analysis
        customer_summary = (
            df.groupBy("customer_id")
            .agg(
                count("*").alias("total_transactions"),
                spark_sum("amount").alias("total_amount")
            )
            .orderBy(col("total_amount").desc())
            .toPandas()
        )

        # 9. Convert processed records for Streamlit
        transactions = df.toPandas()

        return (
            transactions,
            type_summary,
            daily_summary,
            customer_summary
        )

    finally:
        spark.stop()
