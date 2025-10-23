import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "iaarlichunyu"


def create_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_databases(driver):
    query = "SHOW DATABASES"
    with driver.session() as session:
        result = session.run(query)
        databases = [record["name"] for record in result]
    return databases


def is_database_empty(driver, db_name):
    query = "MATCH (n) RETURN count(n) AS cnt"
    with driver.session(database=db_name) as session:
        result = session.run(query).single()
        node_count = result["cnt"]
    return node_count == 0


def count_empty_bench_databases(driver):
    databases = get_databases(driver)
    empty_count = 0
    empty_dbs = []
    for db in databases:
        if db.startswith("mm-long-bench-single-"):
            try:
                if is_database_empty(driver, db):
                    empty_count += 1
                    empty_dbs.append(db)
            except Exception as e:
                print(f"⚠️ Error checking database {db}: {e}")
    print(f"共有 {empty_count} 个空的 mm-long-bench-single-* 数据库")
    if empty_dbs:
        print("空数据库列表:")
        for db in empty_dbs:
            print(f"  - {db}")


if __name__ == "__main__":
    driver = create_driver()
    try:
        count_empty_bench_databases(driver)
    except ServiceUnavailable as e:
        print(f"Error: {e}")
    finally:
        driver.close()
