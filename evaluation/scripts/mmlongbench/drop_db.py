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


def drop_database(driver, db_name):
    query = f"DROP DATABASE `{db_name}`"
    with driver.session() as session:
        session.run(query)
        print(f"Database '{db_name}' has been deleted.")


def delete_bench_databases(driver):
    databases = get_databases(driver)
    for db in databases:
        if db.startswith("mm-long-bench-single-"):
            drop_database(driver, db)


if __name__ == "__main__":
    driver = create_driver()
    try:
        delete_bench_databases(driver)
    except ServiceUnavailable as e:
        print(f"Error: {e}")
    finally:
        driver.close()
