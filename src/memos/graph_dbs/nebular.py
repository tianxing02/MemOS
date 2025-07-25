from datetime import datetime
from typing import Any, Literal

from nebulagraph_python.value_wrapper import ValueWrapper

from memos.configs.graph_db import NebulaGraphDBConfig
from memos.dependency import require_python_package
from memos.graph_dbs.base import BaseGraphDB
from memos.log import get_logger


logger = get_logger(__name__)


def _escape_str(value: str) -> str:
    return value.replace('"', '\\"')


def _format_value(val: Any, key: str = "") -> str:
    if isinstance(val, str):
        return f'"{_escape_str(val)}"'
    elif isinstance(val, (int | float)):
        return str(val)
    elif isinstance(val, datetime):
        return f'datetime("{val.isoformat()}")'
    elif isinstance(val, list):
        if key == "embedding":
            dim = len(val)
            joined = ",".join(str(float(x)) for x in val)
            return f"VECTOR<{dim}, FLOAT>([{joined}])"
        else:
            return f"[{', '.join(_format_value(v) for v in val)}]"
    elif val is None:
        return "NULL"
    else:
        return f'"{_escape_str(str(val))}"'


def _format_datetime(value: str | datetime) -> str:
    """Ensure datetime is in ISO 8601 format string."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class NebulaGraphDB(BaseGraphDB):
    """
    NebulaGraph-based implementation of a graph memory store.
    """

    @require_python_package(
        import_name="nebulagraph_python",
        install_command="pip install ... @Tianxing",
        install_link=".....",
    )
    def __init__(self, config: NebulaGraphDBConfig):
        """
        NebulaGraph DB client initialization.

        Required config attributes:
        - hosts: list[str] like ["host1:port", "host2:port"]
        - user: str
        - password: str
        - space: str (optional for basic commands)

        Example config:
            {
                "hosts": ["xxx.xx.xx.xxx:xxxx"],
                "user": "root",
                "password": "nebula",
                "space": "test"
            }
        """
        from nebulagraph_python.client import NebulaClient

        self.config = config
        self.client = NebulaClient(
            hosts=config.get("hosts"),
            username=config.get("user_name"),
            password=config.get("password"),
        )
        self.space = config.get("space")
        self.user_name = config.user_name
        self.system_db_name = "system" if config.use_multi_db else config.space
        if config.auto_create:
            self._ensure_database_exists()

        # Create only if not exists
        self.create_index(dimensions=config.embedding_dimension)

        logger.info("Connected to NebulaGraph successfully.")

    def create_index(
        self,
        label: str = "Memory",
        vector_property: str = "embedding",
        dimensions: int = 1536,
        index_name: str = "memory_vector_index",
    ) -> None:
        """raise NotImplementedError"""

    def get_memory_count(self, memory_type: str) -> int:
        raise NotImplementedError

    def count_nodes(self, scope: str) -> int:
        raise NotImplementedError

    def remove_oldest_memory(self, memory_type: str, keep_latest: int) -> None:
        raise NotImplementedError

    def add_node(self, id: str, memory: str, metadata: dict[str, Any]) -> None:
        """
        Insert or update a Memory node in NebulaGraph.
        """
        if not self.config.use_multi_db and self.config.user_name:
            metadata["user_name"] = self.config.user_name

        now = datetime.utcnow()
        metadata = metadata.copy()
        metadata.setdefault("created_at", now)
        metadata.setdefault("updated_at", now)
        metadata["node_type"] = metadata.pop("type")
        metadata["id"] = id
        metadata["memory"] = memory

        properties = ", ".join(f"{k}: {_format_value(v, k)}" for k, v in metadata.items())
        gql = f"INSERT OR IGNORE (n@Memory {{{properties}}})"

        try:
            self.client.execute(gql)
        except Exception as e:
            logger.error(f"Failed to insert vertex {id}: {e}")

    def update_node(self, id: str, fields: dict[str, Any]) -> None:
        """
        Update node fields in Nebular, auto-converting `created_at` and `updated_at` to datetime type if present.
        """
        fields = fields.copy()
        set_clauses = []
        for k, v in fields.items():
            set_clauses.append(f"n.{k} = {_format_value(v, k)}")

        set_clause_str = ",\n    ".join(set_clauses)

        query = f"""
            MATCH (n@Memory {{id: "{id}"}})
            """

        if not self.config.use_multi_db and self.config.user_name:
            query += f'WHERE n.user_name = "{self.config.user_name}"'

        query += f"\nSET {set_clause_str}"
        self.client.execute(query)

    def delete_node(self, id: str) -> None:
        raise NotImplementedError

    # Edge (Relationship) Management
    def add_edge(self, source_id: str, target_id: str, type: str):
        """
        Create an edge from source node to target node.
        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            type: Relationship type (e.g., 'RELATE_TO', 'PARENT').
        """
        if not source_id or not target_id:
            raise ValueError("[add_edge] source_id and target_id must be provided")

        props = ""
        if not self.config.use_multi_db and self.config.user_name:
            props = f'{{user_name: "{self.config.user_name}"}}'

        insert_stmt = f'''
               MATCH (a@Memory {{id: "{source_id}"}}), (b@Memory {{id: "{target_id}"}})
               INSERT (a) -[e@{type} {props}]-> (b)
           '''
        try:
            self.client.execute(insert_stmt)
        except Exception as e:
            logger.error(f"Failed to insert edge: {e}", exc_info=True)

    def delete_edge(self, source_id: str, target_id: str, type: str) -> None:
        raise NotImplementedError

    def edge_exists(
        self, source_id: str, target_id: str, type: str = "ANY", direction: str = "OUTGOING"
    ) -> bool:
        raise NotImplementedError

    # Graph Query & Reasoning
    def get_node(self, id: str) -> dict[str, Any] | None:
        """
        Retrieve a Memory node by its unique ID.

        Args:
            id (str): Node ID (Memory.id)

        Returns:
            dict: Node properties as key-value pairs, or None if not found.
        """
        gql = f"""
               USE memory_graph
               MATCH (v {{id: '{id}'}})
               RETURN v
           """

        try:
            result = self.client.execute(gql)
            record = result.one_or_none()
            if record is None:
                return None

            node_wrapper = record["v"].as_node()
            props = node_wrapper.get_properties()

            return {key: self._parse_node(val) for key, val in props.items()}

        except Exception as e:
            logger.error(f"[get_node] Failed to retrieve node '{id}': {e}")
            return None

    def get_nodes(self, ids: list[str]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_edges(self, id: str, type: str = "ANY", direction: str = "ANY") -> list[dict[str, str]]:
        raise NotImplementedError

    def get_neighbors(
        self, id: str, type: str, direction: Literal["in", "out", "both"] = "out"
    ) -> list[str]:
        raise NotImplementedError

    def get_neighbors_by_tag(
        self,
        tags: list[str],
        exclude_ids: list[str],
        top_k: int = 5,
        min_overlap: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Find top-K neighbor nodes with maximum tag overlap.

        Args:
            tags: The list of tags to match.
            exclude_ids: Node IDs to exclude (e.g., local cluster).
            top_k: Max number of neighbors to return.
            min_overlap: Minimum number of overlapping tags required.

        Returns:
            List of dicts with node details and overlap count.
        """
        if not tags:
            return []

        tag_filter_str = ", ".join(f'"{tag}"' for tag in tags)

        gql = """
                MATCH (n@Memory)
                WHERE n.status = "activated"
                  AND n.node_type <> "reasoning"
                  AND n.memory_type <> "WorkingMemory"
            """

        if exclude_ids:
            exclude_str = ", ".join(f'"{eid}"' for eid in exclude_ids)
            gql += f" AND id(n) NOT IN [{exclude_str}]"

        if self.config.user_name and not self.config.use_multi_db:
            gql += f' AND n.user_name = "{self.config.user_name}"'
            gql += f"""
                WITH n, n.tags AS tags
                UNWIND tags AS tag
                WITH n, tag
                WHERE tag IN [{tag_filter_str}]
                WITH n, count(*) AS overlap_count
                WHERE overlap_count >= {min_overlap}
                RETURN n, overlap_count
                ORDER BY overlap_count DESC
                LIMIT {top_k}
                """

        print("===== GQL:\n", gql)
        try:
            result = self.client.execute(gql)
            return [
                {**self._parse_node(row.values[0]), "overlap_count": row.values[1].cast(int)}
                for row in result.rows()
            ]
        except Exception as e:
            logger.error(f"Failed to get neighbors by tag: {e}", exc_info=True)
            return []

    def get_children_with_embeddings(self, id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[str]:
        raise NotImplementedError

    def get_subgraph(
        self, center_id: str, depth: int = 2, center_status: str = "activated"
    ) -> dict[str, Any]:
        """
        Retrieve a local subgraph centered at a given node.
        Args:
            center_id: The ID of the center node.
            depth: The hop distance for neighbors.
            center_status: Required status for center node.
        Returns:
            {
                "core_node": {...},
                "neighbors": [...],
                "edges": [...]
            }
        """
        if not 1 <= depth <= 5:
            raise ValueError("depth must be 1-5")

        user_name = self.config.user_name
        gql = f"""
             MATCH (center@Memory)
            WHERE center.id = '{center_id}'
              AND center.status = '{center_status}'
              AND center.user_name = '{user_name}'
            OPTIONAL MATCH p = (center)-[e]->{{1,{depth}}}(neighbor@Memory)
            WHERE neighbor.user_name = '{user_name}'
            RETURN center,
                   collect(DISTINCT neighbor) AS neighbors,
                   collect(EDGES(p)) AS edge_chains
            """

        result = self.client.execute(gql).one_or_none()  # 执行查询
        if not result or result.size == 0:
            return {"core_node": None, "neighbors": [], "edges": []}

        core_node = self._parse_node(result["center"])
        neighbors = [self._parse_node(n) for n in result["neighbors"].value]
        edges = []
        for rel_chains in result["edge_chains"].value:
            for chain in rel_chains.value:
                edge = chain.value
                edges.append(
                    {
                        "type": edge.get_type(),
                        "source": edge.get_src_id(),
                        "target": edge.get_dst_id(),
                    }
                )

        return {"core_node": core_node, "neighbors": neighbors, "edges": edges}

    def get_context_chain(self, id: str, type: str = "FOLLOWS") -> list[str]:
        raise NotImplementedError

    # Search / recall operations
    def search_by_embedding(
        self,
        vector: list[float],
        top_k: int = 5,
        scope: str | None = None,
        status: str | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Retrieve node IDs based on vector similarity.

        Args:
            vector (list[float]): The embedding vector representing query semantics.
            top_k (int): Number of top similar nodes to retrieve.
            scope (str, optional): Memory type filter (e.g., 'WorkingMemory', 'LongTermMemory').
            status (str, optional): Node status filter (e.g., 'active', 'archived').
                            If provided, restricts results to nodes with matching status.
            threshold (float, optional): Minimum similarity score threshold (0 ~ 1).

        Returns:
            list[dict]: A list of dicts with 'id' and 'score', ordered by similarity.

        Notes:
            - This method uses Neo4j native vector indexing to search for similar nodes.
            - If scope is provided, it restricts results to nodes with matching memory_type.
            - If 'status' is provided, only nodes with the matching status will be returned.
            - If threshold is provided, only results with score >= threshold will be returned.
            - Typical use case: restrict to 'status = activated' to avoid
            matching archived or merged nodes.
        """
        dim = len(vector)
        vector_str = ",".join(f"{float(x)}" for x in vector)
        gql_vector = f"VECTOR<{dim}, FLOAT>([{vector_str}])"

        where_clauses = []
        if scope:
            where_clauses.append(f'n.memory_type = "{scope}"')
        if status:
            where_clauses.append(f'n.status = "{status}"')
        if not self.config.use_multi_db and self.config.user_name:
            where_clauses.append(f'n.user_name = "{self.config.user_name}"')

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        gql = f"""
               USE memory_graph
               MATCH (n@Memory)
               {where_clause}
               ORDER BY euclidean(n.embedding, {gql_vector}) ASC
               APPROXIMATE
               LIMIT {top_k}
               OPTIONS {{ METRIC: L2, TYPE: IVF, NPROBE: 8 }}
               RETURN n.id AS id, euclidean(n.embedding, {gql_vector}) AS score
           """

        try:
            result = self.client.execute(gql)
        except Exception as e:
            logger.error(f"[search_by_embedding] Query failed: {e}")
            return []

        try:
            output = []
            for row in result:
                values = row.values()
                id_val = values[0].as_string()
                score_val = values[1].as_double()
                if threshold is None or score_val <= threshold:
                    output.append({"id": id_val, "score": score_val})
            return output
        except Exception as e:
            logger.error(f"[search_by_embedding] Result parse failed: {e}")
            return []

    def get_by_metadata(self, filters: list[dict[str, Any]]) -> list[str]:
        raise NotImplementedError

    def get_grouped_counts(
        self,
        group_fields: list[str],
        where_clause: str = "",
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Count nodes grouped by any fields.

        Args:
            group_fields (list[str]): Fields to group by, e.g., ["memory_type", "status"]
            where_clause (str, optional): Extra WHERE condition. E.g.,
            "WHERE n.status = 'activated'"
            params (dict, optional): Parameters for WHERE clause.

        Returns:
            list[dict]: e.g., [{ 'memory_type': 'WorkingMemory', 'status': 'active', 'count': 10 }, ...]
        """
        if not group_fields:
            raise ValueError("group_fields cannot be empty")

            # GQL-specific modifications
        if not self.config.use_multi_db and self.config.user_name:
            user_clause = f"n.user_name = '{self.config.user_name}'"
            if where_clause:
                where_clause = where_clause.strip()
                if where_clause.upper().startswith("WHERE"):
                    where_clause += f" AND {user_clause}"
                else:
                    where_clause = f"WHERE {where_clause} AND {user_clause}"
            else:
                where_clause = f"WHERE {user_clause}"

        # Inline parameters if provided
        if params:
            for key, value in params.items():
                # Handle different value types appropriately
                if isinstance(value, str):
                    value = f"'{value}'"
                where_clause = where_clause.replace(f"${key}", str(value))

        return_fields = []
        group_by_fields = []

        for field in group_fields:
            alias = field.replace(".", "_")  # 防止特殊字符
            return_fields.append(f"n.{field} AS {alias}")
            group_by_fields.append(alias)
        # Full GQL query construction
        gql = f"""
            MATCH (n)
            {where_clause}
            RETURN {", ".join(return_fields)}, COUNT(n) AS count
            GROUP BY {", ".join(group_by_fields)}
            """
        result = self.client.execute(gql)  # Pure GQL string execution

        output = []
        for record in result:
            group_values = {}
            for i, field in enumerate(group_fields):
                value = record.values()[i].as_string()
                group_values[field] = value
            count_value = record["count"].value
            output.append({**group_values, "count": count_value})

        return output

    # Structure Maintenance
    def deduplicate_nodes(self) -> None:
        raise NotImplementedError

    def detect_conflicts(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def merge_nodes(self, id1: str, id2: str) -> str:
        raise NotImplementedError

    # Utilities
    def clear(self) -> None:
        """
        raise NotImplementedError
        """

    def export_graph(self) -> dict[str, Any]:
        """
        Export all graph nodes and edges in a structured form.

        Returns:
            {
                "nodes": [ { "id": ..., "memory": ..., "metadata": {...} }, ... ],
                "edges": [ { "source": ..., "target": ..., "type": ... }, ... ]
            }
        """
        node_query = "MATCH (n@Memory)"
        edge_query = "MATCH (a@Memory)-[r]->(b@Memory)"

        if not self.config.use_multi_db and self.config.user_name:
            username = self.config.user_name
            node_query += f' WHERE n.user_name = "{username}"'
            edge_query += f' WHERE r.user_name = "{username}"'

        try:
            full_node_query = f"{node_query} RETURN n"
            node_result = self.client.execute(full_node_query)
            nodes = []
            for row in node_result:
                node_wrapper = row.values()[0].as_node()
                props = node_wrapper.get_properties()

                metadata = {key: self._parse_node(val) for key, val in props.items()}

                memory = metadata.get("memory", "")

                nodes.append({"id": node_wrapper.get_id(), "memory": memory, "metadata": metadata})
        except Exception as e:
            raise RuntimeError(f"[EXPORT GRAPH - NODES] Exception: {e}") from e

        try:
            full_edge_query = f"{edge_query} RETURN a.id AS source, b.id AS target"
            edge_result = self.client.execute(full_edge_query)
            edges = [
                {"source": row.values()[0].value, "target": row.values()[1].value}
                for row in edge_result
            ]
        except Exception as e:
            raise RuntimeError(f"[EXPORT GRAPH - EDGES] Exception: {e}") from e

        return {"nodes": nodes, "edges": edges}

    def import_graph(self, data: dict[str, Any]) -> None:
        raise NotImplementedError

    def get_all_memory_items(self, scope: str) -> list[dict]:
        raise NotImplementedError

    def get_structure_optimization_candidates(self, scope: str) -> list[dict]:
        raise NotImplementedError

    def drop_database(self) -> None:
        raise NotImplementedError

    def _ensure_database_exists(self):
        create_tag = """
            CREATE GRAPH TYPE IF NOT EXISTS MemoryGraphType AS {
            NODE Memory (:MemoryTag {
                id STRING,
                memory STRING,
                user_name STRING,
                created_at STRING,
                updated_at STRING,
                status STRING,
                node_type STRING,
                memory_time STRING,
                source STRING,
                confidence FLOAT,
                entities LIST<STRING>,
                tags LIST<STRING>,
                visibility STRING,
                memory_type STRING,
                key STRING,
                sources LIST<STRING>,
                usage LIST<STRING>,
                background STRING,
                hierarchy_level STRING,
                embedding VECTOR<3072, FLOAT>,
                PRIMARY KEY(id)
            }),
            EDGE RELATE_TO (Memory) -[{user_name STRING}]-> (Memory),
            EDGE PARENT (Memory) -[{user_name STRING}]-> (Memory)
        }
        """

        create_graph = "CREATE GRAPH IF NOT EXISTS memory_graph TYPED MemoryGraphType"
        set_graph_working = "SESSION SET GRAPH memory_graph"
        create_vector_index = """
                CREATE VECTOR INDEX IF NOT EXISTS memory_vector_index
                ON NODE Memory::embedding
                OPTIONS {
                    DIM: 3072,
                    METRIC: L2,
                    TYPE: IVF,
                    NLIST: 100,
                    TRAINSIZE: 1000
                }
                FOR memory_graph
            """
        try:
            self.client.execute(create_tag)
            self.client.execute(create_graph)
            self.client.execute(set_graph_working)
            self.client.execute(create_vector_index)
            logger.info("✅ Graph `memory_graph` is now the working graph.")
        except Exception as e:
            logger.error(f"❌ Failed to create tag: {e}")

    def _vector_index_exists(self, index_name: str = "memory_vector_index") -> bool:
        """raise NotImplementedError"""

    def _create_vector_index(
        self, label: str, vector_property: str, dimensions: int, index_name: str
    ) -> None:
        """raise NotImplementedError"""

    def _create_basic_property_indexes(self) -> None:
        """raise NotImplementedError"""

    def _index_exists(self, index_name: str) -> bool:
        """raise NotImplementedError"""

    def _parse_node(self, value: ValueWrapper) -> Any:
        if value is None or value.is_null():
            return None
        try:
            primitive_value = value.cast_primitive()
        except Exception as e:
            logger.warning(f"cast_primitive failed for value: {value}, error: {e}")
            try:
                primitive_value = value.cast()
            except Exception as e2:
                logger.warning(f"cast failed for value: {value}, error: {e2}")
                return str(value)

        if isinstance(primitive_value, ValueWrapper):
            return self._parse_node(primitive_value)

        if isinstance(primitive_value, list):
            return [
                self._parse_node(v) if isinstance(v, ValueWrapper) else v for v in primitive_value
            ]

        return primitive_value
