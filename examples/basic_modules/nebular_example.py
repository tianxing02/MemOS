import json
import os

from datetime import datetime

import numpy as np

from dotenv import load_dotenv

from memos.configs.embedder import EmbedderConfigFactory
from memos.configs.graph_db import GraphDBConfigFactory
from memos.embedders.factory import EmbedderFactory
from memos.graph_dbs.factory import GraphStoreFactory
from memos.memories.textual.item import TextualMemoryItem, TreeNodeTextualMemoryMetadata


load_dotenv()

embedder_config = EmbedderConfigFactory.model_validate(
    {
        "backend": "universal_api",
        "config": {
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
            "model_name_or_path": "text-embedding-3-large",
            "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        },
    }
)
embedder = EmbedderFactory.from_config(embedder_config)


def embed_memory_item(memory: str) -> list[float]:
    embedding = embedder.embed([memory])[0]
    embedding_np = np.array(embedding, dtype=np.float32)
    embedding_list = embedding_np.tolist()
    return embedding_list


def example_multi_db(db_name: str = "paper"):
    # Step 1: Build factory config
    config = GraphDBConfigFactory(
        backend="nebular",
        config={
            "hosts": json.loads(os.getenv("NEBULAR_HOSTS", "localhost")),
            "user_name": os.getenv("NEBULAR_USER", "root"),
            "password": os.getenv("NEBULAR_PASSWORD", "xxxxxx"),
            "space": db_name,
            "auto_create": True,
            "embedding_dimension": 3072,
            "use_multi_db": True,
        },
    )

    # Step 2: Instantiate the graph store
    graph = GraphStoreFactory.from_config(config)
    graph.clear()

    # Step 3: Create topic node
    topic = TextualMemoryItem(
        memory="This research addresses long-term multi-UAV navigation for energy-efficient communication coverage.",
        metadata=TreeNodeTextualMemoryMetadata(
            memory_type="LongTermMemory",
            key="Multi-UAV Long-Term Coverage",
            hierarchy_level="topic",
            type="fact",
            memory_time="2024-01-01",
            source="file",
            sources=["paper://multi-uav-coverage/intro"],
            status="activated",
            confidence=95.0,
            tags=["UAV", "coverage", "multi-agent"],
            entities=["UAV", "coverage", "navigation"],
            visibility="public",
            updated_at=datetime.now().isoformat(),
            embedding=embed_memory_item(
                "This research addresses long-term "
                "multi-UAV navigation for "
                "energy-efficient communication "
                "coverage."
            ),
        ),
    )

    graph.add_node(
        id=topic.id, memory=topic.memory, metadata=topic.metadata.model_dump(exclude_none=True)
    )


def example_shared_db(db_name: str = "shared-traval-group"):
    """
    Example: Single(Shared)-DB multi-tenant (logical isolation)
    Multiple users' data in the same Neo4j DB with user_name as a tag.
    """
    # users
    user_list = ["root"]

    for user_name in user_list:
        # Step 1: Build factory config
        config = GraphDBConfigFactory(
            backend="nebular",
            config={
                "hosts": json.loads(os.getenv("NEBULAR_HOSTS", "localhost")),
                "user_name": os.getenv("NEBULAR_USER", "root"),
                "password": os.getenv("NEBULAR_PASSWORD", "xxxxxx"),
                "space": db_name,
                "auto_create": True,
                "embedding_dimension": 3072,
                "use_multi_db": False,
            },
        )

        # Step 2: Instantiate graph store
        graph = GraphStoreFactory.from_config(config)
        print(f"\n[INFO] Working in shared DB: {db_name}, for user: {user_name}")
        graph.clear()

        # Step 3: Create topic node
        topic = TextualMemoryItem(
            memory="This research addresses long-term multi-UAV navigation for energy-efficient communication coverage.",
            metadata=TreeNodeTextualMemoryMetadata(
                memory_type="LongTermMemory",
                key="Multi-UAV Long-Term Coverage",
                hierarchy_level="topic",
                type="fact",
                memory_time="2024-01-01",
                source="file",
                sources=["paper://multi-uav-coverage/intro"],
                status="activated",
                confidence=95.0,
                tags=["UAV", "coverage", "multi-agent"],
                entities=["UAV", "coverage", "navigation"],
                visibility="public",
                updated_at=datetime.now().isoformat(),
                embedding=embed_memory_item(
                    "This research addresses long-term "
                    "multi-UAV navigation for "
                    "energy-efficient communication "
                    "coverage."
                ),
            ),
        )

        graph.add_node(
            id=topic.id, memory=topic.memory, metadata=topic.metadata.model_dump(exclude_none=True)
        )

        # Step 4: Add a concept for each user
        concept = TextualMemoryItem(
            memory=f"Itinerary plan for {user_name}",
            metadata=TreeNodeTextualMemoryMetadata(
                memory_type="LongTermMemory",
                key="Multi-UAV Long-Term Coverage",
                hierarchy_level="concept",
                type="fact",
                memory_time="2024-01-01",
                source="file",
                sources=["paper://multi-uav-coverage/intro"],
                status="activated",
                confidence=95.0,
                tags=["UAV", "coverage", "multi-agent"],
                entities=["UAV", "coverage", "navigation"],
                visibility="public",
                updated_at=datetime.now().isoformat(),
                embedding=embed_memory_item(f"Itinerary plan for {user_name}"),
            ),
        )

        graph.add_node(
            id=concept.id,
            memory=concept.memory,
            metadata=concept.metadata.model_dump(exclude_none=True),
        )

        # Link concept to topic
        graph.add_edge(source_id=concept.id, target_id=topic.id, type="RELATE_TO")
        print(f"[INFO] Added nodes for {user_name}")

        # Step 5: Query and print ALL for verification
    print("\n=== Export entire DB (for verification, includes ALL users) ===")
    graph = GraphStoreFactory.from_config(config)
    all_graph_data = graph.export_graph()
    print(all_graph_data)

    # Step 6: Search for alice's data only
    print("\n=== Search for travel_member_alice ===")
    config_alice = GraphDBConfigFactory(
        backend="nebular",
        config={
            "hosts": json.loads(os.getenv("NEBULAR_HOSTS", "localhost")),
            "user_name": os.getenv("NEBULAR_USER", "root"),
            "password": os.getenv("NEBULAR_PASSWORD", "xxxxxx"),
            "space": db_name,
            "embedding_dimension": 3072,
        },
    )
    graph_alice = GraphStoreFactory.from_config(config_alice)
    nodes = graph_alice.search_by_embedding(vector=embed_memory_item("travel itinerary"), top_k=1)
    for node in nodes:
        print(graph_alice.get_node(node["id"]))


if __name__ == "__main__":
    print("\n=== Example: Multi-DB ===")
    example_multi_db(db_name="paper")

    print("\n=== Example: Single-DB ===")
    example_shared_db(db_name="shared_traval_group")
