import json
import os

from datetime import datetime, timezone

import numpy as np

from dotenv import load_dotenv

from memos.configs.embedder import EmbedderConfigFactory
from memos.configs.graph_db import GraphDBConfigFactory
from memos.embedders.factory import EmbedderFactory
from memos.graph_dbs.factory import GraphStoreFactory
from memos.memories.textual.item import TextualMemoryItem, TreeNodeTextualMemoryMetadata


load_dotenv()

gpt_config = {
    "backend": "universal_api",
    "config": {
        "provider": "openai",
        "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
        "model_name_or_path": "text-embedding-3-large",
        "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    },
}
nebular_config = {
    "hosts": json.loads(os.getenv("NEBULAR_HOSTS", "localhost")),
    "user_name": os.getenv("NEBULAR_USER", "root"),
    "password": os.getenv("NEBULAR_PASSWORD", "xxxxxx"),
    "space": "test_memory_count",
    "auto_create": True,
    "embedding_dimension": 3072,
    "use_multi_db": False,
}


embedder_config = EmbedderConfigFactory.model_validate(gpt_config)
embedder = EmbedderFactory.from_config(embedder_config)


def embed_memory_item(memory: str) -> list[float]:
    embedding = embedder.embed([memory])[0]
    embedding_np = np.array(embedding, dtype=np.float32)
    embedding_list = embedding_np.tolist()
    return embedding_list


now = datetime.now(timezone.utc).isoformat()
test_node1 = TextualMemoryItem(
    memory="This is a test node",
    metadata=TreeNodeTextualMemoryMetadata(
        memory_type="LongTermMemory",
        key="Research Topic",
        hierarchy_level="topic",
        type="fact",
        memory_time="2024-01-01",
        status="activated",
        visibility="public",
        updated_at=now,
        embedding=embed_memory_item("This is a test node"),
    ),
)

test_node2 = TextualMemoryItem(
    memory="This is another test node",
    metadata=TreeNodeTextualMemoryMetadata(
        memory_type="LongTermMemory",
        key="Research Topic",
        hierarchy_level="topic",
        type="fact",
        memory_time="2024-01-01",
        status="activated",
        visibility="public",
        updated_at=now,
        embedding=embed_memory_item("This is another test node"),
    ),
)


def test_get_memory_count():
    config = GraphDBConfigFactory(backend="nebular", config=nebular_config)
    graph = GraphStoreFactory.from_config(config)
    graph.clear()

    mem = test_node1
    graph.add_node(mem.id, mem.memory, mem.metadata.model_dump(exclude_none=True))

    count = graph.get_memory_count('"LongTermMemory"')  # quoting string literal for Cypher
    print("Memory Count:", count)
    assert count == 1


def test_count_nodes():
    graph = GraphStoreFactory.from_config(
        GraphDBConfigFactory(
            backend="nebular",
            config=nebular_config,
        )
    )
    graph.clear()

    # Insert two nodes
    for i in range(2):
        mem = TextualMemoryItem(
            memory=f"Memory {i}",
            metadata=TreeNodeTextualMemoryMetadata(
                memory_type="LongTermMemory",
                key="Research Topic",
                hierarchy_level="topic",
                type="fact",
                memory_time="2024-01-01",
                status="activated",
                visibility="public",
                updated_at=now,
                embedding=embed_memory_item(f"Memory {i}"),
            ),
        )
        graph.add_node(mem.id, mem.memory, mem.metadata.model_dump(exclude_none=True))

    count = graph.count_nodes('"LongTermMemory"')
    print("Node Count:", count)
    assert count == 2


def test_get_nodes():
    graph = GraphStoreFactory.from_config(
        GraphDBConfigFactory(backend="nebular", config=nebular_config)
    )
    graph.clear()

    mem = test_node1
    graph.add_node(mem.id, mem.memory, mem.metadata.model_dump(exclude_none=True))

    nodes = graph.get_nodes([mem.id])
    assert len(nodes) == 1
    assert nodes[0]["properties"]["id"] == mem.id


def test_edge_exists():
    graph = GraphStoreFactory.from_config(
        GraphDBConfigFactory(backend="nebular", config=nebular_config)
    )
    graph.clear()

    topic = TextualMemoryItem(
        memory="Edge topic",
        metadata=TreeNodeTextualMemoryMetadata(
            memory_type="LongTermMemory",
            key="Research Topic",
            hierarchy_level="topic",
            type="fact",
            memory_time="2024-01-01",
            status="activated",
            visibility="public",
            updated_at=now,
            embedding=embed_memory_item("Edge topic"),
        ),
    )

    concept = TextualMemoryItem(
        memory="Edge concept",
        metadata=TreeNodeTextualMemoryMetadata(
            memory_type="LongTermMemory",
            key="Research Topic",
            hierarchy_level="topic",
            type="fact",
            memory_time="2024-01-01",
            status="activated",
            visibility="public",
            updated_at=now,
            embedding=embed_memory_item("Edge concept"),
        ),
    )

    graph.add_node(topic.id, topic.memory, topic.metadata.model_dump(exclude_none=True))
    graph.add_node(concept.id, concept.memory, concept.metadata.model_dump(exclude_none=True))
    graph.add_edge(topic.id, concept.id, type="RELATE_TO")

    assert graph.edge_exists(topic.id, concept.id, type="RELATE_TO", direction="OUTGOING")


def test_get_edges():
    graph = GraphStoreFactory.from_config(
        GraphDBConfigFactory(backend="nebular", config=nebular_config)
    )
    graph.clear()

    source = TextualMemoryItem(
        memory="Source",
        metadata=TreeNodeTextualMemoryMetadata(
            memory_type="LongTermMemory",
            key="Research Topic",
            hierarchy_level="topic",
            type="fact",
            memory_time="2024-01-01",
            status="activated",
            visibility="public",
            updated_at=now,
            embedding=embed_memory_item("Source"),
        ),
    )
    target = TextualMemoryItem(
        memory="Target",
        metadata=TreeNodeTextualMemoryMetadata(
            memory_type="LongTermMemory",
            key="Research Topic",
            hierarchy_level="topic",
            type="fact",
            memory_time="2024-01-01",
            status="activated",
            visibility="public",
            updated_at=now,
            embedding=embed_memory_item("Target"),
        ),
    )
    graph.add_node(source.id, source.memory, source.metadata.model_dump(exclude_none=True))
    graph.add_node(target.id, target.memory, target.metadata.model_dump(exclude_none=True))
    graph.add_edge(source.id, target.id, type="PARENT")

    edges = graph.get_edges(source.id, type="PARENT", direction="OUTGOING")
    assert len(edges) == 1
    assert edges[0]["from"] == source.id
    assert edges[0]["to"] == target.id
    assert edges[0]["type"] == "PARENT"
