def filter_memory_data(memories_data):
    filtered_data = {}
    for key, value in memories_data.items():
        if key == "data" and "memory_detail_list" in value:
            # Handle new data structure
            filtered_data[key] = {}
            filtered_memories = []
            for memory_item in value["memory_detail_list"]:
                filtered_item = {
                    "id": memory_item.get("id"),
                    "memory_value": memory_item.get("memory_value"),
                    "metadata": {},
                }
                # Filter metadata
                if "metadata" in memory_item:
                    for k, v in memory_item["metadata"].items():
                        if k != "embedding":
                            filtered_item["metadata"][k] = v
                filtered_memories.append(filtered_item)
            filtered_data[key]["memory_detail_list"] = filtered_memories
            # Copy other fields in data if any
            for k, v in value.items():
                if k != "memory_detail_list":
                    filtered_data[key][k] = v

        elif key == "text_mem":
            filtered_data[key] = []
            for mem_group in value:
                # Check if it's the new data structure (list of TextualMemoryItem objects)
                if "memories" in mem_group and isinstance(mem_group["memories"], list):
                    # New data structure: directly a list of TextualMemoryItem objects
                    filtered_memories = []
                    for memory_item in mem_group["memories"]:
                        # Create filtered dictionary
                        filtered_item = {
                            "id": memory_item.id,
                            "memory": memory_item.memory,
                            "metadata": {},
                        }
                        # Filter metadata, excluding embedding
                        if hasattr(memory_item, "metadata") and memory_item.metadata:
                            for attr_name in dir(memory_item.metadata):
                                if not attr_name.startswith("_") and attr_name != "embedding":
                                    attr_value = getattr(memory_item.metadata, attr_name)
                                    if not callable(attr_value):
                                        filtered_item["metadata"][attr_name] = attr_value
                        filtered_memories.append(filtered_item)

                    filtered_group = {
                        "cube_id": mem_group.get("cube_id", ""),
                        "memories": filtered_memories,
                    }
                    filtered_data[key].append(filtered_group)
                else:
                    # Old data structure: dictionary with nodes and edges
                    filtered_group = {
                        "memories": {"nodes": [], "edges": mem_group["memories"].get("edges", [])}
                    }
                    for node in mem_group["memories"].get("nodes", []):
                        filtered_node = {
                            "id": node.get("id"),
                            "memory": node.get("memory"),
                            "metadata": {
                                k: v
                                for k, v in node.get("metadata", {}).items()
                                if k != "embedding"
                            },
                        }
                        filtered_group["memories"]["nodes"].append(filtered_node)
                    filtered_data[key].append(filtered_group)
        else:
            filtered_data[key] = value
    return filtered_data
