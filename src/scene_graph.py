def build_scene_graph(
    world,
    scene,
    memory
):
    """
    Build a semantic scene graph.

    Inputs:
        world  -> world_state.py
        scene  -> vlm_scene.py
        memory -> object_memory.py

    Returns:
        graph dictionary
    """

    graph = {

        "frame_id":
            world["timestamp"],

        "nodes": [],

        "relations": []
    }

    # --------------------------------------------------
    # ENVIRONMENT NODE
    # --------------------------------------------------

    graph["nodes"].append({

        "id": "environment",

        "type": scene["environment_type"],

        "attributes":
        world["environment"]
    })

    # --------------------------------------------------
    # HUMAN NODE
    # --------------------------------------------------

    if world["human"]["present"]:

        graph["nodes"].append({

            "id": "human",

            "type": "human",

            "distance":
                world["human"]["distance"],

            "proximity":
                world["human"]["proximity"]
        })

    # --------------------------------------------------
    # OBJECT MEMORY NODES
    # --------------------------------------------------

    memory_objects = memory.summary()[
        "objects"
    ]

    for obj in memory_objects:

        graph["nodes"].append({

            "id":
                obj["id"],

            "type":
                obj["type"],

            "status":
                obj["status"]
        })

    # --------------------------------------------------
    # ENVIRONMENT RELATIONS
    # --------------------------------------------------

    for obj in memory_objects:

        graph["relations"].append({

            "source":
                obj["id"],

            "relation":
                "inside",

            "target":
                "environment"
        })

    # --------------------------------------------------
    # HUMAN RELATIONS
    # --------------------------------------------------

    if world["human"]["present"]:

        for obj in memory_objects:

            graph["relations"].append({

                "source":
                    "human",

                "relation":
                    "near",

                "target":
                    obj["id"]
            })

    # --------------------------------------------------
    # NAVIGATION RELATION
    # --------------------------------------------------

    nav = scene["navigation"]

    if nav["aisle_detected"]:

        graph["relations"].append({

            "source":
                "robot",

            "relation":
                "can_navigate",

            "target":
                nav["walkable_region"]
        })

    return graph

def graph_summary(graph):

    lines = []

    lines.append(
        f"Frame: {graph['frame_id']}"
    )

    lines.append("\nNodes:")

    for node in graph["nodes"]:

        node_type = node.get(
            "type",
            "unknown"
        )

        lines.append(
            f"- {node['id']} ({node_type})"
        )

    lines.append("\nRelations:")

    for rel in graph["relations"]:

        lines.append(

            f"- {rel['source']} "
            f"{rel['relation']} "
            f"{rel['target']}"
        )

    return "\n".join(lines)

def scene_graph_summary(graph):

    return graph_summary(graph)
