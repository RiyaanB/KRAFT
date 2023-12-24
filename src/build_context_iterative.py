from src.kraft import *
import networkx as nx

def build_context_iterative(llm, embedding_model, question, choose_type='classic', choose_count=3, max_depth=2, max_branching=2):
    """
    Iteratively constructs a context from a given question using a large language model (LLM) and an embedding model.

    Args:
    llm (LanguageModel): An instance of a large language model used for text processing.
    embedding_model (EmbeddingModel): An embedding model used for nearest neighbor calculations. Required if choose_type is 'nearest_neighbor'.
    question (str): The input question or query to process.
    choose_type (str): The method for choosing properties, either 'classic' or 'nearest_neighbor'.
    choose_count (int): The number of properties to consider in the context-building process.
    max_depth (int): The maximum depth to traverse in the context-building process.

    Raises:
    ValueError: If embedding_model is None and choose_type is 'nearest_neighbor'.

    Returns:
    str: A string containing the constructed information context.
    """

    if embedding_model is None and choose_type == 'nearest_neighbor':
        raise ValueError("Must provide embedding_model when using nearest_neighbor choose_type")

    # Step 1: Extract entities from the question
    entity_labels = kw.extract_entity_labels(llm, question)
    entity_ids = kw.get_entity_ids(entity_labels)
    info_context = ""
    info_graph = nx.DiGraph()

    # Step 2: Keep a set of processed entities to avoid repeats
    # Also keep a queue of entities yet to process
    processed_entities = set()
    entity_queue = [(entity, 0) for entity in entity_ids]  # Tuple of (entity_id, depth)

    # Step 3: Iterate through the queue of entities, while queue is not empty
    while entity_queue:
        print(entity_queue)

        # Step 4: Pop the first entity from the queue
        subject_qid, depth = entity_queue.pop(0)
        if subject_qid in processed_entities or depth > max_depth:
            # Ensure it's not a repeat and not too separated from the root entities
            continue
        processed_entities.add(subject_qid)

        # Step 5: Get the entity dictionary and snaks
        subject_label = kw.get_entity_dict(subject_qid)['labels']['en']['value']
        entity_dict = kw.get_entity_dict(subject_qid)
        snaks = kw.get_snaks_with_labels(entity_dict)
        description = entity_dict.get('descriptions', {}).get('en', {}).get('value', '')

        # Step 6: Add the entity to the context
        info_context += f"NEW SUBJECT:  {subject_label}: {description} \n"
        info_graph.add_node(subject_qid, label=subject_label, description='')

        print("Subject: " + subject_label)

        branch_count = 0
        # Step 7: Choose properties for this entity
        edge_pids = kw.choose_properties(llm, embedding_model, question, entity_dict, snaks, choose_type=choose_type, choose_count=choose_count)
        for edge_pid in edge_pids:
            if edge_pid not in snaks:
                print("Continuing due to no snaks")
                continue
            # Step 8: Add the property and object to the context
            edge_label = snaks[edge_pid]["label"]
            print(edge_label)
            for object_entity in snaks[edge_pid]['entities']:
                
                # Process the entity based on its type
                # If it's an actual entity, add it to the queue for further processing

                if 'time' in object_entity:
                    try:
                        object_label = object_entity['time']
                        info_graph.add_node('Time', label=object_label, description='')
                        print("Is a time, so didn't add to queue")
                    except:
                        continue
                elif 'entity-type' in object_entity and object_entity['entity-type'] == 'item':
                    object_qid = object_entity['id']
                    try:
                        if branch_count > max_branching:
                            break
                        object_label = kw.get_entity_dict(object_qid)['labels']['en']['value']
                        if object_qid and object_qid not in processed_entities:
                            entity_queue.append((object_qid, depth + 1))
                            info_graph.add_node(object_qid, label=object_label, description='')
                            info_graph.add_edge(subject_qid, object_qid, label=edge_label)
                            branch_count += 1
                            print(f"[{subject_label}] Added to queue: {object_qid} ({object_label})")
                        else:
                            print("Failed due to repeat")
                    except:
                        print("Failed due to exception")
                        continue
                else:
                    print("Something else")
                    continue
                
                # Add the property and object to the context
                info_context += f"{edge_label}: {object_label} \n"

    # Return the constructed context
    return info_context, info_graph