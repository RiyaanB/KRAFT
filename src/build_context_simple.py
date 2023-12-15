from src.kraft import *


def build_context_simple(llm, embedding_model, question, choose_type='classic', choose_count=3):
    """
    Constructs a context from a given question using a large language model (LLM) and an embedding model.

    Args:
    llm (LanguageModel): An instance of a large language model used for text processing.
    embedding_model (EmbeddingModel): An embedding model used for nearest neighbor calculations. Required if choose_type is 'nearest_neighbor'.
    question (str): The input question or query to process.
    choose_type (str): The method for choosing properties, either 'classic' or 'nearest_neighbor'.
    choose_count (int): The number of properties to consider in the context-building process.

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

    # Step 2: Iterate through the entities
    for i, subject_qid in enumerate(entity_ids):

        # Step 3: Get the entity dictionary and snaks
        subject_label = entity_labels[i]
        entity_dict = kw.get_entity_dict(subject_qid)
        snaks = kw.get_snaks_with_labels(entity_dict)
        description = entity_dict.get('descriptions', {}).get('en', {}).get('value', '')
        info_context += f"NEW SUBJECT:  {subject_label}: {description} \n"

        # Step 4: Choose properties for this entity
        edge_pids = kw.choose_properties(llm, embedding_model, question, entity_dict, snaks, choose_type=choose_type, choose_count=choose_count)
        print(edge_pids)
        for edge_pid in edge_pids:
            if edge_pid not in snaks:
                continue

            edge_label = snaks[edge_pid]["label"]
            for object_entity in snaks[edge_pid]['entities']:
                # Process the entity based on its type
                if 'time' in object_entity:
                    object_label = object_entity['time']
                elif 'entity-type' in object_entity and object_entity['entity-type'] == 'item':
                    object_qid = object_entity['id']
                    try:
                        object_label = kw.get_entity_dict(object_qid)['labels']['en']['value']
                    except:
                        continue
                else:
                    continue

                # Add the property and object to the context
                info_context += f"{edge_label}: {object_label} \n"
    
    # Return the constructed context
    return info_context, None


