import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api
from tqdm import tqdm

# Define the function to get the label for a single item
def get_label(pid_or_qid):
    try:
        entity_dict = get_entity_dict_from_api(pid_or_qid)
        label = entity_dict['labels']['en']['value']
        description = entity_dict['descriptions']['en']['value']
        aliases = [alias['value'] for alias in entity_dict['aliases']['en']]
        return pid_or_qid, label, description, aliases
    except Exception as e:
        # If there's an error, we return None
        return pid_or_qid, None, None, None

# The main function to make the dictionary
def make_pid_dict(output_file, max_workers=16):
    pid_to_list = {}
    # Use ThreadPoolExecutor to parallelize the API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare the futures
        futures = {executor.submit(get_label, "P" + str(i)): i for i in range(0, 12500)}
        # Process as they complete
        for future in tqdm(as_completed(futures), total=len(futures)):
            pid, label, description, aliases = future.result()
            if label is not None:
                pid_to_list[pid] = [label, description, aliases]
    
    # Write the results to a file
    with open(output_file, 'w') as f:
        json.dump(pid_to_list, f)

# Now we can call make_dict
make_pid_dict('pid_to_label.json')
