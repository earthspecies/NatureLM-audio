import requests
import json

def get_scientific_name(vernacular_name: str) -> str:
    """
    Queries the GBIF API to find the scientific name for a given vernacular name.

    Args:
        vernacular_name: The common name of the animal.

    Returns:
        The scientific name, or a 'Not found' message.
    """
    # The base URL for the GBIF species API's name matching endpoint
    api_url = f"https://api.gbif.org/v1/species/search?q={vernacular_name}"
    
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        data = response.json()

        for result in data.get("results", []):
            if vernacular_name in [name["vernacularName"].lower() for name in result.get("vernacularNames", [])]:
                return result.get("species", "Scientific name not found")
            
    except requests.exceptions.RequestException as e:
        return f"An API error occurred: {e}"


if __name__ == "__main__":
    # --- Example Usage ---
    # # A list of vernacular names for birds and aquatic animals
    # animal_names = ["Bald Eagle", "Spotted Elachura", "Dall's Porpoise"]

    # # Create a dictionary to store the results
    # name_mapping = {}

    # print("Mapping vernacular names to scientific names...")
    # for name in animal_names:
    #     scientific_name = get_scientific_name(str.lower(name))
    #     name_mapping[name] = scientific_name

    # # Print the results
    # for common, scientific in name_mapping.items():
    #     print(f"- {common}: {scientific}")

    import json

    with open("results/watkins_and_cbi_originals/beans_zero_eval_100_cbi.jsonl", 'r') as json_file:
        json_list = list(json_file)

    names = set()

    for json_str in json_list:
        result = json.loads(json_str)
        names.add(result['label'])

    # Convert the set to a list
    names_list = list(names)
    scientific_names = []
    for name in names_list:
        scientific_name = get_scientific_name(name.lower())
        scientific_names.append(scientific_name)
    
    # Save the scientific names to a new JSON file
    with open("cbi_vernacular_scientific_mapping.jsonl", 'w') as json_file:
        for name, scientific_name in zip(names_list, scientific_names):
            json.dump({"label": name, "scientific_name": scientific_name}, json_file)
            json_file.write('\n')