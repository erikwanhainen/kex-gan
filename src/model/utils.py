import yaml

def load_params(): 
    with open('./config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params 

