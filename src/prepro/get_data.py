import requests
import json

QURERY_PARAMS = 'species:mouse,rat&fq=cell_type:pyramidal&fq=brain_region:neocortex'

def get_swc(archive_name, CNG_version, neruon_name, neuron_id):
    url = 'http://neuromorpho.org/dableFiles/' + archive_name.lower() + '/' + CNG_version + '/' + neruon_name + '.CNG.swc'
    response = requests.get(url)
    with open(r'./swc_files/' + str(neuron_id) + '.CNG.swc', 'wb') as f:
        f.write(response.content)


def get_page(page_number):
    url = 'http://neuromorpho.org/api/neuron/select?q=' + QURERY_PARAMS + '&page=' + str(page_number)
    response = requests.get(url)
    return response


def get_neruon_data(amount):
    count = 0
    url = 'http://neuromorpho.org/api/neuron/select?q=' + QURERY_PARAMS
    response = requests.get(url)
    if response.status_code == 200:
        json_response = json.loads(response.content)
        total_pages = json_response['page']['totalPages']
        for page_number in range(total_pages):
            print('getting page:', page_number)
            page = get_page(page_number)
            if page.status_code != 200:
                continue
            else:
                json_response = json.loads(page.content)
                for neuron in json_response['_embedded']['neuronResources']:
                    try:
                        print('getting swc, neruon_id =', neuron['neuron_id'])
                        get_swc(neuron['archive'], 'CNG%20version', neuron['neuron_name'], neuron['neuron_id'])
                        count += 1
                    except Exception as e:
                        print(e)
                    if count >= amount:
                        print('DONE')
                        return

if __name__ == '__main__':
    get_neruon_data(0)
