import pygmtools as pygm

# Some test utils functions
def data_from_numpy(*data, device=None):
    return_list = []
    for d in data:
        return_list.append(pygm.utils.from_numpy(d, device))
    if len(return_list) > 1:
        return return_list
    else:
        return return_list[0]


def data_to_numpy(*data):
    return_list = []
    for d in data:
        return_list.append(pygm.utils.to_numpy(d))
    if len(return_list) > 1:
        return return_list
    else:
        return return_list[0]
