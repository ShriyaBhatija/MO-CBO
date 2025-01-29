def list_interventional_ranges(dict_ranges, intervention_variables):
    list_min_ranges = []
    list_max_ranges = []
    for j in range(len(intervention_variables)):
      list_min_ranges.append(dict_ranges[intervention_variables[j]][0])
      list_max_ranges.append(dict_ranges[intervention_variables[j]][1])
    return list_min_ranges, list_max_ranges


def get_interventional_dict(intervention_variables):
    interventional_dict = {}
    for i in range(len(intervention_variables)):
      interventional_dict[intervention_variables[i]] = ''
    return interventional_dict
