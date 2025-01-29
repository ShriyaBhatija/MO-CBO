import numpy as np

def define_initial_data_MOCBO(interventional_data, num_interventions, exploration_set, name_index):

    data_list = []
    data_x_list = []
    data_y_list = []

    for j in range(len(exploration_set)):
      data = interventional_data[j].copy()
      num_variables = data[0]
      if num_variables == 1:
        data_x = np.asarray(data[(num_variables+1)])
        data_y = np.asarray(data[-1])
      else:
        data_x = np.asarray(data[(num_variables+1):(num_variables*2)][0])
        data_y = np.asarray(data[-1])

      if len(data_y.shape) == 1:
          data_y = data_y[:,np.newaxis]

      if len(data_x.shape) == 1:
          data_x = data_x[:,np.newaxis]

      all_data = np.concatenate((data_x, data_y), axis =1)

      ## Need to reset the global seed 
      state = np.random.get_state()

      np.random.seed(name_index)
      np.random.shuffle(all_data)

      np.random.set_state(state)

      subset_all_data = all_data[:num_interventions]
      data_list.append(subset_all_data)
      data_x_list.append(data_list[j][:, :num_variables])

      if len(data_y.shape) == 1:
          data_y_list.append(data_list[j][:, num_variables:][:,np.newaxis])
      else:
          data_y_list.append(data_list[j][:, num_variables:])

    return data_x_list, data_y_list
