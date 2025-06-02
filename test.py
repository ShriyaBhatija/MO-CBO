import numpy as np

interventional_data = np.load(f'./Data/mo-cbo-credit/mo-cbo/0/observations.pkl', allow_pickle=True)
print(interventional_data)

print(interventional_data['X7'].mean())
print(interventional_data['X7'].std())