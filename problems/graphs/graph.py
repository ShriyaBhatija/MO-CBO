import abc


class GraphStructure:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def define_SEM():
        raise NotImplementedError("Subclass should implement this.")
    
    @abc.abstractmethod
    def get_targets():
        raise NotImplementedError("Subclass should implement this.")
    
    @abc.abstractmethod
    def get_exploration_sets():
        raise NotImplementedError("Subclass should implement this.")
    
    @abc.abstractmethod
    def get_set_MOBO():
        raise NotImplementedError("Subclass should implement this.")
    
    @abc.abstractmethod
    def get_interventional_ranges():
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_cost_structure(self):
        raise NotImplementedError("Subclass should implement this.")
