from .StateSpacingProtocols import PlottingProtocol
import matplotlib.pyplot as plt
import numpy as np


class Plotting_Matplotlib(PlottingProtocol):
    
    def plot_states(self, filter_map, signal_components, state_only, *args):
        
        a, att, a_hat, y, T, Z = args

        filter_names = list(filter_map.values())
        filter_keys = list(filter_map.keys())

        # check if components in signal are in fact part of filters
        for component in signal_components:
            if not component in filter_names:
                raise ValueError(f'{component} not part of signal')
                
        # get indices of components passed to signal argument
        indxs = []
        for component in signal_components:
            index = filter_names.index(component)
            indxs.append(filter_keys[index])
        
        # number of plots is 1 (includes obeservations) + 1 for each filter in filter_map
        nfilters = len(filter_map.keys())
        num_plots = nfilters + 1
        
        # get z, y and a_hat in correct shape
        Z = np.squeeze(Z)
        y = np.squeeze(y)
        a_hat = np.squeeze(a_hat)

        # get dimension of state
        
        # create plot 
        fig, axs = plt.subplots(num_plots, sharex=True)
        fig.set_size_inches(15, 10)
        
        axs[0].set_title('Observation + Level (at index 0)')
        axs[0].plot(range(len(y)), y)
        axs[0].grid()
        
        # machanics of slicing depends on number of states
        if nfilters == 1:
            if not state_only:
                axs[0].plot(Z * a_hat)
            else:
                axs[0].plot(a_hat)
        else:
            if not state_only:
                signal = np.sum([Z[i, :] * a_hat[i, :] for i in indxs], axis=0)
                axs[0].plot(signal)
            else:
                signal = np.sum([a_hat[i, :] for i in indxs], axis=0)
                axs[0].plot(signal)
        
        # for each plot, plot the row of a_hat given in filter_map with the given title
        for idx, i in enumerate(filter_map.keys()):
            
            axs[idx + 1].set_title(filter_map[i])
            axs[idx + 1].grid()
            
            if nfilters == 1:
                axs[idx + 1].plot(a_hat)
            else:
                axs[idx + 1].plot(a_hat[i, :])

    def plot_state(self, state_name: str, *args, filter_type='smoothed'):
        """ Plots a particular state with is variance

        Args:
            state_name (str): which state you want to plot
            filter_type (str): what type of filter you want to plot
    
        Returns:
             A plot of filter mean and variance
        """

        a, att, a_hat, P, Ptt, P_hat, y, T, Z, filter_map= args
        
        # check if entered type is allowed
        allowed_types = ['smoothed', 'incasted', 'filtered']
        if filter_type not in allowed_types:
            raise ValueError(f'filter type is {filter_type} but must be in {allowed_types}')
        
        # check what type of filter to use -> set filter_ and filter_var accordingly
        if filter_type == 'smoothed':
            filter_ = a_hat
            filter_var = P_hat
            filter_label = r'$E(a_t|Y_n)$'
            filter_var_label = r'$var(a_t|Y_n)$'
        elif filter_type == 'incasted':
            filter_ = att
            filter_var = Ptt
            filter_label = r'$E(a_t|Y_t)$'
            filter_var_label = r'$var(a_t|Y_t)$'
        else:
            filter_ = a
            filter_var = P
            filter_label = r'$E(a_t|Y_{t-1})$'
            filter_var_label = r'$var(a_t|Y_{t-1})$'
        
        # get shapes
        filter_names = list(filter_map.values())
        filter_keys = list(filter_map.keys())
        
        # for example, if state_name is 'exogenous', check if 'exogenous' in filter_names, ow can't plot it
        if state_name not in filter_names:
            raise ValueError(f'{state_name} not part of filter_map')
        
        # get the index of the state name in the state vector
        index = filter_keys[filter_names.index(state_name)]
        
        # create plot 
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches(10, 10)
        
        # plot the filter and filter variance
        axs[0].plot(np.squeeze(filter_[index, :]), label=filter_label)
        axs[1].plot(np.squeeze(filter_var[index, index, :]), label=filter_var_label)
    
        # styling
        for ax in axs:
            ax.grid()
            ax.legend()
        
    def plot_forecast(*args, time=10):
        pass