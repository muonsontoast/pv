import cothread
from cothread.catools import caget, caput
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
import jax.numpy as jnp
'''
# ding dong
import os
os.environ['PATH'] += ':/Library/TeX/texbin'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times'
plt.rcParams['font.size'] = 12
'''

class PV:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'PV')
        self.READID = kwargs.get('READID', '')
        self.SETID = kwargs.get('SETID', '')
        self.default_value = 0
        self.tolerance = kwargs.get('tolerance', [0, 0])
        self.units = kwargs.get('units', '')

    def __str__(self):
        attrs = [v for v in self.__params() if v[0] != 'name']
        print('==' * 30)
        print(self.name, 'PV\n')
        for attr in attrs:
            print(f"{' '.join(attr[0].split('_')).title()}: {attr[1]}") if attr[0] != 'ID' else print(f"{' '.join(attr[0].split('_'))}: {attr[1]}")
        print('==' * 30)
        return ''
    
    def put(self, samples = 1, timeout = .2):
        if self.ID == '':
            print('No PV ID supplied!')
            return np.zeros(0)
        v = np.zeros(samples)
        for i in range(samples):
            try: v[i] = caget(self.ID)
            except:
                print('The following PV ID doesn\'t exist:', self.ID)
                return np.zeros(0)
            cothread.Sleep(timeout)
        return v

    def get(self):
        if self.ID == '':
            print('No PV ID supplied!')
            return np.zeros(0)
        return caget(self.ID)
    
    def __params(self):
        return [(k, v) for k, v in vars(self).items() if not callable(v)]

def perform_single_measurement(input, outputPV, PVs, timeout, numRepeats = 0, hysteresisCompensation = .2):
    cothread.Sleep(1)
    result = jnp.zeros(numRepeats + 1)
    for i, pv in enumerate(PVs):
        if pv.units.lower() == 'amps' and caget(pv.READID) < input[i]:
            caput(pv.SETID, input[i] - hysteresisCompensation)
    cothread.Sleep(timeout)
    for i, pv in enumerate(PVs):
        caput(pv.SETID, input[i])
    cothread.Sleep(1)
    for _ in range(numRepeats + 1):
        result = result.at[_].set(caget(outputPV.READID))
        cothread.Sleep(.2)
    return result
    
def perform_measurement(inputs, output_PV, PVs, timeout, repeats):
    caput('LI-TI-MTGEN-01:START', 1) # start the LINAC
    num_inputs, num_PVs = len(inputs), len(PVs)
    samples, variances, v = np.zeros(num_inputs), np.zeros(num_inputs), np.zeros(repeats)
    true_values = np.zeros((num_inputs, repeats))

    try:
        for i in range(num_inputs):
            current_PV_values = [caget(PVs[i].SETID) for i in range(num_PVs)]
            for j in range(num_PVs):
                if PVs[j].units.lower() == 'amps':
                    if inputs[i][j] < current_PV_values[j]:
                        caput(PVs[j].SETID, inputs[i][j] - .2)
                        current_PV_values[j] = inputs[i][j]
                        cothread.Sleep(timeout) # allow the magnet time to ramp
                # print(f'i: {i}, j: {j}', inputs[i][j])
                caput(PVs[j].SETID, inputs[i][j])
            cothread.Sleep(1)
            cothread.Sleep(timeout)
            for k in range(repeats):
                cothread.Sleep(.2)
                v[k] = caget(output_PV.READID)
                true_values[i, k] = v[k]
            print("Step %d/%d complete!" % (i + 1, num_inputs)) if len(inputs) > 1 else None
    except:
        print('An error caused the script to crash.')
        pass

    for pv in PVs:
        caput(pv.SETID, pv.default_value)

    print('Stopped Injection!')
    caput('LI-TI-MTGEN-01:STOP', 1) # stop the LINAC

    return true_values

def grid_scan(samples_per_dimension, repeats, timeout, output_PV, *PVs):
    '''Perform a grid scan over PVs.\n
    PARAMS:\n
        `samples_per_dimension` = resolution along each axis in the format `(x, y, ...)` (uniform if a single number is given).\n
        `repeats` = # of times to repeat each measurement.\n
        `timeout` = time to wait between measurements in seconds.\n
        `output_PV` = PV to measure.\n
        `PVs` = a sequence of PVs in the format `PV1`, `PV2`, ...'''
    t = time.time()
    num_PVs = len(PVs)
    if num_PVs == 0: print('Err: No PVs supplied!'); return
    samples_per_dimension = samples_per_dimension[::-1] if type(samples_per_dimension) in [np.ndarray, list, tuple] else samples_per_dimension
    if type(samples_per_dimension) in [int, float]:
        samples_per_dimension = int(samples_per_dimension) * np.ones(num_PVs, dtype = int)
    elif type(samples_per_dimension) in [list, tuple]:
        if len(samples_per_dimension) == 1:
            samples_per_dimension = int(samples_per_dimension[0]) * np.ones(num_PVs, dtype = int)
        elif len(samples_per_dimension) < num_PVs: print('There are more PVs than the length of <samples_per_dimension>'); return
    else: print('Err: <samples_per_dimension> should be either an int, list or tuple!'); return
    samples_per_dimension = samples_per_dimension * np.ones(num_PVs, dtype = int) if type(samples_per_dimension) in [int, float] else samples_per_dimension[:num_PVs]
    inputs = list((itertools.product(*[np.linspace(PVs[i].tolerance[0], PVs[i].tolerance[1], samples_per_dimension[i]) for i in range(num_PVs)])))
    print('Taking measurements ... please wait ...')
    data = perform_measurement(inputs, output_PV, PVs, timeout, repeats)
    samples = np.mean(data, axis = 1)
    variances = np.var(data, axis = 1)
    print('The scan took {:.2f} seconds.'.format(time.time() - t))
    
    if num_PVs < 3:
        fig = plt.figure(figsize = (5, 4));
        ax = fig.add_subplot(111)
        ax.set_xlabel(PVs[0].name) if PVs[0].units == '' else ax.set_xlabel(f'{PVs[0].name} ({PVs[0].units})')
        if num_PVs == 1:
            ax.minorticks_on()
            ax.set_ylabel(output_PV.name) if output_PV.units == '' else ax.set_ylabel(f'{output_PV.name} ({output_PV.units})')
            ax.plot(np.linspace(PVs[0].tolerance[0], PVs[0].tolerance[1], samples_per_dimension[0]), samples)
            ax.errorbar(np.linspace(PVs[0].tolerance[0], PVs[0].tolerance[1], samples_per_dimension[0]), samples, yerr = np.ravel(np.sqrt(np.sqrt(variances))), color = 'tab:orange', label = r'$\mu_{x_i}\pm\sigma_{x_i}$', markersize = 2.5, zorder = 5, capsize = 2.5, fmt = 'o', linestyle = '')
            ax.set_xlim(PVs[0].tolerance[0], PVs[0].tolerance[1])
        else:
            units = '\\' + PVs[1].units
            ax.set_ylabel(PVs[1].name) if PVs[1].units == '' else ax.set_ylabel(f'{PVs[1].name} ({PVs[1].units})')
            im = ax.imshow(samples, extent = [*PVs[0].tolerance, *PVs[1].tolerance], aspect = 'auto')
            cb = plt.colorbar(im)
            cb.set_label(output_PV.name) if output_PV.units == '' else cb.set_label(f'{output_PV.name} ({output_PV.units})', rotation = 270, labelpad = 15)
        
        fig.savefig('grid-scan.png', dpi = 300, bbox_inches = 'tight', transparent = True)
        plt.close(fig)

    return data
