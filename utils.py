import os
import sys


def update_params(parameters, has_linear_algebra_backend, qd_degree=6):
    '''
    Select the Epetra linear algebra backend if it is
    compilled and set the CPP optimization of the forms
    to true. Reduce the 
    
    Parameters
    ----------
    parameters : FEniCS Parameter object
        Parameters object.
    
    qd_degree : optional, int
        Quadrature degree.


    Returns
    -------
    parameters : FEniCS Parameter object
        Updated parameters object.

    ffc_options : dict
        FFC options.
    '''
    if has_linear_algebra_backend("Epetra"):
        parameters["linear_algebra_backend"] = "Epetra"
    
    parameters["form_compiler"]["cpp_optimize"] = True
    
    ffc_options = {"quadrature_degree": qd_degree, "optimize": True}

    return parameters, ffc_options


def setup_case_directory(case, dir_results='./results'):
    '''
    Function to setup the directories to save the results.

    Parameters
    ----------
    case : string
        Name of the current case.
    dir_results : optional string
        The results directory

    Returns
    -------
    dir_case : string
        Path for the case results directory.
    '''
    
    # Check if the results directory exists, otherwise create it
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)

    # Case directory path
    dir_case = dir_results + '/' + case

    # Check if the case directory exists, otherwise create it
    if not os.path.exists(dir_case):
        os.mkdir(dir_case)
    return dir_case


# Class for creating log files
class Logger(object):
    '''
    Class for redirecting the sys.stdout to the log file.
    '''
    def __init__(self, dir_log):
        self.terminal = sys.stdout
        self.log = open(dir_log, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def ppo(x):
    '''
    Function to return the positive part only of x.

    Parameters
    ----------
    x : FEniCS function
        Object to retrieve the positive part.

    Returns
    -------
    ppo_val : FEniCS function
        Positive part only of x.
    '''

    ppo_val = 0.5 * (x + abs(x))
    return ppo_val


def npo(x):
    '''
    Function to return the negative part only of x.

    Parameters
    ----------
    x : FEniCS function
        Object to retrieve the negative part.

    Returns
    -------
    npo_val : FEniCS function
        Negative part only of x.
    '''

    npo_val = 0.5 * (x - abs(x))
    return npo_val