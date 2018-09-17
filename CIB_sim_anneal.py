import numpy as np
from scipy.sparse import csgraph
import re

class CIB:
    '''Objects and methods for implementing cross-impact balance analysis'''
    def __init__(self, scw_file, sl_file = None, kernel = None, mc_threshold = 10000, acceptall = True, paccept = 1.0):
        ''' Read a ScenarioWizard definition (.scw) file, and, alternately a solutions (.sl) file
            Parse the .scw file, stepping through the sections using a finite state machine
            If present, call a method to parse the solutions file and populate the kernel
            Populate an internal structure:
                a) Descriptors (d)
                b) Variants (states) for each descriptor (v_d)
                a) Cross-impact matrix (CIM) (nxn), n = sum(1 to d) v_d
            Note three ways to represent a scenario vector:
                signature   : An integer value guaranteed to be unique for each scenario (most compact)
                varndx      : A d-dimensional vector of indices (moderately compact)
                tablendx    : All possible descriptor/variant combination (least compact): The CIM is in this format
        '''
        self.structure = {}
        self.nvariants = []
        self.mc_threshold = mc_threshold
        
        # Initialize state machine
        s = 0
        # Initialize dimension counter
        n = 0
        # Initialize row counter
        r = 0
        # Initialize descriptor counter
        d = -1
        # Initialize variant counter
        v = 0
        # State whether to accept all values in simulated annealing
        self.acceptall = acceptall
        # Set acceptance probability (unless accepting all values)
        self.paccept = paccept
        with open(scw_file, 'r') as f:
            for line in f:
                # Skip blank lines and lines that are only whitespace
                if line.lstrip() == '':
                    continue
                if s == 0:
                    if line.lstrip()[0] == '&':
                        descr = line.lstrip()[1:].strip()
                        self.structure[descr] = []
                        if d > -1:
                            self.nvariants.append(v)
                        # Re-initialize variant counter
                        v = 0
                        d += 1
                    elif line.lstrip()[0] == '-':
                        self.structure[descr].append(line.lstrip()[1:].strip())
                        n += 1
                        v += 1
                    elif line.lstrip()[0] == '#':
                        s = 1
                        self.cim = np.empty([n,n])
                        self.nvariants.append(v)
                        self.ndim = n + 1
                        self.ndesc = d + 1
                        self.thresholds = [0] * self.ndesc
                elif (s < 5) & (line.lstrip()[0] == '#'):
                    s += 1
                elif s == 5:
                    if line.lstrip()[0] == '#':
                        s += 1
                    else:
                        self.cim[r] = map(int,line.split(','))
                        r += 1
                else:
                    continue      
        if not kernel is None:
            self.kernel = kernel
        elif not sl_file is None:
            self.kernel = self.get_kernel_from_file(sl_file)
        else:
            self.kernel = find_consistent()
        
    @property
    def paccept(self):
        '''Probability of acceptance in simulated annealing (unless acceptall = True)'''
        return self._paccept
    
    @paccept.setter
    def paccept(self, paccept):
        self._paccept = paccept
    
    @property
    def acceptall(self):
        '''Boolean: True if accept all valid results in simulated annealing'''
        return self._acceptall
    
    @acceptall.setter
    def acceptall(self, acceptall):
        self._acceptall = acceptall
    
    @property
    def mc_threshold(self):
        '''The cutoff for switching to Monte Carlo mode'''
        return self._mc_threshold
    
    @mc_threshold.setter
    def mc_threshold(self, mc_threshold):
        self._mc_threshold = mc_threshold
    
    @property
    def thresholds(self):
        '''A list of ndesc elements'''
        return self._thresholds
    
    @thresholds.setter
    def thresholds(self, thresholds):
        self._thresholds = thresholds
        
    @property
    def kernel(self):
        '''A list of lists of ndesc elements'''
        return self._kernel
    
    @kernel.setter
    def thresholds(self, kernel):
        self._kernel = kernel
    
    @property
    def structure(self):
        '''A dictionary with variants (as lists) for each descriptor (the keys)'''
        return self._structure
    
    @structure.setter
    def structure(self, structure):
        self._structure = structure
        
    @property
    def nvariants(self):
        return self._nvariants
    
    @nvariants.setter
    def nvariants(self, nvariants):
        self._nvariants = nvariants
        
    @property
    def cim(self):
        '''The cross-impact matrix, a Numpy array'''
        return self._cim
    
    @cim.setter
    def cim(self, cim):
        self._cim = cim
        
    @property
    def ndim(self):
        return self._ndim
    
    @ndim.setter
    def ndim(self, ndim):
        '''Size of the CIM'''
        self._ndim = ndim
        
    @property
    def ndesc(self):
        return self._ndesc
    
    @ndesc.setter
    def ndesc(self, ndesc):
        self._ndesc = ndesc
        
    def _varndx_to_tablendx(self, u):
        '''Convert a list of ndesc variants, expressed as indices, into a list of indexes on the CIM'''
        u_ndx = list(u)
        ndx = 0
        for i in range(len(u)):
            u_ndx[i] = ndx + u[i]
            ndx += self.nvariants[i]
        return u_ndx
    
    def _varndx_to_vector(self, u):
        '''Convert list of ndesc variants, expressed as indices, to a boolean Numpy array of length ndim'''
        u_vect = np.full(self.ndim, False, dtype = bool)
        u_vect[self._varndx_to_tablendx(u)] = True
        return u_vect
    
    def impact_balance(self, u):
        '''u is a list with ndesc elements; returns a list with ndim elements'''
        return np.sum(self.cim[self._varndx_to_tablendx(u)], axis=0)
        
    def own_impact_balance(self, u):
        '''The entries for the impact balance corresponding to the scenario itself'''
        return self.impact_balance(u)[self._varndx_to_tablendx(u)]
        
    def cross_impact_balance(self, u, v):
        '''The entries for the impact balance corresponding to another scenario'''
        return self.impact_balance(u)[self._varndx_to_tablendx(v)]
        
    def inner_product(self, u, v):
        '''u and v are lists with ndesc elements; returns a scalar'''
        ib = self.impact_balance(u)
        ib_sel = ib[self._varndx_to_tablendx(v)]
        return(sum(ib_sel))
        
    def succession_global_1(self, u):
        '''Do a one-step succession using global algorithm'''
        ib = self.impact_balance(u)
        v = list(u)
        start = 0
        for i in range(self.ndesc):
            stop = start + self.nvariants[i]
            ib_desc = ib[start:stop]
            max_var_desc = ib_desc[u[i]]
            for j in range(len(ib_desc)):
                if ib_desc[j] > max_var_desc: # In case of a tie, this always goes to the first entry
                    max_var_desc = ib_desc[j]
                    v[i] = j
            start = stop
        return v
    
    def rand_scenario(self):
        '''Generate a pseudo-random scenario'''
        u = []
        for nv in x.nvariants:
            u.append(np.random.randint(nv))
        return u
    
    def get_scenario_signatures(self, max = None, allow_dups = False):
        '''Generate a set of scenario signatures: either all or a random selection, no more than max'''
        n = self.max_signature()
        all = range(n)
        if not max is None and n > max:
            return(np.random.choice(all, max, replace = not allow_dups))
        else:
            return(all)
    
    def signature(self, u):
        '''The signature is an integer that is guaranteed to be unique for a given scenario'''
        order = 1
        sig = 0
        for ui,nv in zip(u,self.nvariants):
            sig += order * ui
            order *= nv
        return sig
    
    def inv_signature(self, s):
        u = []
        order = 1
        for nv in self.nvariants:
            order *= nv
            u.append(s % nv)
            s //= nv
        return(u)
    
    def max_signature(self):
        '''The maximum value for the signature'''
        return(self.signature(self.nvariants))
    
    def succession_global(self, u):
        '''Follow a succession all the way to the end, with the possibility of a cycle (stop if there is a repeat)'''
        iterations = []
        iterations_sig = []
        iterations.append(u)
        iterations_sig.append(self.signature(u))
        foundit = False
        v = list(u) # Initialize
        while True:
            v = self.succession_global_1(v)
            n = 1 # Count length of cycle, if any
            v_sig = self.signature(v)
            for hist in iterations_sig[::-1]:
                if hist == v_sig:
                    foundit = True
                    break
                else:
                    n += 1
            if foundit:
                break
            else:
                iterations.append(v)
                iterations_sig.append(v_sig)
        return [n,v]
                
    def find_consistent(self, ignore_cycles = True):
        ''' Find consistent scenarios starting from all scenarios or a random selection of n scenarios, whichever is the smaller set
            Set max = None to ensure all scenarios are sampled
            Note that this is much slower than ScenarioWizard: better to use "get_kernel_from_file"'''
        kernel = []
        signatures = set()
        for v_sig in self.get_scenario_signatures(self.mc_threshold):
            v = self.inv_signature(v_sig)
            nper,veqm = self.succession_global(v)
            if ignore_cycles & (nper > 1):
                continue
            veqm_sig = self.signature(veqm)
            if not veqm_sig in signatures:
                signatures.add(veqm_sig)
                kernel.append(veqm)
        return kernel
    
    def get_kernel_from_file(self, sl_file):
        '''Load consistent scenarios from a ScenarioWizard .sl file'''
        kernel = []
        indices_re = re.compile(r'^"([^"]+)"')
        with open(sl_file, 'r') as f:
            for line in f:
                # Skip blank lines and lines that are only whitespace
                if line.lstrip() == '':
                    continue
                # Skip lines that don't start with a quote mark
                if line.lstrip()[0] != '"':
                    continue
                # Make zero-based
                kernel.append([x - y for x, y in zip(map(int, indices_re.match(line.lstrip()).group(1).split()), [1] * self.ndesc)])
        return kernel
                
    def sim_anneal(self, u, ignore_cycles = True):
        '''Implement a form of simulated annealing'''
        accessible = []
        signatures = set()
        signatures.add(self.signature(u))
        uib = self.own_impact_balance(u)
        for v_sig in self.get_scenario_signatures(self.mc_threshold):
            v = self.inv_signature(v_sig)
            xib = self.cross_impact_balance(u, v)
            valid = True # They all have to satisfy the criterion, so start with true and reject if any fail the test
            for ui,xi,thr in zip(uib,xib,self.thresholds):
                if xi + thr <= ui:
                    valid = False
                    break
            if valid and (self.acceptall or (np.random.random_sample() < self.paccept)):
                nper,veqm = self.succession_global(v)
                if ignore_cycles & (nper > 1):
                    continue
                veqm_sig = self.signature(veqm)
                if not veqm_sig in signatures:
                    signatures.add(veqm_sig)
                    accessible.append(veqm)
        return accessible
    
    def inner_product_matrix(self):
            M = []
            for u in self.kernel:
                r = []
                for v in self.kernel:
                    r.append(self.inner_product(u,v))
                M.append(r)
            return M
                
    def graph(self):
        '''Map elements of the kernel related to one another through fluctuations as a graph'''
        k_size = len(self.kernel)
        kernel_sig = map(self.signature, self.kernel)
        adj = np.zeros([k_size,k_size])
        for r,u in enumerate(self.kernel):
            accessible = self.sim_anneal(u)
            for w_sig in map(self.signature, accessible):
                adj[r,kernel_sig.index(w_sig)] = 1
        return csgraph.csgraph_from_dense(adj)
    
    def merge(self):
        '''Report connected components of the fluctuation-connected graph'''
        merge = []
        ncomp, labels = csgraph.connected_components(self.graph())
        for i in range(ncomp):
            merge.append([self.signature(self.kernel[k]) for k in [j for j,z in enumerate(labels) if z == i]])
        return merge
        
if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Select the CIM (.scw) and consistent scenario (.sl) files
    # -----------------------------------------------------------------------
    x = CIB('sample_files/CIB_global.scw', 'sample_files/CIB_global.sl')
    
    # -----------------------------------------------------------------------
    # Print each consistent scenario with its "signature" (a unique integer identifier)
    # -----------------------------------------------------------------------
    print "Scenario kernel:"
    for u in x.kernel:
        print x.signature(u), ":", [i + 1 for i in u]

    # -----------------------------------------------------------------------
    # Find scenarios that become merged under simulated annealing
    # -----------------------------------------------------------------------
    # First, set the threshold (the same value to be applied to each descriptor)
    x.thresholds = [3] * x.ndesc
    x.acceptall = True
    x.paccept = 0.1
    print "Merged scenarios:"
    print x.merge()
    
    # -----------------------------------------------------------------------
    # Report the matrix of inner products of the kernel
    # -----------------------------------------------------------------------
    print "Matrix of inner products:"
    for i in x.inner_product_matrix():
        print('\t'.join([str(y) for y in i]))
