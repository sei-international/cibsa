import numpy as np
from scipy.sparse import csgraph
import re

class CIB:
    '''Objects and methods for implementing cross-impact balance analysis'''
    def __init__(self, scw_file, sl_file = None, kernel = None):
        self.structure = {}
        self.nvariants = []
        
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
        
    def accessible_scenarios(self, u):
        '''Find all indices, accessible via fluctuations defined by the threshold'''
        ib = self.impact_balance(u)
        v = self._varndx_to_vector(u)
        
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
        u = []
        for nv in x.nvariants:
            u.append(np.random.randint(nv))
        return u
    
    def signature(self, u):
        '''The signature is an integer that is guaranteed to be unique for a given scenario'''
        order = 1
        sig = 0
        for ui,nv in zip(u,self.nvariants):
            sig += order * ui
            order *= nv
        return sig
    
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
                
    def find_consistent(self, n = 10000, ignore_cycles = True):
        '''Find consistent scenarios through a Monte Carlo approach'''
        kernel = []
        signatures = set()
        for i in range(n):
            v = self.rand_scenario()
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
                
    def sim_anneal(self, u, n = 1000, ignore_cycles = True):
        '''Implement a form of simulated annealing'''
        accessible = []
        signatures = set()
        signatures.add(self.signature(u))
        uib = self.own_impact_balance(u)
        for i in range(n):
            v = self.rand_scenario()
            xib = self.cross_impact_balance(u, v)
            valid = True # They all have to satisfy the criterion, so start with true and reject if any fail the test
            for ui,xi,thr in zip(uib,xib,self.thresholds):
                if xi + thr <= ui:
                    valid = False
                    break
            if valid:
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
        
    #def old_merge(self):
    #    '''Combine scenarios that become merged under simulated annealing'''
    #    descendents = {}
    #    for u in self.kernel:
    #        accessible_sig = set()
    #        accessible = self.sim_anneal(u)
    #        for w_sig in map(self.signature, accessible):
    #            accessible_sig.add(w_sig)
    #        descendents[self.signature(u)] = accessible_sig
    #    merge = []
    #    for u_sig in descendents.keys():
    #        bag = descendents[u_sig].copy()
    #        bag.add(u_sig)
    #        for ud_set_sig in descendents.values():
    #            if u_sig in ud_set_sig:
    #                bag = bag.union(ud_set_sig)
    #        have_already = False
    #        for i in merge:
    #            if len(i.symmetric_difference(bag)) == 0:
    #                have_already = True
    #                break
    #        if not have_already:
    #            merge.append(bag)
    #    return merge
                
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
    # Select the CIM (.scw) and consistent scenario (.sl) files
    #x = CIB('CIB_global.scw', 'CIB_global.sl')
    #x = CIB('SDG-CIB-HC.scw', 'SDG-CIB-HC.sl')
    x = CIB('CIB_natl_regional.scw', 'nested_natl_regional_from_global.sl')
    # Set the threshold (the same value to be applied to each descriptor)
    threshold = 1
    
    # Note that the "signature" is an integer value guaranteed to be unique for each scenario -- it makes it easier to compare scenarios
    x.thresholds = [threshold] * x.ndesc
    for u in x.kernel:
        print x.signature(u), ":", [i + 1 for i in u]
    #print "Merged scenarios:"
    #print x.merge()
    print "Matrix of inner products:"
    for i in x.inner_product_matrix():
        print('\t'.join([str(y) for y in i]))
    #u=[0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0]
    #print x.succession_global(u)
    #print map(x.signature,x.sim_anneal(u,5000))