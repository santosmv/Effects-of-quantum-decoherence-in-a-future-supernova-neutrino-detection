import numpy as np
import math
from numpy import array, sqrt, exp, trace
import numpy as np
import numpy.linalg as la
from numba import jit, njit, vectorize, float64
import numba as nb
from time import time
from scipy.linalg import expm
from numba.pycc import CC
#uncomment next line to save compiled compatible to OS
# cc = CC('Pij_sn')

#uncomment next line to save compiled compatible to OS
# cc.verbose = True

ti = time()

# data = np.loadtxt('data/hydro-LS220-s20.0-MUONS-T=0.025017259.txt', delimiter=',', skiprows=5)

file1 = 'data/hydro-Huedepohl-1D-Accretion-ls180-s20.0-profiles-0.019s.txt'
file2 = 'data/hydro-Huedepohl-1D-Accretion-ls180-s40.0-profiles-0.027s.txt'
data = np.loadtxt(file1, delimiter=',', skiprows=4)
# data = np.loadtxt(file2, delimiter=',', skiprows=4)
c = 2.99792458e8 #m/s
Gf = 1.16632e-23 #eV^-2
g_to_eV = c**2/1000 * 6.242e18
cm3_to_eV3 = (0.197e9*1e-15*100)**3
r_data = data[:,0]/100/1000 #km
rho = data[:,2]
Ye = data[:,6]
rho_nat = rho * g_to_eV * cm3_to_eV3 #eV^4
mn = (939.5656305188e6 + 938.2723404280e6)/2 #eV
# ne = Ye*rho_nat/mn

#initial radius of the SN
r0 = min(r_data) #km
#final radius to calculate Pee
rf = max(r_data) #km
#list of all points along SN radius to calculate Pee
r_list_pijsn = np.geomspace(r0, rf, 1001) #km
# r_list_pijsn = np.geomspace(r0, 5e9, 1001) #km

i = 0.+1.j
a = 0.+0.j

l1 = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]], dtype=np.complex128)
l2 = np.array([[0.,-1j,0.],[1j,0.,0.],[0.,0.,0.]], dtype=np.complex128)
l3 = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,0.]], dtype=np.complex128)
l4 = np.array([[0.,0.,1.],[0.,0.,0.],[1.,0.,0.]], dtype=np.complex128)
l5 = np.array([[0.,0.,-1j],[0.,0.,0.],[1j,0.,0.]], dtype=np.complex128)
l6 = np.array([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.]], dtype=np.complex128)
l7 = np.array([[0.,0.,0.],[0.,0.,-1j],[0.,1j,0.]], dtype=np.complex128)
l8 = 1./math.sqrt(3)*np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,-2]], dtype=np.complex128)

# @jit(nopython=True, cache=True)
#uncomment next line to save compiled compatible to OS
# @cc.export('Pij_SN_surface', 'f8(f8, f8, string, string, string)')
def Pij_SN_surface(ne, Gamma3, Gamma8, mix, ij, nu_nubar):
    E = 10
    Deltam21 = 7.42e-5 #eV² +0.21 -0.20
    Deltam31 = 2.510e-3 #NH #eV² +-0.027 https://arxiv.org/pdf/2111.03086.pdf
    theta12 = 33.45 #+0.77 -0.75 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
    theta13 = 8.62  #+-0.12 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
    theta23 = 49.2 #+1.0 -1.3 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
    theta12 = theta12*math.pi/180
    theta13 = theta13*math.pi/180
    theta23 = theta23*math.pi/180

    c12=np.cos(theta12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    c23=np.cos(theta23)
    s23=np.sin(theta23)
    U23 = np.array([ [1.,0.,0.],[0.,c23,s23],[0,-s23,c23] ], dtype=np.complex128)
    U13 = np.array([ [c13,0.,s13],[0.,1.,0.],[-s13,0,c13] ], dtype=np.complex128)
    U12 = np.array([ [c12,s12,0.],[-s12,c12,0],[0.,0.,1.] ], dtype=np.complex128)
    U3 = U23 @ U13 @ U12
    U3_dag = np.transpose(U3)

    def pade_approximation(a):
        n = a.shape[0]
        q = 6
        a2 = a.copy ( )
        a_norm = np.linalg.norm ( a2, np.inf )
        ee = ( int ) ( np.log2 ( a_norm ) ) + 1
        s = max ( 0, ee + 1 )
        a2 = a2 / ( 2.0 ** s )
        x = a2.copy ( )
        c = 0.5
        e = np.eye ( n, dtype = np.complex64 ) + c * a2
        d = np.eye ( n, dtype = np.complex64 ) - c * a2
        p = True
        for k in range ( 2, q + 1 ):
            c = c * float ( q - k + 1 ) / float ( k * ( 2 * q - k + 1 ) )
            x = np.dot ( a2, x )
            e = e + c * x
            if ( p ):
                d = d + c * x
            else:
                d = d - c * x
            p = not p
        #  E -> inverse(D) * E
        e = np.linalg.solve ( d, e )
        #  E -> E^(2*S)
        for k in range ( 0, s ):
            e = np.dot ( e, e )
        return e

    #Energy in MeV and rho in g cm^-3
    def lindbladian(E, vcc0):
        E = E*1e6

        #Re-evaluating the M2 with the E parameter for the function
        if mix == 'NH':
            M2 = np.array([[0.,0.,0.],[0.,Deltam21,0.],[0.,0.,Deltam31]], dtype=np.complex128)
        elif mix == 'IH':
            M2 = np.array([[0.,0.,0.],[0.,Deltam21,0.],[0.,0.,-Deltam31]], dtype=np.complex128)
        
        if nu_nubar == 'nue':
            vcc0 = vcc0
        elif nu_nubar == 'nuebar':
            vcc0 = -vcc0

        #Hamiltonian in flavour basis
        M2f = U3 @ M2 @ U3_dag
        
        #Potential in matter
        Vcc = np.array([[vcc0,0.,0.],[0.,0.,0.],[0.,0.,0.]], dtype=np.complex128)
        Hf = 1/(2*E) * M2f + Vcc

        #eigenvalues
        eigvals, eigvecs = la.eig(Hf)
        eigvals = eigvals.real
        #sorting eigenvalues list
        id_sor = np.argsort(np.abs(eigvals))
        #adding eigenvalues to a list
        eval1 = eigvals[id_sor[0]]
        eval2 = eigvals[id_sor[1]]
        eval3 = eigvals[id_sor[2]]
        #collecting eigenvectors from sorted eigenvalues
        eve1 = eigvecs[:,id_sor[0]]
        eve2 = eigvecs[:,id_sor[1]]
        eve3 = eigvecs[:,id_sor[2]]
        #Eigenvector for electron neutrino spectrum
        Ue1 = (eve1[0])
        Ue2 = (eve2[0])
        Ue3 = (eve3[0])
        #Eigenvector for muon neutrino spectrum
        Umu1 = (eve1[1])
        Umu2 = (eve2[1])
        Umu3 = (eve3[1])
        #Eigenvector for tau neutrino spectrum
        Utau1 = (eve1[2])
        Utau2 = (eve2[2])
        Utau3 = (eve3[2])
        Um = array([[Ue1,Ue2,Ue3],
                    [Umu1,Umu2,Umu3],
                    [Utau1,Utau2,Utau3]], dtype=np.complex128)
        Um_dag = np.transpose(np.conjugate(Um))
        Hm =  np.array([[eval1, 0., 0.],
                        [0., eval2, 0.],
                        [0., 0., eval3]], dtype=np.complex128)
        h1 = 1./2.*trace(Hm@l1)
        h2 = 1./2.*trace(Hm@l2)
        h3 = 1./2.*trace(Hm@l3)
        h4 = 1./2.*trace(Hm@l4)
        h5 = 1./2.*trace(Hm@l5)
        h6 = 1./2.*trace(Hm@l6)
        h7 = 1./2.*trace(Hm@l7)
        h8 = 1./2.*trace(Hm@l8)
        
        Hlm =   -2 * array([[a, a, a, a, a, a, a, a, a],
                            [a, a, h3, -h2, h7/2., -h6/2., h5/2., -h4/2., a], 
                            [a, -h3, a, h1, h6/2., h7/2., -h4/2., -h5/2., a], 
                            [a, h2, -h1, a, h5/2., -h4/2., -h7/2., h6/2., a], 
                            [a, -h7/2., -h6/2., -h5/2., a, h3/2. + sqrt(3.)*h8/2., h2/2., h1/2., -sqrt(3.)*h5/2.],
                            [a, h6/2., -h7/2., h4/2., -h3/2. - sqrt(3.)*h8/2., a, -h1/2., h2/2., sqrt(3.)*h4/2.],
                            [a, -h5/2., h4/2., h7/2., -h2/2., h1/2., a, -h3/2. + sqrt(3.)*h8/2., -sqrt(3.)*h7/2.],
                            [a, h4/2., h5/2., -h6/2., -h1/2., -h2/2., h3/2. - sqrt(3.)*h8/2., a, sqrt(3.)*h6/2.],
                            [a, a, a, a, sqrt(3.)*h5/2., -sqrt(3.)*h4/2., sqrt(3.)*h7/2., -sqrt(3.)*h6/2., a]])

        # Gamma1 = Gamma2 = Gamma4 = Gamma5 = Gamma6 = Gamma7 = 0
        Gamma1 = Gamma2 = Gamma4 = Gamma5 = Gamma6 = Gamma7 = Gamma8
        # Gamma3 = 1e-35 #eV
        # Gamma8 = 1e-32 #eV
        gamma1 = gamma2 = gamma3 = Gamma8
        A = Um_dag @ U3
        a11 = A[0,0]
        a12 = A[0,1]
        a13 = A[0,2]
        a21 = A[1,0]
        a22 = A[1,1]
        a23 = A[1,2]
        a31 = A[2,0]
        a32 = A[2,1]
        a33 = A[2,2]

        a_vec = a11,a12,a13,a21,a22,a23,a31,a32,a33
        Gamma_vec = Gamma1,Gamma2,Gamma3,Gamma4,Gamma5,Gamma6,Gamma7,Gamma8,gamma1,gamma2,gamma3

        def Dm_function(a_vec, Gamma_vec):
            a11,a12,a13,a21,a22,a23,a31,a32,a33 = a_vec
            # From Pedro's picture (decoherence parameters were ignored):
            Gamma1,Gamma2,Gamma3,Gamma4,Gamma5,Gamma6,Gamma7,Gamma8,gamma1,gamma2,gamma3 = Gamma_vec
            
            Gamma1 = Gamma3 - Gamma2
            Gamma7 = 2*Gamma8 - Gamma4 - Gamma5 - Gamma6
            D11 = Gamma1 + Gamma8/3 + gamma1
            D22 = Gamma2 + Gamma8/3 + gamma1
            D33 = Gamma3 + Gamma8/3
            D44 = Gamma3/4 + Gamma4 + gamma2
            D55 = Gamma3/4 + Gamma5 + gamma2
            D66 = Gamma3/4 + Gamma6 + gamma3
            D77 = Gamma3/4 + Gamma7 + gamma3
            D88 = Gamma8

            Dm = array([[a, a, a, a, a, a, a, a, a], 
                        [a, D11*a11**2*a22**2 + 2*D11*a11*a12*a21*a22 + D11*a12**2*a21**2 + D33*a11**2*a21**2 - 2*D33*a11*a12*a21*a22 + D33*a12**2*a22**2 + D44*a11**2*a23**2 + 2*D44*a11*a13*a21*a23 + D44*a13**2*a21**2 + D66*a12**2*a23**2 + 2*D66*a12*a13*a22*a23 + D66*a13**2*a22**2 + D88*a11**2*a21**2/3 + 2*D88*a11*a12*a21*a22/3 - 4*D88*a11*a13*a21*a23/3 + D88*a12**2*a22**2/3 - 4*D88*a12*a13*a22*a23/3 + 4*D88*a13**2*a23**2/3, a, D11*a11**2*a12*a22 + D11*a11*a12**2*a21 - D11*a11*a21*a22**2 - D11*a12*a21**2*a22 + D33*a11**3*a21/2 - D33*a11**2*a12*a22/2 - D33*a11*a12**2*a21/2 - D33*a11*a21**3/2 + D33*a11*a21*a22**2/2 + D33*a12**3*a22/2 + D33*a12*a21**2*a22/2 - D33*a12*a22**3/2 + D44*a11**2*a13*a23 + D44*a11*a13**2*a21 - D44*a11*a21*a23**2 - D44*a13*a21**2*a23 + D66*a12**2*a13*a23 + D66*a12*a13**2*a22 - D66*a12*a22*a23**2 - D66*a13*a22**2*a23 + D88*a11**3*a21/6 + D88*a11**2*a12*a22/6 - D88*a11**2*a13*a23/3 + D88*a11*a12**2*a21/6 - D88*a11*a13**2*a21/3 - D88*a11*a21**3/6 - D88*a11*a21*a22**2/6 + D88*a11*a21*a23**2/3 + D88*a12**3*a22/6 - D88*a12**2*a13*a23/3 - D88*a12*a13**2*a22/3 - D88*a12*a21**2*a22/6 - D88*a12*a22**3/6 + D88*a12*a22*a23**2/3 + 2*D88*a13**3*a23/3 + D88*a13*a21**2*a23/3 + D88*a13*a22**2*a23/3 - 2*D88*a13*a23**3/3, D11*a11**2*a22*a32 + D11*a11*a12*a21*a32 + D11*a11*a12*a22*a31 + D11*a12**2*a21*a31 + D33*a11**2*a21*a31 - D33*a11*a12*a21*a32 - D33*a11*a12*a22*a31 + D33*a12**2*a22*a32 + D44*a11**2*a23*a33 + D44*a11*a13*a21*a33 + D44*a11*a13*a23*a31 + D44*a13**2*a21*a31 + D66*a12**2*a23*a33 + D66*a12*a13*a22*a33 + D66*a12*a13*a23*a32 + D66*a13**2*a22*a32 + D88*a11**2*a21*a31/3 + D88*a11*a12*a21*a32/3 + D88*a11*a12*a22*a31/3 - 2*D88*a11*a13*a21*a33/3 - 2*D88*a11*a13*a23*a31/3 + D88*a12**2*a22*a32/3 - 2*D88*a12*a13*a22*a33/3 - 2*D88*a12*a13*a23*a32/3 + 4*D88*a13**2*a23*a33/3, a, D11*a11*a21*a22*a32 + D11*a11*a22**2*a31 + D11*a12*a21**2*a32 + D11*a12*a21*a22*a31 + D33*a11*a21**2*a31 - D33*a11*a21*a22*a32 - D33*a12*a21*a22*a31 + D33*a12*a22**2*a32 + D44*a11*a21*a23*a33 + D44*a11*a23**2*a31 + D44*a13*a21**2*a33 + D44*a13*a21*a23*a31 + D66*a12*a22*a23*a33 + D66*a12*a23**2*a32 + D66*a13*a22**2*a33 + D66*a13*a22*a23*a32 + D88*a11*a21**2*a31/3 + D88*a11*a21*a22*a32/3 - 2*D88*a11*a21*a23*a33/3 + D88*a12*a21*a22*a31/3 + D88*a12*a22**2*a32/3 - 2*D88*a12*a22*a23*a33/3 - 2*D88*a13*a21*a23*a31/3 - 2*D88*a13*a22*a23*a32/3 + 4*D88*a13*a23**2*a33/3, a, sqrt(3)*D11*a11**2*a12*a22/3 + sqrt(3)*D11*a11*a12**2*a21/3 + sqrt(3)*D11*a11*a21*a22**2/3 - 2*sqrt(3)*D11*a11*a22*a31*a32/3 + sqrt(3)*D11*a12*a21**2*a22/3 - 2*sqrt(3)*D11*a12*a21*a31*a32/3 + sqrt(3)*D33*a11**3*a21/6 - sqrt(3)*D33*a11**2*a12*a22/6 - sqrt(3)*D33*a11*a12**2*a21/6 + sqrt(3)*D33*a11*a21**3/6 - sqrt(3)*D33*a11*a21*a22**2/6 - sqrt(3)*D33*a11*a21*a31**2/3 + sqrt(3)*D33*a11*a21*a32**2/3 + sqrt(3)*D33*a12**3*a22/6 - sqrt(3)*D33*a12*a21**2*a22/6 + sqrt(3)*D33*a12*a22**3/6 + sqrt(3)*D33*a12*a22*a31**2/3 - sqrt(3)*D33*a12*a22*a32**2/3 + sqrt(3)*D44*a11**2*a13*a23/3 + sqrt(3)*D44*a11*a13**2*a21/3 + sqrt(3)*D44*a11*a21*a23**2/3 - 2*sqrt(3)*D44*a11*a23*a31*a33/3 + sqrt(3)*D44*a13*a21**2*a23/3 - 2*sqrt(3)*D44*a13*a21*a31*a33/3 + sqrt(3)*D66*a12**2*a13*a23/3 + sqrt(3)*D66*a12*a13**2*a22/3 + sqrt(3)*D66*a12*a22*a23**2/3 - 2*sqrt(3)*D66*a12*a23*a32*a33/3 + sqrt(3)*D66*a13*a22**2*a23/3 - 2*sqrt(3)*D66*a13*a22*a32*a33/3 + sqrt(3)*D88*a11**3*a21/18 + sqrt(3)*D88*a11**2*a12*a22/18 - sqrt(3)*D88*a11**2*a13*a23/9 + sqrt(3)*D88*a11*a12**2*a21/18 - sqrt(3)*D88*a11*a13**2*a21/9 + sqrt(3)*D88*a11*a21**3/18 + sqrt(3)*D88*a11*a21*a22**2/18 - sqrt(3)*D88*a11*a21*a23**2/9 - sqrt(3)*D88*a11*a21*a31**2/9 - sqrt(3)*D88*a11*a21*a32**2/9 + 2*sqrt(3)*D88*a11*a21*a33**2/9 + sqrt(3)*D88*a12**3*a22/18 - sqrt(3)*D88*a12**2*a13*a23/9 - sqrt(3)*D88*a12*a13**2*a22/9 + sqrt(3)*D88*a12*a21**2*a22/18 + sqrt(3)*D88*a12*a22**3/18 - sqrt(3)*D88*a12*a22*a23**2/9 - sqrt(3)*D88*a12*a22*a31**2/9 - sqrt(3)*D88*a12*a22*a32**2/9 + 2*sqrt(3)*D88*a12*a22*a33**2/9 + 2*sqrt(3)*D88*a13**3*a23/9 - sqrt(3)*D88*a13*a21**2*a23/9 - sqrt(3)*D88*a13*a22**2*a23/9 + 2*sqrt(3)*D88*a13*a23**3/9 + 2*sqrt(3)*D88*a13*a23*a31**2/9 + 2*sqrt(3)*D88*a13*a23*a32**2/9 - 4*sqrt(3)*D88*a13*a23*a33**2/9], 
                        [a, a, D22*a11**2*a22**2 - 2*D22*a11*a12*a21*a22 + D22*a12**2*a21**2 + D55*a11**2*a23**2 - 2*D55*a11*a13*a21*a23 + D55*a13**2*a21**2 + D77*a12**2*a23**2 - 2*D77*a12*a13*a22*a23 + D77*a13**2*a22**2, a, a, D22*a11**2*a22*a32 - D22*a11*a12*a21*a32 - D22*a11*a12*a22*a31 + D22*a12**2*a21*a31 + D55*a11**2*a23*a33 - D55*a11*a13*a21*a33 - D55*a11*a13*a23*a31 + D55*a13**2*a21*a31 + D77*a12**2*a23*a33 - D77*a12*a13*a22*a33 - D77*a12*a13*a23*a32 + D77*a13**2*a22*a32, a, D22*a11*a21*a22*a32 - D22*a11*a22**2*a31 - D22*a12*a21**2*a32 + D22*a12*a21*a22*a31 + D55*a11*a21*a23*a33 - D55*a11*a23**2*a31 - D55*a13*a21**2*a33 + D55*a13*a21*a23*a31 + D77*a12*a22*a23*a33 - D77*a12*a23**2*a32 - D77*a13*a22**2*a33 + D77*a13*a22*a23*a32, a], 
                        [a, D11*a11**2*a12*a22 + D11*a11*a12**2*a21 - D11*a11*a21*a22**2 - D11*a12*a21**2*a22 + D33*a11**3*a21/2 - D33*a11**2*a12*a22/2 - D33*a11*a12**2*a21/2 - D33*a11*a21**3/2 + D33*a11*a21*a22**2/2 + D33*a12**3*a22/2 + D33*a12*a21**2*a22/2 - D33*a12*a22**3/2 + D44*a11**2*a13*a23 + D44*a11*a13**2*a21 - D44*a11*a21*a23**2 - D44*a13*a21**2*a23 + D66*a12**2*a13*a23 + D66*a12*a13**2*a22 - D66*a12*a22*a23**2 - D66*a13*a22**2*a23 + D88*a11**3*a21/6 + D88*a11**2*a12*a22/6 - D88*a11**2*a13*a23/3 + D88*a11*a12**2*a21/6 - D88*a11*a13**2*a21/3 - D88*a11*a21**3/6 - D88*a11*a21*a22**2/6 + D88*a11*a21*a23**2/3 + D88*a12**3*a22/6 - D88*a12**2*a13*a23/3 - D88*a12*a13**2*a22/3 - D88*a12*a21**2*a22/6 - D88*a12*a22**3/6 + D88*a12*a22*a23**2/3 + 2*D88*a13**3*a23/3 + D88*a13*a21**2*a23/3 + D88*a13*a22**2*a23/3 - 2*D88*a13*a23**3/3, a, D11*a11**2*a12**2 - 2*D11*a11*a12*a21*a22 + D11*a21**2*a22**2 + D33*a11**4/4 - D33*a11**2*a12**2/2 - D33*a11**2*a21**2/2 + D33*a11**2*a22**2/2 + D33*a12**4/4 + D33*a12**2*a21**2/2 - D33*a12**2*a22**2/2 + D33*a21**4/4 - D33*a21**2*a22**2/2 + D33*a22**4/4 + D44*a11**2*a13**2 - 2*D44*a11*a13*a21*a23 + D44*a21**2*a23**2 + D66*a12**2*a13**2 - 2*D66*a12*a13*a22*a23 + D66*a22**2*a23**2 + D88*a11**4/12 + D88*a11**2*a12**2/6 - D88*a11**2*a13**2/3 - D88*a11**2*a21**2/6 - D88*a11**2*a22**2/6 + D88*a11**2*a23**2/3 + D88*a12**4/12 - D88*a12**2*a13**2/3 - D88*a12**2*a21**2/6 - D88*a12**2*a22**2/6 + D88*a12**2*a23**2/3 + D88*a13**4/3 + D88*a13**2*a21**2/3 + D88*a13**2*a22**2/3 - 2*D88*a13**2*a23**2/3 + D88*a21**4/12 + D88*a21**2*a22**2/6 - D88*a21**2*a23**2/3 + D88*a22**4/12 - D88*a22**2*a23**2/3 + D88*a23**4/3, D11*a11**2*a12*a32 + D11*a11*a12**2*a31 - D11*a11*a21*a22*a32 - D11*a12*a21*a22*a31 + D33*a11**3*a31/2 - D33*a11**2*a12*a32/2 - D33*a11*a12**2*a31/2 - D33*a11*a21**2*a31/2 + D33*a11*a22**2*a31/2 + D33*a12**3*a32/2 + D33*a12*a21**2*a32/2 - D33*a12*a22**2*a32/2 + D44*a11**2*a13*a33 + D44*a11*a13**2*a31 - D44*a11*a21*a23*a33 - D44*a13*a21*a23*a31 + D66*a12**2*a13*a33 + D66*a12*a13**2*a32 - D66*a12*a22*a23*a33 - D66*a13*a22*a23*a32 + D88*a11**3*a31/6 + D88*a11**2*a12*a32/6 - D88*a11**2*a13*a33/3 + D88*a11*a12**2*a31/6 - D88*a11*a13**2*a31/3 - D88*a11*a21**2*a31/6 - D88*a11*a22**2*a31/6 + D88*a11*a23**2*a31/3 + D88*a12**3*a32/6 - D88*a12**2*a13*a33/3 - D88*a12*a13**2*a32/3 - D88*a12*a21**2*a32/6 - D88*a12*a22**2*a32/6 + D88*a12*a23**2*a32/3 + 2*D88*a13**3*a33/3 + D88*a13*a21**2*a33/3 + D88*a13*a22**2*a33/3 - 2*D88*a13*a23**2*a33/3, a, D11*a11*a12*a21*a32 + D11*a11*a12*a22*a31 - D11*a21**2*a22*a32 - D11*a21*a22**2*a31 + D33*a11**2*a21*a31/2 - D33*a11**2*a22*a32/2 - D33*a12**2*a21*a31/2 + D33*a12**2*a22*a32/2 - D33*a21**3*a31/2 + D33*a21**2*a22*a32/2 + D33*a21*a22**2*a31/2 - D33*a22**3*a32/2 + D44*a11*a13*a21*a33 + D44*a11*a13*a23*a31 - D44*a21**2*a23*a33 - D44*a21*a23**2*a31 + D66*a12*a13*a22*a33 + D66*a12*a13*a23*a32 - D66*a22**2*a23*a33 - D66*a22*a23**2*a32 + D88*a11**2*a21*a31/6 + D88*a11**2*a22*a32/6 - D88*a11**2*a23*a33/3 + D88*a12**2*a21*a31/6 + D88*a12**2*a22*a32/6 - D88*a12**2*a23*a33/3 - D88*a13**2*a21*a31/3 - D88*a13**2*a22*a32/3 + 2*D88*a13**2*a23*a33/3 - D88*a21**3*a31/6 - D88*a21**2*a22*a32/6 + D88*a21**2*a23*a33/3 - D88*a21*a22**2*a31/6 + D88*a21*a23**2*a31/3 - D88*a22**3*a32/6 + D88*a22**2*a23*a33/3 + D88*a22*a23**2*a32/3 - 2*D88*a23**3*a33/3, a, sqrt(3)*D11*a11**2*a12**2/3 - 2*sqrt(3)*D11*a11*a12*a31*a32/3 - sqrt(3)*D11*a21**2*a22**2/3 + 2*sqrt(3)*D11*a21*a22*a31*a32/3 + sqrt(3)*D33*a11**4/12 - sqrt(3)*D33*a11**2*a12**2/6 - sqrt(3)*D33*a11**2*a31**2/6 + sqrt(3)*D33*a11**2*a32**2/6 + sqrt(3)*D33*a12**4/12 + sqrt(3)*D33*a12**2*a31**2/6 - sqrt(3)*D33*a12**2*a32**2/6 - sqrt(3)*D33*a21**4/12 + sqrt(3)*D33*a21**2*a22**2/6 + sqrt(3)*D33*a21**2*a31**2/6 - sqrt(3)*D33*a21**2*a32**2/6 - sqrt(3)*D33*a22**4/12 - sqrt(3)*D33*a22**2*a31**2/6 + sqrt(3)*D33*a22**2*a32**2/6 + sqrt(3)*D44*a11**2*a13**2/3 - 2*sqrt(3)*D44*a11*a13*a31*a33/3 - sqrt(3)*D44*a21**2*a23**2/3 + 2*sqrt(3)*D44*a21*a23*a31*a33/3 + sqrt(3)*D66*a12**2*a13**2/3 - 2*sqrt(3)*D66*a12*a13*a32*a33/3 - sqrt(3)*D66*a22**2*a23**2/3 + 2*sqrt(3)*D66*a22*a23*a32*a33/3 + sqrt(3)*D88*a11**4/36 + sqrt(3)*D88*a11**2*a12**2/18 - sqrt(3)*D88*a11**2*a13**2/9 - sqrt(3)*D88*a11**2*a31**2/18 - sqrt(3)*D88*a11**2*a32**2/18 + sqrt(3)*D88*a11**2*a33**2/9 + sqrt(3)*D88*a12**4/36 - sqrt(3)*D88*a12**2*a13**2/9 - sqrt(3)*D88*a12**2*a31**2/18 - sqrt(3)*D88*a12**2*a32**2/18 + sqrt(3)*D88*a12**2*a33**2/9 + sqrt(3)*D88*a13**4/9 + sqrt(3)*D88*a13**2*a31**2/9 + sqrt(3)*D88*a13**2*a32**2/9 - 2*sqrt(3)*D88*a13**2*a33**2/9 - sqrt(3)*D88*a21**4/36 - sqrt(3)*D88*a21**2*a22**2/18 + sqrt(3)*D88*a21**2*a23**2/9 + sqrt(3)*D88*a21**2*a31**2/18 + sqrt(3)*D88*a21**2*a32**2/18 - sqrt(3)*D88*a21**2*a33**2/9 - sqrt(3)*D88*a22**4/36 + sqrt(3)*D88*a22**2*a23**2/9 + sqrt(3)*D88*a22**2*a31**2/18 + sqrt(3)*D88*a22**2*a32**2/18 - sqrt(3)*D88*a22**2*a33**2/9 - sqrt(3)*D88*a23**4/9 - sqrt(3)*D88*a23**2*a31**2/9 - sqrt(3)*D88*a23**2*a32**2/9 + 2*sqrt(3)*D88*a23**2*a33**2/9], 
                        [a, D11*a11**2*a22*a32 + D11*a11*a12*a21*a32 + D11*a11*a12*a22*a31 + D11*a12**2*a21*a31 + D33*a11**2*a21*a31 - D33*a11*a12*a21*a32 - D33*a11*a12*a22*a31 + D33*a12**2*a22*a32 + D44*a11**2*a23*a33 + D44*a11*a13*a21*a33 + D44*a11*a13*a23*a31 + D44*a13**2*a21*a31 + D66*a12**2*a23*a33 + D66*a12*a13*a22*a33 + D66*a12*a13*a23*a32 + D66*a13**2*a22*a32 + D88*a11**2*a21*a31/3 + D88*a11*a12*a21*a32/3 + D88*a11*a12*a22*a31/3 - 2*D88*a11*a13*a21*a33/3 - 2*D88*a11*a13*a23*a31/3 + D88*a12**2*a22*a32/3 - 2*D88*a12*a13*a22*a33/3 - 2*D88*a12*a13*a23*a32/3 + 4*D88*a13**2*a23*a33/3, a, D11*a11**2*a12*a32 + D11*a11*a12**2*a31 - D11*a11*a21*a22*a32 - D11*a12*a21*a22*a31 + D33*a11**3*a31/2 - D33*a11**2*a12*a32/2 - D33*a11*a12**2*a31/2 - D33*a11*a21**2*a31/2 + D33*a11*a22**2*a31/2 + D33*a12**3*a32/2 + D33*a12*a21**2*a32/2 - D33*a12*a22**2*a32/2 + D44*a11**2*a13*a33 + D44*a11*a13**2*a31 - D44*a11*a21*a23*a33 - D44*a13*a21*a23*a31 + D66*a12**2*a13*a33 + D66*a12*a13**2*a32 - D66*a12*a22*a23*a33 - D66*a13*a22*a23*a32 + D88*a11**3*a31/6 + D88*a11**2*a12*a32/6 - D88*a11**2*a13*a33/3 + D88*a11*a12**2*a31/6 - D88*a11*a13**2*a31/3 - D88*a11*a21**2*a31/6 - D88*a11*a22**2*a31/6 + D88*a11*a23**2*a31/3 + D88*a12**3*a32/6 - D88*a12**2*a13*a33/3 - D88*a12*a13**2*a32/3 - D88*a12*a21**2*a32/6 - D88*a12*a22**2*a32/6 + D88*a12*a23**2*a32/3 + 2*D88*a13**3*a33/3 + D88*a13*a21**2*a33/3 + D88*a13*a22**2*a33/3 - 2*D88*a13*a23**2*a33/3, D11*a11**2*a32**2 + 2*D11*a11*a12*a31*a32 + D11*a12**2*a31**2 + D33*a11**2*a31**2 - 2*D33*a11*a12*a31*a32 + D33*a12**2*a32**2 + D44*a11**2*a33**2 + 2*D44*a11*a13*a31*a33 + D44*a13**2*a31**2 + D66*a12**2*a33**2 + 2*D66*a12*a13*a32*a33 + D66*a13**2*a32**2 + D88*a11**2*a31**2/3 + 2*D88*a11*a12*a31*a32/3 - 4*D88*a11*a13*a31*a33/3 + D88*a12**2*a32**2/3 - 4*D88*a12*a13*a32*a33/3 + 4*D88*a13**2*a33**2/3, a, D11*a11*a21*a32**2 + D11*a11*a22*a31*a32 + D11*a12*a21*a31*a32 + D11*a12*a22*a31**2 + D33*a11*a21*a31**2 - D33*a11*a22*a31*a32 - D33*a12*a21*a31*a32 + D33*a12*a22*a32**2 + D44*a11*a21*a33**2 + D44*a11*a23*a31*a33 + D44*a13*a21*a31*a33 + D44*a13*a23*a31**2 + D66*a12*a22*a33**2 + D66*a12*a23*a32*a33 + D66*a13*a22*a32*a33 + D66*a13*a23*a32**2 + D88*a11*a21*a31**2/3 + D88*a11*a22*a31*a32/3 - 2*D88*a11*a23*a31*a33/3 + D88*a12*a21*a31*a32/3 + D88*a12*a22*a32**2/3 - 2*D88*a12*a23*a32*a33/3 - 2*D88*a13*a21*a31*a33/3 - 2*D88*a13*a22*a32*a33/3 + 4*D88*a13*a23*a33**2/3, a, sqrt(3)*D11*a11**2*a12*a32/3 + sqrt(3)*D11*a11*a12**2*a31/3 + sqrt(3)*D11*a11*a21*a22*a32/3 - 2*sqrt(3)*D11*a11*a31*a32**2/3 + sqrt(3)*D11*a12*a21*a22*a31/3 - 2*sqrt(3)*D11*a12*a31**2*a32/3 + sqrt(3)*D33*a11**3*a31/6 - sqrt(3)*D33*a11**2*a12*a32/6 - sqrt(3)*D33*a11*a12**2*a31/6 + sqrt(3)*D33*a11*a21**2*a31/6 - sqrt(3)*D33*a11*a22**2*a31/6 - sqrt(3)*D33*a11*a31**3/3 + sqrt(3)*D33*a11*a31*a32**2/3 + sqrt(3)*D33*a12**3*a32/6 - sqrt(3)*D33*a12*a21**2*a32/6 + sqrt(3)*D33*a12*a22**2*a32/6 + sqrt(3)*D33*a12*a31**2*a32/3 - sqrt(3)*D33*a12*a32**3/3 + sqrt(3)*D44*a11**2*a13*a33/3 + sqrt(3)*D44*a11*a13**2*a31/3 + sqrt(3)*D44*a11*a21*a23*a33/3 - 2*sqrt(3)*D44*a11*a31*a33**2/3 + sqrt(3)*D44*a13*a21*a23*a31/3 - 2*sqrt(3)*D44*a13*a31**2*a33/3 + sqrt(3)*D66*a12**2*a13*a33/3 + sqrt(3)*D66*a12*a13**2*a32/3 + sqrt(3)*D66*a12*a22*a23*a33/3 - 2*sqrt(3)*D66*a12*a32*a33**2/3 + sqrt(3)*D66*a13*a22*a23*a32/3 - 2*sqrt(3)*D66*a13*a32**2*a33/3 + sqrt(3)*D88*a11**3*a31/18 + sqrt(3)*D88*a11**2*a12*a32/18 - sqrt(3)*D88*a11**2*a13*a33/9 + sqrt(3)*D88*a11*a12**2*a31/18 - sqrt(3)*D88*a11*a13**2*a31/9 + sqrt(3)*D88*a11*a21**2*a31/18 + sqrt(3)*D88*a11*a22**2*a31/18 - sqrt(3)*D88*a11*a23**2*a31/9 - sqrt(3)*D88*a11*a31**3/9 - sqrt(3)*D88*a11*a31*a32**2/9 + 2*sqrt(3)*D88*a11*a31*a33**2/9 + sqrt(3)*D88*a12**3*a32/18 - sqrt(3)*D88*a12**2*a13*a33/9 - sqrt(3)*D88*a12*a13**2*a32/9 + sqrt(3)*D88*a12*a21**2*a32/18 + sqrt(3)*D88*a12*a22**2*a32/18 - sqrt(3)*D88*a12*a23**2*a32/9 - sqrt(3)*D88*a12*a31**2*a32/9 - sqrt(3)*D88*a12*a32**3/9 + 2*sqrt(3)*D88*a12*a32*a33**2/9 + 2*sqrt(3)*D88*a13**3*a33/9 - sqrt(3)*D88*a13*a21**2*a33/9 - sqrt(3)*D88*a13*a22**2*a33/9 + 2*sqrt(3)*D88*a13*a23**2*a33/9 + 2*sqrt(3)*D88*a13*a31**2*a33/9 + 2*sqrt(3)*D88*a13*a32**2*a33/9 - 4*sqrt(3)*D88*a13*a33**3/9], 
                        [a, a, D22*a11**2*a22*a32 - D22*a11*a12*a21*a32 - D22*a11*a12*a22*a31 + D22*a12**2*a21*a31 + D55*a11**2*a23*a33 - D55*a11*a13*a21*a33 - D55*a11*a13*a23*a31 + D55*a13**2*a21*a31 + D77*a12**2*a23*a33 - D77*a12*a13*a22*a33 - D77*a12*a13*a23*a32 + D77*a13**2*a22*a32, a, a, D22*a11**2*a32**2 - 2*D22*a11*a12*a31*a32 + D22*a12**2*a31**2 + D55*a11**2*a33**2 - 2*D55*a11*a13*a31*a33 + D55*a13**2*a31**2 + D77*a12**2*a33**2 - 2*D77*a12*a13*a32*a33 + D77*a13**2*a32**2, a, D22*a11*a21*a32**2 - D22*a11*a22*a31*a32 - D22*a12*a21*a31*a32 + D22*a12*a22*a31**2 + D55*a11*a21*a33**2 - D55*a11*a23*a31*a33 - D55*a13*a21*a31*a33 + D55*a13*a23*a31**2 + D77*a12*a22*a33**2 - D77*a12*a23*a32*a33 - D77*a13*a22*a32*a33 + D77*a13*a23*a32**2, a], 
                        [a, D11*a11*a21*a22*a32 + D11*a11*a22**2*a31 + D11*a12*a21**2*a32 + D11*a12*a21*a22*a31 + D33*a11*a21**2*a31 - D33*a11*a21*a22*a32 - D33*a12*a21*a22*a31 + D33*a12*a22**2*a32 + D44*a11*a21*a23*a33 + D44*a11*a23**2*a31 + D44*a13*a21**2*a33 + D44*a13*a21*a23*a31 + D66*a12*a22*a23*a33 + D66*a12*a23**2*a32 + D66*a13*a22**2*a33 + D66*a13*a22*a23*a32 + D88*a11*a21**2*a31/3 + D88*a11*a21*a22*a32/3 - 2*D88*a11*a21*a23*a33/3 + D88*a12*a21*a22*a31/3 + D88*a12*a22**2*a32/3 - 2*D88*a12*a22*a23*a33/3 - 2*D88*a13*a21*a23*a31/3 - 2*D88*a13*a22*a23*a32/3 + 4*D88*a13*a23**2*a33/3, a, D11*a11*a12*a21*a32 + D11*a11*a12*a22*a31 - D11*a21**2*a22*a32 - D11*a21*a22**2*a31 + D33*a11**2*a21*a31/2 - D33*a11**2*a22*a32/2 - D33*a12**2*a21*a31/2 + D33*a12**2*a22*a32/2 - D33*a21**3*a31/2 + D33*a21**2*a22*a32/2 + D33*a21*a22**2*a31/2 - D33*a22**3*a32/2 + D44*a11*a13*a21*a33 + D44*a11*a13*a23*a31 - D44*a21**2*a23*a33 - D44*a21*a23**2*a31 + D66*a12*a13*a22*a33 + D66*a12*a13*a23*a32 - D66*a22**2*a23*a33 - D66*a22*a23**2*a32 + D88*a11**2*a21*a31/6 + D88*a11**2*a22*a32/6 - D88*a11**2*a23*a33/3 + D88*a12**2*a21*a31/6 + D88*a12**2*a22*a32/6 - D88*a12**2*a23*a33/3 - D88*a13**2*a21*a31/3 - D88*a13**2*a22*a32/3 + 2*D88*a13**2*a23*a33/3 - D88*a21**3*a31/6 - D88*a21**2*a22*a32/6 + D88*a21**2*a23*a33/3 - D88*a21*a22**2*a31/6 + D88*a21*a23**2*a31/3 - D88*a22**3*a32/6 + D88*a22**2*a23*a33/3 + D88*a22*a23**2*a32/3 - 2*D88*a23**3*a33/3, D11*a11*a21*a32**2 + D11*a11*a22*a31*a32 + D11*a12*a21*a31*a32 + D11*a12*a22*a31**2 + D33*a11*a21*a31**2 - D33*a11*a22*a31*a32 - D33*a12*a21*a31*a32 + D33*a12*a22*a32**2 + D44*a11*a21*a33**2 + D44*a11*a23*a31*a33 + D44*a13*a21*a31*a33 + D44*a13*a23*a31**2 + D66*a12*a22*a33**2 + D66*a12*a23*a32*a33 + D66*a13*a22*a32*a33 + D66*a13*a23*a32**2 + D88*a11*a21*a31**2/3 + D88*a11*a22*a31*a32/3 - 2*D88*a11*a23*a31*a33/3 + D88*a12*a21*a31*a32/3 + D88*a12*a22*a32**2/3 - 2*D88*a12*a23*a32*a33/3 - 2*D88*a13*a21*a31*a33/3 - 2*D88*a13*a22*a32*a33/3 + 4*D88*a13*a23*a33**2/3, a, D11*a21**2*a32**2 + 2*D11*a21*a22*a31*a32 + D11*a22**2*a31**2 + D33*a21**2*a31**2 - 2*D33*a21*a22*a31*a32 + D33*a22**2*a32**2 + D44*a21**2*a33**2 + 2*D44*a21*a23*a31*a33 + D44*a23**2*a31**2 + D66*a22**2*a33**2 + 2*D66*a22*a23*a32*a33 + D66*a23**2*a32**2 + D88*a21**2*a31**2/3 + 2*D88*a21*a22*a31*a32/3 - 4*D88*a21*a23*a31*a33/3 + D88*a22**2*a32**2/3 - 4*D88*a22*a23*a32*a33/3 + 4*D88*a23**2*a33**2/3, a, sqrt(3)*D11*a11*a12*a21*a32/3 + sqrt(3)*D11*a11*a12*a22*a31/3 + sqrt(3)*D11*a21**2*a22*a32/3 + sqrt(3)*D11*a21*a22**2*a31/3 - 2*sqrt(3)*D11*a21*a31*a32**2/3 - 2*sqrt(3)*D11*a22*a31**2*a32/3 + sqrt(3)*D33*a11**2*a21*a31/6 - sqrt(3)*D33*a11**2*a22*a32/6 - sqrt(3)*D33*a12**2*a21*a31/6 + sqrt(3)*D33*a12**2*a22*a32/6 + sqrt(3)*D33*a21**3*a31/6 - sqrt(3)*D33*a21**2*a22*a32/6 - sqrt(3)*D33*a21*a22**2*a31/6 - sqrt(3)*D33*a21*a31**3/3 + sqrt(3)*D33*a21*a31*a32**2/3 + sqrt(3)*D33*a22**3*a32/6 + sqrt(3)*D33*a22*a31**2*a32/3 - sqrt(3)*D33*a22*a32**3/3 + sqrt(3)*D44*a11*a13*a21*a33/3 + sqrt(3)*D44*a11*a13*a23*a31/3 + sqrt(3)*D44*a21**2*a23*a33/3 + sqrt(3)*D44*a21*a23**2*a31/3 - 2*sqrt(3)*D44*a21*a31*a33**2/3 - 2*sqrt(3)*D44*a23*a31**2*a33/3 + sqrt(3)*D66*a12*a13*a22*a33/3 + sqrt(3)*D66*a12*a13*a23*a32/3 + sqrt(3)*D66*a22**2*a23*a33/3 + sqrt(3)*D66*a22*a23**2*a32/3 - 2*sqrt(3)*D66*a22*a32*a33**2/3 - 2*sqrt(3)*D66*a23*a32**2*a33/3 + sqrt(3)*D88*a11**2*a21*a31/18 + sqrt(3)*D88*a11**2*a22*a32/18 - sqrt(3)*D88*a11**2*a23*a33/9 + sqrt(3)*D88*a12**2*a21*a31/18 + sqrt(3)*D88*a12**2*a22*a32/18 - sqrt(3)*D88*a12**2*a23*a33/9 - sqrt(3)*D88*a13**2*a21*a31/9 - sqrt(3)*D88*a13**2*a22*a32/9 + 2*sqrt(3)*D88*a13**2*a23*a33/9 + sqrt(3)*D88*a21**3*a31/18 + sqrt(3)*D88*a21**2*a22*a32/18 - sqrt(3)*D88*a21**2*a23*a33/9 + sqrt(3)*D88*a21*a22**2*a31/18 - sqrt(3)*D88*a21*a23**2*a31/9 - sqrt(3)*D88*a21*a31**3/9 - sqrt(3)*D88*a21*a31*a32**2/9 + 2*sqrt(3)*D88*a21*a31*a33**2/9 + sqrt(3)*D88*a22**3*a32/18 - sqrt(3)*D88*a22**2*a23*a33/9 - sqrt(3)*D88*a22*a23**2*a32/9 - sqrt(3)*D88*a22*a31**2*a32/9 - sqrt(3)*D88*a22*a32**3/9 + 2*sqrt(3)*D88*a22*a32*a33**2/9 + 2*sqrt(3)*D88*a23**3*a33/9 + 2*sqrt(3)*D88*a23*a31**2*a33/9 + 2*sqrt(3)*D88*a23*a32**2*a33/9 - 4*sqrt(3)*D88*a23*a33**3/9], 
                        [a, a, D22*a11*a21*a22*a32 - D22*a11*a22**2*a31 - D22*a12*a21**2*a32 + D22*a12*a21*a22*a31 + D55*a11*a21*a23*a33 - D55*a11*a23**2*a31 - D55*a13*a21**2*a33 + D55*a13*a21*a23*a31 + D77*a12*a22*a23*a33 - D77*a12*a23**2*a32 - D77*a13*a22**2*a33 + D77*a13*a22*a23*a32, a, a, D22*a11*a21*a32**2 - D22*a11*a22*a31*a32 - D22*a12*a21*a31*a32 + D22*a12*a22*a31**2 + D55*a11*a21*a33**2 - D55*a11*a23*a31*a33 - D55*a13*a21*a31*a33 + D55*a13*a23*a31**2 + D77*a12*a22*a33**2 - D77*a12*a23*a32*a33 - D77*a13*a22*a32*a33 + D77*a13*a23*a32**2, a, D22*a21**2*a32**2 - 2*D22*a21*a22*a31*a32 + D22*a22**2*a31**2 + D55*a21**2*a33**2 - 2*D55*a21*a23*a31*a33 + D55*a23**2*a31**2 + D77*a22**2*a33**2 - 2*D77*a22*a23*a32*a33 + D77*a23**2*a32**2, a], 
                        [a, sqrt(3)*D11*a11**2*a12*a22/3 + sqrt(3)*D11*a11*a12**2*a21/3 + sqrt(3)*D11*a11*a21*a22**2/3 - 2*sqrt(3)*D11*a11*a22*a31*a32/3 + sqrt(3)*D11*a12*a21**2*a22/3 - 2*sqrt(3)*D11*a12*a21*a31*a32/3 + sqrt(3)*D33*a11**3*a21/6 - sqrt(3)*D33*a11**2*a12*a22/6 - sqrt(3)*D33*a11*a12**2*a21/6 + sqrt(3)*D33*a11*a21**3/6 - sqrt(3)*D33*a11*a21*a22**2/6 - sqrt(3)*D33*a11*a21*a31**2/3 + sqrt(3)*D33*a11*a21*a32**2/3 + sqrt(3)*D33*a12**3*a22/6 - sqrt(3)*D33*a12*a21**2*a22/6 + sqrt(3)*D33*a12*a22**3/6 + sqrt(3)*D33*a12*a22*a31**2/3 - sqrt(3)*D33*a12*a22*a32**2/3 + sqrt(3)*D44*a11**2*a13*a23/3 + sqrt(3)*D44*a11*a13**2*a21/3 + sqrt(3)*D44*a11*a21*a23**2/3 - 2*sqrt(3)*D44*a11*a23*a31*a33/3 + sqrt(3)*D44*a13*a21**2*a23/3 - 2*sqrt(3)*D44*a13*a21*a31*a33/3 + sqrt(3)*D66*a12**2*a13*a23/3 + sqrt(3)*D66*a12*a13**2*a22/3 + sqrt(3)*D66*a12*a22*a23**2/3 - 2*sqrt(3)*D66*a12*a23*a32*a33/3 + sqrt(3)*D66*a13*a22**2*a23/3 - 2*sqrt(3)*D66*a13*a22*a32*a33/3 + sqrt(3)*D88*a11**3*a21/18 + sqrt(3)*D88*a11**2*a12*a22/18 - sqrt(3)*D88*a11**2*a13*a23/9 + sqrt(3)*D88*a11*a12**2*a21/18 - sqrt(3)*D88*a11*a13**2*a21/9 + sqrt(3)*D88*a11*a21**3/18 + sqrt(3)*D88*a11*a21*a22**2/18 - sqrt(3)*D88*a11*a21*a23**2/9 - sqrt(3)*D88*a11*a21*a31**2/9 - sqrt(3)*D88*a11*a21*a32**2/9 + 2*sqrt(3)*D88*a11*a21*a33**2/9 + sqrt(3)*D88*a12**3*a22/18 - sqrt(3)*D88*a12**2*a13*a23/9 - sqrt(3)*D88*a12*a13**2*a22/9 + sqrt(3)*D88*a12*a21**2*a22/18 + sqrt(3)*D88*a12*a22**3/18 - sqrt(3)*D88*a12*a22*a23**2/9 - sqrt(3)*D88*a12*a22*a31**2/9 - sqrt(3)*D88*a12*a22*a32**2/9 + 2*sqrt(3)*D88*a12*a22*a33**2/9 + 2*sqrt(3)*D88*a13**3*a23/9 - sqrt(3)*D88*a13*a21**2*a23/9 - sqrt(3)*D88*a13*a22**2*a23/9 + 2*sqrt(3)*D88*a13*a23**3/9 + 2*sqrt(3)*D88*a13*a23*a31**2/9 + 2*sqrt(3)*D88*a13*a23*a32**2/9 - 4*sqrt(3)*D88*a13*a23*a33**2/9, a, sqrt(3)*D11*a11**2*a12**2/3 - 2*sqrt(3)*D11*a11*a12*a31*a32/3 - sqrt(3)*D11*a21**2*a22**2/3 + 2*sqrt(3)*D11*a21*a22*a31*a32/3 + sqrt(3)*D33*a11**4/12 - sqrt(3)*D33*a11**2*a12**2/6 - sqrt(3)*D33*a11**2*a31**2/6 + sqrt(3)*D33*a11**2*a32**2/6 + sqrt(3)*D33*a12**4/12 + sqrt(3)*D33*a12**2*a31**2/6 - sqrt(3)*D33*a12**2*a32**2/6 - sqrt(3)*D33*a21**4/12 + sqrt(3)*D33*a21**2*a22**2/6 + sqrt(3)*D33*a21**2*a31**2/6 - sqrt(3)*D33*a21**2*a32**2/6 - sqrt(3)*D33*a22**4/12 - sqrt(3)*D33*a22**2*a31**2/6 + sqrt(3)*D33*a22**2*a32**2/6 + sqrt(3)*D44*a11**2*a13**2/3 - 2*sqrt(3)*D44*a11*a13*a31*a33/3 - sqrt(3)*D44*a21**2*a23**2/3 + 2*sqrt(3)*D44*a21*a23*a31*a33/3 + sqrt(3)*D66*a12**2*a13**2/3 - 2*sqrt(3)*D66*a12*a13*a32*a33/3 - sqrt(3)*D66*a22**2*a23**2/3 + 2*sqrt(3)*D66*a22*a23*a32*a33/3 + sqrt(3)*D88*a11**4/36 + sqrt(3)*D88*a11**2*a12**2/18 - sqrt(3)*D88*a11**2*a13**2/9 - sqrt(3)*D88*a11**2*a31**2/18 - sqrt(3)*D88*a11**2*a32**2/18 + sqrt(3)*D88*a11**2*a33**2/9 + sqrt(3)*D88*a12**4/36 - sqrt(3)*D88*a12**2*a13**2/9 - sqrt(3)*D88*a12**2*a31**2/18 - sqrt(3)*D88*a12**2*a32**2/18 + sqrt(3)*D88*a12**2*a33**2/9 + sqrt(3)*D88*a13**4/9 + sqrt(3)*D88*a13**2*a31**2/9 + sqrt(3)*D88*a13**2*a32**2/9 - 2*sqrt(3)*D88*a13**2*a33**2/9 - sqrt(3)*D88*a21**4/36 - sqrt(3)*D88*a21**2*a22**2/18 + sqrt(3)*D88*a21**2*a23**2/9 + sqrt(3)*D88*a21**2*a31**2/18 + sqrt(3)*D88*a21**2*a32**2/18 - sqrt(3)*D88*a21**2*a33**2/9 - sqrt(3)*D88*a22**4/36 + sqrt(3)*D88*a22**2*a23**2/9 + sqrt(3)*D88*a22**2*a31**2/18 + sqrt(3)*D88*a22**2*a32**2/18 - sqrt(3)*D88*a22**2*a33**2/9 - sqrt(3)*D88*a23**4/9 - sqrt(3)*D88*a23**2*a31**2/9 - sqrt(3)*D88*a23**2*a32**2/9 + 2*sqrt(3)*D88*a23**2*a33**2/9, sqrt(3)*D11*a11**2*a12*a32/3 + sqrt(3)*D11*a11*a12**2*a31/3 + sqrt(3)*D11*a11*a21*a22*a32/3 - 2*sqrt(3)*D11*a11*a31*a32**2/3 + sqrt(3)*D11*a12*a21*a22*a31/3 - 2*sqrt(3)*D11*a12*a31**2*a32/3 + sqrt(3)*D33*a11**3*a31/6 - sqrt(3)*D33*a11**2*a12*a32/6 - sqrt(3)*D33*a11*a12**2*a31/6 + sqrt(3)*D33*a11*a21**2*a31/6 - sqrt(3)*D33*a11*a22**2*a31/6 - sqrt(3)*D33*a11*a31**3/3 + sqrt(3)*D33*a11*a31*a32**2/3 + sqrt(3)*D33*a12**3*a32/6 - sqrt(3)*D33*a12*a21**2*a32/6 + sqrt(3)*D33*a12*a22**2*a32/6 + sqrt(3)*D33*a12*a31**2*a32/3 - sqrt(3)*D33*a12*a32**3/3 + sqrt(3)*D44*a11**2*a13*a33/3 + sqrt(3)*D44*a11*a13**2*a31/3 + sqrt(3)*D44*a11*a21*a23*a33/3 - 2*sqrt(3)*D44*a11*a31*a33**2/3 + sqrt(3)*D44*a13*a21*a23*a31/3 - 2*sqrt(3)*D44*a13*a31**2*a33/3 + sqrt(3)*D66*a12**2*a13*a33/3 + sqrt(3)*D66*a12*a13**2*a32/3 + sqrt(3)*D66*a12*a22*a23*a33/3 - 2*sqrt(3)*D66*a12*a32*a33**2/3 + sqrt(3)*D66*a13*a22*a23*a32/3 - 2*sqrt(3)*D66*a13*a32**2*a33/3 + sqrt(3)*D88*a11**3*a31/18 + sqrt(3)*D88*a11**2*a12*a32/18 - sqrt(3)*D88*a11**2*a13*a33/9 + sqrt(3)*D88*a11*a12**2*a31/18 - sqrt(3)*D88*a11*a13**2*a31/9 + sqrt(3)*D88*a11*a21**2*a31/18 + sqrt(3)*D88*a11*a22**2*a31/18 - sqrt(3)*D88*a11*a23**2*a31/9 - sqrt(3)*D88*a11*a31**3/9 - sqrt(3)*D88*a11*a31*a32**2/9 + 2*sqrt(3)*D88*a11*a31*a33**2/9 + sqrt(3)*D88*a12**3*a32/18 - sqrt(3)*D88*a12**2*a13*a33/9 - sqrt(3)*D88*a12*a13**2*a32/9 + sqrt(3)*D88*a12*a21**2*a32/18 + sqrt(3)*D88*a12*a22**2*a32/18 - sqrt(3)*D88*a12*a23**2*a32/9 - sqrt(3)*D88*a12*a31**2*a32/9 - sqrt(3)*D88*a12*a32**3/9 + 2*sqrt(3)*D88*a12*a32*a33**2/9 + 2*sqrt(3)*D88*a13**3*a33/9 - sqrt(3)*D88*a13*a21**2*a33/9 - sqrt(3)*D88*a13*a22**2*a33/9 + 2*sqrt(3)*D88*a13*a23**2*a33/9 + 2*sqrt(3)*D88*a13*a31**2*a33/9 + 2*sqrt(3)*D88*a13*a32**2*a33/9 - 4*sqrt(3)*D88*a13*a33**3/9, a, sqrt(3)*D11*a11*a12*a21*a32/3 + sqrt(3)*D11*a11*a12*a22*a31/3 + sqrt(3)*D11*a21**2*a22*a32/3 + sqrt(3)*D11*a21*a22**2*a31/3 - 2*sqrt(3)*D11*a21*a31*a32**2/3 - 2*sqrt(3)*D11*a22*a31**2*a32/3 + sqrt(3)*D33*a11**2*a21*a31/6 - sqrt(3)*D33*a11**2*a22*a32/6 - sqrt(3)*D33*a12**2*a21*a31/6 + sqrt(3)*D33*a12**2*a22*a32/6 + sqrt(3)*D33*a21**3*a31/6 - sqrt(3)*D33*a21**2*a22*a32/6 - sqrt(3)*D33*a21*a22**2*a31/6 - sqrt(3)*D33*a21*a31**3/3 + sqrt(3)*D33*a21*a31*a32**2/3 + sqrt(3)*D33*a22**3*a32/6 + sqrt(3)*D33*a22*a31**2*a32/3 - sqrt(3)*D33*a22*a32**3/3 + sqrt(3)*D44*a11*a13*a21*a33/3 + sqrt(3)*D44*a11*a13*a23*a31/3 + sqrt(3)*D44*a21**2*a23*a33/3 + sqrt(3)*D44*a21*a23**2*a31/3 - 2*sqrt(3)*D44*a21*a31*a33**2/3 - 2*sqrt(3)*D44*a23*a31**2*a33/3 + sqrt(3)*D66*a12*a13*a22*a33/3 + sqrt(3)*D66*a12*a13*a23*a32/3 + sqrt(3)*D66*a22**2*a23*a33/3 + sqrt(3)*D66*a22*a23**2*a32/3 - 2*sqrt(3)*D66*a22*a32*a33**2/3 - 2*sqrt(3)*D66*a23*a32**2*a33/3 + sqrt(3)*D88*a11**2*a21*a31/18 + sqrt(3)*D88*a11**2*a22*a32/18 - sqrt(3)*D88*a11**2*a23*a33/9 + sqrt(3)*D88*a12**2*a21*a31/18 + sqrt(3)*D88*a12**2*a22*a32/18 - sqrt(3)*D88*a12**2*a23*a33/9 - sqrt(3)*D88*a13**2*a21*a31/9 - sqrt(3)*D88*a13**2*a22*a32/9 + 2*sqrt(3)*D88*a13**2*a23*a33/9 + sqrt(3)*D88*a21**3*a31/18 + sqrt(3)*D88*a21**2*a22*a32/18 - sqrt(3)*D88*a21**2*a23*a33/9 + sqrt(3)*D88*a21*a22**2*a31/18 - sqrt(3)*D88*a21*a23**2*a31/9 - sqrt(3)*D88*a21*a31**3/9 - sqrt(3)*D88*a21*a31*a32**2/9 + 2*sqrt(3)*D88*a21*a31*a33**2/9 + sqrt(3)*D88*a22**3*a32/18 - sqrt(3)*D88*a22**2*a23*a33/9 - sqrt(3)*D88*a22*a23**2*a32/9 - sqrt(3)*D88*a22*a31**2*a32/9 - sqrt(3)*D88*a22*a32**3/9 + 2*sqrt(3)*D88*a22*a32*a33**2/9 + 2*sqrt(3)*D88*a23**3*a33/9 + 2*sqrt(3)*D88*a23*a31**2*a33/9 + 2*sqrt(3)*D88*a23*a32**2*a33/9 - 4*sqrt(3)*D88*a23*a33**3/9, a, D11*a11**2*a12**2/3 + 2*D11*a11*a12*a21*a22/3 - 4*D11*a11*a12*a31*a32/3 + D11*a21**2*a22**2/3 - 4*D11*a21*a22*a31*a32/3 + 4*D11*a31**2*a32**2/3 + D33*a11**4/12 - D33*a11**2*a12**2/6 + D33*a11**2*a21**2/6 - D33*a11**2*a22**2/6 - D33*a11**2*a31**2/3 + D33*a11**2*a32**2/3 + D33*a12**4/12 - D33*a12**2*a21**2/6 + D33*a12**2*a22**2/6 + D33*a12**2*a31**2/3 - D33*a12**2*a32**2/3 + D33*a21**4/12 - D33*a21**2*a22**2/6 - D33*a21**2*a31**2/3 + D33*a21**2*a32**2/3 + D33*a22**4/12 + D33*a22**2*a31**2/3 - D33*a22**2*a32**2/3 + D33*a31**4/3 - 2*D33*a31**2*a32**2/3 + D33*a32**4/3 + D44*a11**2*a13**2/3 + 2*D44*a11*a13*a21*a23/3 - 4*D44*a11*a13*a31*a33/3 + D44*a21**2*a23**2/3 - 4*D44*a21*a23*a31*a33/3 + 4*D44*a31**2*a33**2/3 + D66*a12**2*a13**2/3 + 2*D66*a12*a13*a22*a23/3 - 4*D66*a12*a13*a32*a33/3 + D66*a22**2*a23**2/3 - 4*D66*a22*a23*a32*a33/3 + 4*D66*a32**2*a33**2/3 + D88*a11**4/36 + D88*a11**2*a12**2/18 - D88*a11**2*a13**2/9 + D88*a11**2*a21**2/18 + D88*a11**2*a22**2/18 - D88*a11**2*a23**2/9 - D88*a11**2*a31**2/9 - D88*a11**2*a32**2/9 + 2*D88*a11**2*a33**2/9 + D88*a12**4/36 - D88*a12**2*a13**2/9 + D88*a12**2*a21**2/18 + D88*a12**2*a22**2/18 - D88*a12**2*a23**2/9 - D88*a12**2*a31**2/9 - D88*a12**2*a32**2/9 + 2*D88*a12**2*a33**2/9 + D88*a13**4/9 - D88*a13**2*a21**2/9 - D88*a13**2*a22**2/9 + 2*D88*a13**2*a23**2/9 + 2*D88*a13**2*a31**2/9 + 2*D88*a13**2*a32**2/9 - 4*D88*a13**2*a33**2/9 + D88*a21**4/36 + D88*a21**2*a22**2/18 - D88*a21**2*a23**2/9 - D88*a21**2*a31**2/9 - D88*a21**2*a32**2/9 + 2*D88*a21**2*a33**2/9 + D88*a22**4/36 - D88*a22**2*a23**2/9 - D88*a22**2*a31**2/9 - D88*a22**2*a32**2/9 + 2*D88*a22**2*a33**2/9 + D88*a23**4/9 + 2*D88*a23**2*a31**2/9 + 2*D88*a23**2*a32**2/9 - 4*D88*a23**2*a33**2/9 + D88*a31**4/9 + 2*D88*a31**2*a32**2/9 - 4*D88*a31**2*a33**2/9 + D88*a32**4/9 - 4*D88*a32**2*a33**2/9 + 4*D88*a33**4/9]])
            return Dm

        Dm = -2.* Dm_function(a_vec, Gamma_vec)
        return Hlm+Dm

    #r in km, ne in eV^3
    def ne_func(r):
        ne_f = np.interp(r, r_data, ne)
        return ne_f

    Pij_list = []
    k = 0
    rhom_list = []
    if ij[1] == '1':
        o=1
        p=q=0
    elif ij[1] == '2':
        o=q=0
        p=1
    elif ij[1] == '3':
        o=p=0
        q=1

    rhom_0_j = array([[o,0,0],[0,p,0],[0,0,q]], dtype=np.complex128)
    rhom_list.append(rhom_0_j)

    rm0 = 1/3
    rm1 = 1/2*trace(rhom_0_j @ l1)
    rm2 = 1/2*trace(rhom_0_j @ l2)
    rm3 = 1/2*trace(rhom_0_j @ l3)
    rm4 = 1/2*trace(rhom_0_j @ l4)
    rm5 = 1/2*trace(rhom_0_j @ l5)
    rm6 = 1/2*trace(rhom_0_j @ l6)
    rm7 = 1/2*trace(rhom_0_j @ l7)
    rm8 = 1/2*trace(rhom_0_j @ l8)
    rhom_vec_0 = np.transpose(array([rm0, rm1, rm2, rm3, rm4, rm5, rm6, rm7, rm8], dtype=np.complex128))

    #loop over all points in the SN potential
    for r in r_list_pijsn[1:]:
        vcc0 = math.sqrt(2) * Gf * ne_func(r)
        id = list(r_list_pijsn).index(r)
        Delta_x_km = r - r_list_pijsn[id-1]
        Delta_x_eV = Delta_x_km/(0.197e9 * 1e-15 / 1000) #eV^-1
        Hlm_D = lindbladian(E, vcc0)
        # U = expm(Hlm_D*Delta_x_eV)
        M = Hlm_D*Delta_x_eV
        U = pade_approximation(M)

        if k == 0:
            rhom_vec = U @ rhom_vec_0
        else:
            rhom_vec = U @ rhom_vec

        rhom0 = rhom_vec[0]
        rhom1 = rhom_vec[1]
        rhom2 = rhom_vec[2]
        rhom3 = rhom_vec[3]
        rhom4 = rhom_vec[4]
        rhom5 = rhom_vec[5]
        rhom6 = rhom_vec[6]
        rhom7 = rhom_vec[7]
        rhom8 = rhom_vec[8]
        rhom =   array([[rhom0 + rhom3 + sqrt(3)*rhom8/3, rhom1 - i*rhom2, rhom4 - i*rhom5], 
                        [rhom1 + i*rhom2, rhom0 - rhom3 + sqrt(3)*rhom8/3, rhom6 - i*rhom7], 
                        [rhom4 + i*rhom5, rhom6 + i*rhom7, rhom0 - 2*sqrt(3)*rhom8/3]], dtype=np.complex128)
        
        rhom_list.append(rhom)
        ind = len(rhom_list)-1
        rhom_j = rhom_list[ind]
        if ij[0] == '1':
            u=1
            v=x=0
        elif ij[0] == '2':
            u=x=0
            v=1
        elif ij[0] == '3':
            u=v=0
            x=1
        rhom_0_i = array([[u,0,0],[0,v,0],[0,0,x]], dtype=np.complex128)
        Pij = np.trace(rhom_0_i @ rhom_j)
        Pij_list.append(Pij.real)
        k = k+1
    Pij_list = np.array(Pij_list)
    # return np.mean(Pij_list[-10:])
    return Pij_list, r_list_pijsn[1:]

#uncomment next line to save compiled compatible to OS
# if __name__ == "__main__":
#     cc.compile()