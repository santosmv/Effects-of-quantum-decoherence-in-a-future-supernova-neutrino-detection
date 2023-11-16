import numpy as np
from scipy.interpolate import RectBivariateSpline
from math import pi

Deltam21 = 7.42e-5 #eV² +0.21 -0.20
theta12 = 33.45 #+0.77 -0.75 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
theta12 = theta12*pi/180

nt, nw = 401,401

d21_i = Deltam21 - 0.2e-5
d21_f = Deltam21 + 0.21e-5
w21_list = np.linspace(d21_i/60e6, d21_f/1e6, nw)
t12_list = np.linspace(theta12-0.75*pi/180, theta12+0.77*pi/180, nt)


###############  DUNE  ################

dune_angles = [180,160,140,120]

d_P1e_nh_nue_120 = np.load('regeneration/P1e_NH_nue_120.0z_401th_401w21.npy')
d_P1e_nh_nue_140 = np.load('regeneration/P1e_NH_nue_140.0z_401th_401w21.npy')
d_P1e_nh_nue_160 = np.load('regeneration/P1e_NH_nue_160.0z_401th_401w21.npy')
d_P1e_nh_nue_180 = np.load('regeneration/P1e_NH_nue_180.0z_401th_401w21.npy')

d_P2e_nh_nue_120 = np.load('regeneration/P2e_NH_nue_120.0z_401th_401w21.npy')
d_P2e_nh_nue_140 = np.load('regeneration/P2e_NH_nue_140.0z_401th_401w21.npy')
d_P2e_nh_nue_160 = np.load('regeneration/P2e_NH_nue_160.0z_401th_401w21.npy')
d_P2e_nh_nue_180 = np.load('regeneration/P2e_NH_nue_180.0z_401th_401w21.npy')

d_P1e_ih_nue_120 = np.load('regeneration/P1e_IH_nue_120.0z_401th_401w21.npy')
d_P1e_ih_nue_140 = np.load('regeneration/P1e_IH_nue_140.0z_401th_401w21.npy')
d_P1e_ih_nue_160 = np.load('regeneration/P1e_IH_nue_160.0z_401th_401w21.npy')
d_P1e_ih_nue_180 = np.load('regeneration/P1e_IH_nue_180.0z_401th_401w21.npy')

d_P2e_ih_nue_120 = np.load('regeneration/P2e_IH_nue_120.0z_401th_401w21.npy')
d_P2e_ih_nue_140 = np.load('regeneration/P2e_IH_nue_140.0z_401th_401w21.npy')
d_P2e_ih_nue_160 = np.load('regeneration/P2e_IH_nue_160.0z_401th_401w21.npy')
d_P2e_ih_nue_180 = np.load('regeneration/P2e_IH_nue_180.0z_401th_401w21.npy')

d_P1e_nh_nuebar_120 = np.load('regeneration/P1e_NH_nuebar_120.0z_401th_401w21.npy')
d_P1e_nh_nuebar_140 = np.load('regeneration/P1e_NH_nuebar_140.0z_401th_401w21.npy')
d_P1e_nh_nuebar_160 = np.load('regeneration/P1e_NH_nuebar_160.0z_401th_401w21.npy')
d_P1e_nh_nuebar_180 = np.load('regeneration/P1e_NH_nuebar_180.0z_401th_401w21.npy')

d_P2e_nh_nuebar_120 = np.load('regeneration/P2e_NH_nuebar_120.0z_401th_401w21.npy')
d_P2e_nh_nuebar_140 = np.load('regeneration/P2e_NH_nuebar_140.0z_401th_401w21.npy')
d_P2e_nh_nuebar_160 = np.load('regeneration/P2e_NH_nuebar_160.0z_401th_401w21.npy')
d_P2e_nh_nuebar_180 = np.load('regeneration/P2e_NH_nuebar_180.0z_401th_401w21.npy')

d_P1e_ih_nuebar_120 = np.load('regeneration/P1e_IH_nuebar_120.0z_401th_401w21.npy')
d_P1e_ih_nuebar_140 = np.load('regeneration/P1e_IH_nuebar_140.0z_401th_401w21.npy')
d_P1e_ih_nuebar_160 = np.load('regeneration/P1e_IH_nuebar_160.0z_401th_401w21.npy')
d_P1e_ih_nuebar_180 = np.load('regeneration/P1e_IH_nuebar_180.0z_401th_401w21.npy')

d_P2e_ih_nuebar_120 = np.load('regeneration/P2e_IH_nuebar_120.0z_401th_401w21.npy')
d_P2e_ih_nuebar_140 = np.load('regeneration/P2e_IH_nuebar_140.0z_401th_401w21.npy')
d_P2e_ih_nuebar_160 = np.load('regeneration/P2e_IH_nuebar_160.0z_401th_401w21.npy')
d_P2e_ih_nuebar_180 = np.load('regeneration/P2e_IH_nuebar_180.0z_401th_401w21.npy')

P1e_nh_nue_180 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_180)
P1e_nh_nue_160 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_160)
P1e_nh_nue_140 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_140)
P1e_nh_nue_120 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_120)

P2e_nh_nue_180 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_180)
P2e_nh_nue_160 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_160)
P2e_nh_nue_140 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_140)
P2e_nh_nue_120 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_120)

P1e_ih_nue_180 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_180)
P1e_ih_nue_160 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_160)
P1e_ih_nue_140 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_140)
P1e_ih_nue_120 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_120)

P2e_ih_nue_180 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_180)
P2e_ih_nue_160 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_160)
P2e_ih_nue_140 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_140)
P2e_ih_nue_120 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_120)

P1e_nh_nuebar_180 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_180)
P1e_nh_nuebar_160 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_160)
P1e_nh_nuebar_140 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_140)
P1e_nh_nuebar_120 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_120)

P2e_nh_nuebar_180 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_180)
P2e_nh_nuebar_160 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_160)
P2e_nh_nuebar_140 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_140)
P2e_nh_nuebar_120 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_120)

P1e_ih_nuebar_180 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_180)
P1e_ih_nuebar_160 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_160)
P1e_ih_nuebar_140 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_140)
P1e_ih_nuebar_120 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_120)

P2e_ih_nuebar_180 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_180)
P2e_ih_nuebar_160 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_160)
P2e_ih_nuebar_140 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_140)
P2e_ih_nuebar_120 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_120)

P1e_nh_nue_180_func = np.vectorize(P1e_nh_nue_180)
P1e_nh_nue_160_func = np.vectorize(P1e_nh_nue_160)
P1e_nh_nue_140_func = np.vectorize(P1e_nh_nue_140)
P1e_nh_nue_120_func = np.vectorize(P1e_nh_nue_120)

P1e_ih_nue_180_func = np.vectorize(P1e_ih_nue_180)
P1e_ih_nue_160_func = np.vectorize(P1e_ih_nue_160)
P1e_ih_nue_140_func = np.vectorize(P1e_ih_nue_140)
P1e_ih_nue_120_func = np.vectorize(P1e_ih_nue_120)

P2e_nh_nue_180_func = np.vectorize(P2e_nh_nue_180)
P2e_nh_nue_160_func = np.vectorize(P2e_nh_nue_160)
P2e_nh_nue_140_func = np.vectorize(P2e_nh_nue_140)
P2e_nh_nue_120_func = np.vectorize(P2e_nh_nue_120)

P2e_ih_nue_180_func = np.vectorize(P2e_ih_nue_180)
P2e_ih_nue_160_func = np.vectorize(P2e_ih_nue_160)
P2e_ih_nue_140_func = np.vectorize(P2e_ih_nue_140)
P2e_ih_nue_120_func = np.vectorize(P2e_ih_nue_120)

P1e_nh_nuebar_180_func = np.vectorize(P1e_nh_nuebar_180)
P1e_nh_nuebar_160_func = np.vectorize(P1e_nh_nuebar_160)
P1e_nh_nuebar_140_func = np.vectorize(P1e_nh_nuebar_140)
P1e_nh_nuebar_120_func = np.vectorize(P1e_nh_nuebar_120)

P1e_ih_nuebar_180_func = np.vectorize(P1e_ih_nuebar_180)
P1e_ih_nuebar_160_func = np.vectorize(P1e_ih_nuebar_160)
P1e_ih_nuebar_140_func = np.vectorize(P1e_ih_nuebar_140)
P1e_ih_nuebar_120_func = np.vectorize(P1e_ih_nuebar_120)

P2e_nh_nuebar_180_func = np.vectorize(P2e_nh_nuebar_180)
P2e_nh_nuebar_160_func = np.vectorize(P2e_nh_nuebar_160)
P2e_nh_nuebar_140_func = np.vectorize(P2e_nh_nuebar_140)
P2e_nh_nuebar_120_func = np.vectorize(P2e_nh_nuebar_120)

P2e_ih_nuebar_180_func = np.vectorize(P2e_ih_nuebar_180)
P2e_ih_nuebar_160_func = np.vectorize(P2e_ih_nuebar_160)
P2e_ih_nuebar_140_func = np.vectorize(P2e_ih_nuebar_140)
P2e_ih_nuebar_120_func = np.vectorize(P2e_ih_nuebar_120)


###############  HK  ################

hk_angles = [97.84421607943858, 95.47306275942896, 129.64136416059688, 129.37887605122702]

d_P1e_nh_nue_97 = np.load('regeneration/P1e_NH_nue_97.8z_401th_401w21.npy')
d_P1e_nh_nue_95 = np.load('regeneration/P1e_NH_nue_95.5z_401th_401w21.npy')
d_P1e_nh_nue_129_6 = np.load('regeneration/P1e_NH_nue_129.6z_401th_401w21.npy')
d_P1e_nh_nue_129_3 = np.load('regeneration/P1e_NH_nue_129.4z_401th_401w21.npy')

d_P2e_nh_nue_97 = np.load('regeneration/P2e_NH_nue_97.8z_401th_401w21.npy')
d_P2e_nh_nue_95 = np.load('regeneration/P2e_NH_nue_95.5z_401th_401w21.npy')
d_P2e_nh_nue_129_6 = np.load('regeneration/P2e_NH_nue_129.6z_401th_401w21.npy')
d_P2e_nh_nue_129_3 = np.load('regeneration/P2e_NH_nue_129.4z_401th_401w21.npy')

d_P1e_ih_nue_97 = np.load('regeneration/P1e_NH_nue_97.8z_401th_401w21.npy')
d_P1e_ih_nue_95 = np.load('regeneration/P1e_NH_nue_95.5z_401th_401w21.npy')
d_P1e_ih_nue_129_6 = np.load('regeneration/P1e_NH_nue_129.6z_401th_401w21.npy')
d_P1e_ih_nue_129_3 = np.load('regeneration/P1e_NH_nue_129.4z_401th_401w21.npy')

d_P2e_ih_nue_97 = np.load('regeneration/P2e_IH_nue_97.8z_401th_401w21.npy')
d_P2e_ih_nue_95 = np.load('regeneration/P2e_IH_nue_95.5z_401th_401w21.npy')
d_P2e_ih_nue_129_6 = np.load('regeneration/P2e_IH_nue_129.6z_401th_401w21.npy')
d_P2e_ih_nue_129_3 = np.load('regeneration/P2e_IH_nue_129.4z_401th_401w21.npy')

d_P1e_nh_nuebar_97 = np.load('regeneration/P1e_NH_nuebar_97.8z_401th_401w21.npy')
d_P1e_nh_nuebar_95 = np.load('regeneration/P1e_NH_nuebar_95.5z_401th_401w21.npy')
d_P1e_nh_nuebar_129_6 = np.load('regeneration/P1e_NH_nuebar_129.6z_401th_401w21.npy')
d_P1e_nh_nuebar_129_3 = np.load('regeneration/P1e_NH_nuebar_129.4z_401th_401w21.npy')

d_P2e_nh_nuebar_97 = np.load('regeneration/P2e_NH_nuebar_97.8z_401th_401w21.npy')
d_P2e_nh_nuebar_95 = np.load('regeneration/P2e_NH_nuebar_95.5z_401th_401w21.npy')
d_P2e_nh_nuebar_129_6 = np.load('regeneration/P2e_NH_nuebar_129.6z_401th_401w21.npy')
d_P2e_nh_nuebar_129_3 = np.load('regeneration/P2e_NH_nuebar_129.4z_401th_401w21.npy')

d_P1e_ih_nuebar_97 = np.load('regeneration/P1e_IH_nuebar_97.8z_401th_401w21.npy')
d_P1e_ih_nuebar_95 = np.load('regeneration/P1e_IH_nuebar_95.5z_401th_401w21.npy')
d_P1e_ih_nuebar_129_6 = np.load('regeneration/P1e_IH_nuebar_129.6z_401th_401w21.npy')
d_P1e_ih_nuebar_129_3 = np.load('regeneration/P1e_IH_nuebar_129.4z_401th_401w21.npy')

d_P2e_ih_nuebar_97 = np.load('regeneration/P2e_IH_nuebar_97.8z_401th_401w21.npy')
d_P2e_ih_nuebar_95 = np.load('regeneration/P2e_IH_nuebar_95.5z_401th_401w21.npy')
d_P2e_ih_nuebar_129_6 = np.load('regeneration/P2e_IH_nuebar_129.6z_401th_401w21.npy')
d_P2e_ih_nuebar_129_3 = np.load('regeneration/P2e_IH_nuebar_129.4z_401th_401w21.npy')

P1e_nh_nue_129_3 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_129_3)
P1e_nh_nue_129_6 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_129_6)
P1e_nh_nue_95 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_95)
P1e_nh_nue_97 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_97)

P2e_nh_nue_129_3 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_129_3)
P2e_nh_nue_129_6 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_129_6)
P2e_nh_nue_95 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_95)
P2e_nh_nue_97 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_97)

P1e_ih_nue_129_3 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_129_3)
P1e_ih_nue_129_6 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_129_6)
P1e_ih_nue_95 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_95)
P1e_ih_nue_97 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_97)

P2e_ih_nue_129_3 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_129_3)
P2e_ih_nue_129_6 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_129_6)
P2e_ih_nue_95 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_95)
P2e_ih_nue_97 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_97)

P1e_nh_nuebar_129_3 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_129_3)
P1e_nh_nuebar_129_6 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_129_6)
P1e_nh_nuebar_95 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_95)
P1e_nh_nuebar_97 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_97)

P2e_nh_nuebar_129_3 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_129_3)
P2e_nh_nuebar_129_6 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_129_6)
P2e_nh_nuebar_95 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_95)
P2e_nh_nuebar_97 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_97)

P1e_ih_nuebar_129_3 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_129_3)
P1e_ih_nuebar_129_6 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_129_6)
P1e_ih_nuebar_95 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_95)
P1e_ih_nuebar_97 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_97)

P2e_ih_nuebar_129_3 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_129_3)
P2e_ih_nuebar_129_6 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_129_6)
P2e_ih_nuebar_95 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_95)
P2e_ih_nuebar_97 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_97)

P1e_nh_nue_129_3_func = np.vectorize(P1e_nh_nue_129_3)
P1e_nh_nue_129_6_func = np.vectorize(P1e_nh_nue_129_6)
P1e_nh_nue_95_func = np.vectorize(P1e_nh_nue_95)
P1e_nh_nue_97_func = np.vectorize(P1e_nh_nue_97)

P2e_nh_nue_129_3_func = np.vectorize(P2e_nh_nue_129_3)
P2e_nh_nue_129_6_func = np.vectorize(P2e_nh_nue_129_6)
P2e_nh_nue_95_func = np.vectorize(P2e_nh_nue_95)
P2e_nh_nue_97_func = np.vectorize(P2e_nh_nue_97)

P1e_ih_nue_129_3_func = np.vectorize(P1e_ih_nue_129_3)
P1e_ih_nue_129_6_func = np.vectorize(P1e_ih_nue_129_6)
P1e_ih_nue_95_func = np.vectorize(P1e_ih_nue_95)
P1e_ih_nue_97_func = np.vectorize(P1e_ih_nue_97)

P2e_ih_nue_129_3_func = np.vectorize(P2e_ih_nue_129_3)
P2e_ih_nue_129_6_func = np.vectorize(P2e_ih_nue_129_6)
P2e_ih_nue_95_func = np.vectorize(P2e_ih_nue_95)
P2e_ih_nue_97_func = np.vectorize(P2e_ih_nue_97)

P1e_nh_nuebar_129_3_func = np.vectorize(P1e_nh_nuebar_129_3)
P1e_nh_nuebar_129_6_func = np.vectorize(P1e_nh_nuebar_129_6)
P1e_nh_nuebar_95_func = np.vectorize(P1e_nh_nuebar_95)
P1e_nh_nuebar_97_func = np.vectorize(P1e_nh_nuebar_97)

P2e_nh_nuebar_129_3_func = np.vectorize(P2e_nh_nuebar_129_3)
P2e_nh_nuebar_129_6_func = np.vectorize(P2e_nh_nuebar_129_6)
P2e_nh_nuebar_95_func = np.vectorize(P2e_nh_nuebar_95)
P2e_nh_nuebar_97_func = np.vectorize(P2e_nh_nuebar_97)

P1e_ih_nuebar_129_3_func = np.vectorize(P1e_ih_nuebar_129_3)
P1e_ih_nuebar_129_6_func = np.vectorize(P1e_ih_nuebar_129_6)
P1e_ih_nuebar_95_func = np.vectorize(P1e_ih_nuebar_95)
P1e_ih_nuebar_97_func = np.vectorize(P1e_ih_nuebar_97)

P2e_ih_nuebar_129_3_func = np.vectorize(P2e_ih_nuebar_129_3)
P2e_ih_nuebar_129_6_func = np.vectorize(P2e_ih_nuebar_129_6)
P2e_ih_nuebar_95_func = np.vectorize(P2e_ih_nuebar_95)
P2e_ih_nuebar_97_func = np.vectorize(P2e_ih_nuebar_97)


###############  JUNO  ################

juno_angles = [105.7170051252261, 120.756437387585129, 146.56494382389732, 146.62548196139585]

d_P1e_nh_nue_105 = np.load('regeneration/P1e_NH_nue_105.7z_401th_401w21.npy')
d_P1e_nh_nue_120_8 = np.load('regeneration/P1e_NH_nue_120.8z_401th_401w21.npy')
d_P1e_nh_nue_146_6 = np.load('regeneration/P1e_NH_nue_146.6z_401th_401w21.npy')

d_P2e_nh_nue_105 = np.load('regeneration/P2e_NH_nue_105.7z_401th_401w21.npy')
d_P2e_nh_nue_120_8 = np.load('regeneration/P2e_NH_nue_120.8z_401th_401w21.npy')
d_P2e_nh_nue_146_6 = np.load('regeneration/P2e_NH_nue_146.6z_401th_401w21.npy')

d_P1e_ih_nue_105 = np.load('regeneration/P1e_IH_nue_105.7z_401th_401w21.npy')
d_P1e_ih_nue_120_8 = np.load('regeneration/P1e_IH_nue_120.8z_401th_401w21.npy')
d_P1e_ih_nue_146_6 = np.load('regeneration/P1e_IH_nue_146.6z_401th_401w21.npy')

d_P2e_ih_nue_105 = np.load('regeneration/P2e_IH_nue_105.7z_401th_401w21.npy')
d_P2e_ih_nue_120_8 = np.load('regeneration/P2e_IH_nue_120.8z_401th_401w21.npy')
d_P2e_ih_nue_146_6 = np.load('regeneration/P2e_IH_nue_146.6z_401th_401w21.npy')

d_P1e_nh_nuebar_105 = np.load('regeneration/P1e_NH_nuebar_105.7z_401th_401w21.npy')
d_P1e_nh_nuebar_120_8 = np.load('regeneration/P1e_NH_nuebar_120.8z_401th_401w21.npy')
d_P1e_nh_nuebar_146_6 = np.load('regeneration/P1e_NH_nuebar_146.6z_401th_401w21.npy')

d_P2e_nh_nuebar_105 = np.load('regeneration/P2e_NH_nuebar_105.7z_401th_401w21.npy')
d_P2e_nh_nuebar_120_8 = np.load('regeneration/P2e_NH_nuebar_120.8z_401th_401w21.npy')
d_P2e_nh_nuebar_146_6 = np.load('regeneration/P2e_NH_nuebar_146.6z_401th_401w21.npy')

d_P1e_ih_nuebar_105 = np.load('regeneration/P1e_IH_nuebar_105.7z_401th_401w21.npy')
d_P1e_ih_nuebar_120_8 = np.load('regeneration/P1e_IH_nuebar_120.8z_401th_401w21.npy')
d_P1e_ih_nuebar_146_6 = np.load('regeneration/P1e_IH_nuebar_146.6z_401th_401w21.npy')

d_P2e_ih_nuebar_105 = np.load('regeneration/P2e_IH_nuebar_105.7z_401th_401w21.npy')
d_P2e_ih_nuebar_120_8 = np.load('regeneration/P2e_IH_nuebar_120.8z_401th_401w21.npy')
d_P2e_ih_nuebar_146_6 = np.load('regeneration/P2e_IH_nuebar_146.6z_401th_401w21.npy')

P1e_nh_nue_146_6 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_146_6)
P1e_nh_nue_120_8 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_120_8)
P1e_nh_nue_105 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nue_105)

P2e_nh_nue_146_6 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_146_6)
P2e_nh_nue_120_8 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_120_8)
P2e_nh_nue_105 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nue_105)

P1e_ih_nue_146_6 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_146_6)
P1e_ih_nue_120_8 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_120_8)
P1e_ih_nue_105 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nue_105)

P2e_ih_nue_146_6 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_146_6)
P2e_ih_nue_120_8 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_120_8)
P2e_ih_nue_105 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nue_105)

P1e_nh_nuebar_146_6 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_146_6)
P1e_nh_nuebar_120_8 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_120_8)
P1e_nh_nuebar_105 = RectBivariateSpline(w21_list, t12_list, d_P1e_nh_nuebar_105)

P2e_nh_nuebar_146_6 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_146_6)
P2e_nh_nuebar_120_8 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_120_8)
P2e_nh_nuebar_105 = RectBivariateSpline(w21_list, t12_list, d_P2e_nh_nuebar_105)

P1e_ih_nuebar_146_6 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_146_6)
P1e_ih_nuebar_120_8 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_120_8)
P1e_ih_nuebar_105 = RectBivariateSpline(w21_list, t12_list, d_P1e_ih_nuebar_105)

P2e_ih_nuebar_146_6 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_146_6)
P2e_ih_nuebar_120_8 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_120_8)
P2e_ih_nuebar_105 = RectBivariateSpline(w21_list, t12_list, d_P2e_ih_nuebar_105)

P1e_nh_nue_146_6_func = np.vectorize(P1e_nh_nue_146_6)
P1e_nh_nue_120_8_func = np.vectorize(P1e_nh_nue_120_8)
P1e_nh_nue_105_func = np.vectorize(P1e_nh_nue_105)

P2e_nh_nue_146_6_func = np.vectorize(P2e_nh_nue_146_6)
P2e_nh_nue_120_8_func = np.vectorize(P2e_nh_nue_120_8)
P2e_nh_nue_105_func = np.vectorize(P2e_nh_nue_105)

P1e_ih_nue_146_6_func = np.vectorize(P1e_ih_nue_146_6)
P1e_ih_nue_120_8_func = np.vectorize(P1e_ih_nue_120_8)
P1e_ih_nue_105_func = np.vectorize(P1e_ih_nue_105)

P2e_ih_nue_146_6_func = np.vectorize(P2e_ih_nue_146_6)
P2e_ih_nue_120_8_func = np.vectorize(P2e_ih_nue_120_8)
P2e_ih_nue_105_func = np.vectorize(P2e_ih_nue_105)

P1e_nh_nuebar_146_6_func = np.vectorize(P1e_nh_nuebar_146_6)
P1e_nh_nuebar_120_8_func = np.vectorize(P1e_nh_nuebar_120_8)
P1e_nh_nuebar_105_func = np.vectorize(P1e_nh_nuebar_105)

P2e_nh_nuebar_146_6_func = np.vectorize(P2e_nh_nuebar_146_6)
P2e_nh_nuebar_120_8_func = np.vectorize(P2e_nh_nuebar_120_8)
P2e_nh_nuebar_105_func = np.vectorize(P2e_nh_nuebar_105)

P1e_ih_nuebar_146_6_func = np.vectorize(P1e_ih_nuebar_146_6)
P1e_ih_nuebar_120_8_func = np.vectorize(P1e_ih_nuebar_120_8)
P1e_ih_nuebar_105_func = np.vectorize(P1e_ih_nuebar_105)

P2e_ih_nuebar_146_6_func = np.vectorize(P2e_ih_nuebar_146_6)
P2e_ih_nuebar_120_8_func = np.vectorize(P2e_ih_nuebar_120_8)
P2e_ih_nuebar_105_func = np.vectorize(P2e_ih_nuebar_105)