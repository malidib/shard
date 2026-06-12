#----------------------------------------------------------------------
#----------------------------------------------------------------------
#
# This file cointans all the constant values used in EFP_rebpund.py
#
#----------------------------------------------------------------------
#----------------------------------------------------------------------


RSUN = 4.6524726e-3            # [AU]
REAR = 4.26352e-5            # [AU]
MEAR = 3.003489616e-06        # Earth mass [MSUN]
MMAR = 3.227154546e-07        # Mars mass [MSUN]
MMOO = 3.6951390215e-8        # Moon mass [MSUN]
MCER = 4.7189547743e-10        # Ceres mass [MSUN]
MSUN = 1.98847e30            # Solar mass [kg]
AU = 1.495978707e11            # Astronomical Unit [m]
YEAR = 31556952.            # year [s]

#SPH resolution
SPHRES=2.e4

# SPHtable coding system
v0s=[1.,1.5,2.,3.,5.]
alphas=[0.,20.,40.,60.]
mtots=[2.*MCER,2.*MMOO,2.*MMAR,2.*MEAR]
gammas=[0.1,0.5,1.]
wfts=[10.,20.]
wfps=[10.,20.]
v0labels=['v1.0','v1.5','v2.0','v3.0','v5.0']
alphalabels=['a0','a20','a40','a60']
mtotlabels=['m21','m23','m24','m25']
gammalabels=['g0.1','g0.5','g1.0']
wftlabels=['wt10.0','wt20.0']
wfplabels=['wp10.0','wp20.0']
allpar=[v0s,alphas,mtots,gammas,wfts,wfps]
bases='544322'
