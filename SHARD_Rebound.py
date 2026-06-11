"""SHARD_Rebound.py

Collision-resolving REBOUND/mercurius driver for the SHARD paper.
Detected embryo--embryo impacts are mapped onto an SPH outcome catalogue;
the two largest remnants are interpolated from ``SPH.table`` and the small
fragment population is compressed with mass-weighted KMeans in velocity space.

Units follow the paper and REBOUND convention unless otherwise noted:
G=1, distance in AU, mass in solar masses, and time in yr/(2*pi).
Water fractions stored in particle parameter ``wf`` are percentages.
"""

from pathlib import Path
import glob
import sys
from random import random as rn

import numpy as np
import rebound
import reboundx
from sklearn.cluster import KMeans

from CONSTANTS import *

# ---------------------------------------------------------------------
# User-facing configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(".")
SCENARIO_NAME = "scenario_name"
SCENARIO_DIR = PROJECT_ROOT / SCENARIO_NAME

# Starting snapshot tag. Use "start" for the initial file or values such as
# "t1000000" for restart files named <scenario>.t1000000.
START_TAG = "start"

TOTAL_INTEGRATION_YEARS = 1.0e8
TIMESTEP_REBOUND = 1.0e-2 * 2.0 * np.pi  # 0.01 yr in REBOUND time units
N_CHECKPOINTS = 1000

N_GAS_GIANTS = 2

# Minimum resolved fragment mass in Earth masses. This sets both the survivor
# cutoff and N_fr=floor(M_fr/MFM), matching the manuscript default.
MIN_FRAGMENT_MASS_MEAR = 5.5e-4

# Debris above this mass remains in the active REBOUND population according to
# the manuscript's activation prescription.
ACTIVE_DEBRIS_MASS_MSUN = 3.0e-7

# Fraction of the SPH debris budget retained as resolved N-body debris.
# The removed fraction is treated as unresolved dust carrying the same mean
# debris water fraction. Set to 1.0 for strict resolved mass conservation.
DUST_MASS_RETENTION_FACTOR = 0.5
KMEANS_RANDOM_STATE = 0

MERCURIUS_HILLFAC = 5.0
IAS15_MIN_DT_FACTOR = 1.0e-4
REMOVAL_CHECK_FREQUENCY_STEPS = 1000

# Open-boundary removal criteria used in the manuscript diagnostics.
REMOVE_ECCENTRICITY_MIN = 1.0
REMOVE_A_MAX_AU = 15.0
REMOVE_A_MIN_AU = 0.1
REMOVE_Q_MIN_AU = 0.01

FALSE_COLLISION_RADIUS_FACTOR = 2.0
MIN_BACKTRACK_DISTANCE_HILL_RADII = 3.0

SPH_TABLE_PATH = PROJECT_ROOT / "SPH.table"
SPH_DEBRIS_DIR = PROJECT_ROOT / "SPHDebris_catalogue"

SAVE_CHECKPOINTS = True
NEW_SIM_WARNING = False
SAVE_PROGRESS = True
SAVE_COLLISION_OUTCOME = True

# Optional diagnostics. Defaults preserve the non-interactive production run.
PRINT_INTERPOLATION_POINTS = True
SHOW_DEBRIS_DIAGNOSTIC_PLOT = False
SHOW_COLLISION_GEOMETRY_PLOT = False
SHOW_FRAME_MAPPING_STEPS = False

# Backwards-compatible names used by the original procedural script.
t0 = START_TAG
Dt = TOTAL_INTEGRATION_YEARS
dt = TIMESTEP_REBOUND
Nsaves = N_CHECKPOINTS
scenario = str(SCENARIO_DIR)
NGG = N_GAS_GIANTS
MFM = MIN_FRAGMENT_MASS_MEAR

if not 0.0 <= DUST_MASS_RETENTION_FACTOR <= 1.0:
	raise ValueError("DUST_MASS_RETENTION_FACTOR must be between 0 and 1.")

save_checkpoints = SAVE_CHECKPOINTS
new_sim_warning = NEW_SIM_WARNING
save_progess = SAVE_PROGRESS
save_collision_outcome = SAVE_COLLISION_OUTCOME

#----------------------------------------------------------------------
#----------------------------------------------------------------------

#****************************   CLASSES   *****************************
		
class SPHcol:
	"""One row of SPH.table plus its associated debris catalogue."""
	def __init__(self,line):
		val=line.split()

		# collision parameters
		self.id=int(val[0])		# collision index (related to code)
		self.code=val[1]		# code with base 544322 for (v0,alpha,mtot,gamma,wt,wp)
		self.params=np.asarray(val[2:8],dtype=float) #(v0[vesc],alpha[deg],mtot[MSUN],gamma,wt,wp)
		
		# fragmented mass [mtot]
		self.mfr=float(val[9])
		
		# surviving bodies
		self.Nbig=int(val[8])	# number of surviving bodies
		
		# largest bodies
		if self.Nbig==-1: self.crashed=True
		else: self.crashed=False
		self.largest=[]

		for i in [10,18]:
			r=np.asarray(val[i:i+3],dtype=float)	# location wrt CoM in sph.coor. ([AU],[rad],[rad])
			v=np.asarray(val[i+3:i+6],dtype=float)	# velocity wrt CoM in sph.coor. ([AU/yr/2pi],[rad],[rad])
			m=float(val[i+6])						# mass [mtot]
			w=float(val[i+7])						# water [wtot]
			self.largest.append([r,v,m,w])
		
		# fragments
		if not self.crashed:
			try:
				str_id=str(self.id)
				while len(str_id)<3: str_id='0'+str_id
				debris_path = SPH_DEBRIS_DIR / '{}_{}.dat'.format(str_id,self.code)
				self.all=np.loadtxt(debris_path) # x,y,z,vx,vy,vz,m,m/mtot,wf [REBOUND units]
				if np.ndim(self.all)==1: self.all = np.asarray([self.all])
			except: pass
		
		# Perfect Merging
		if  self.largest[1][2]==-1. and not self.crashed: self.PM=True
		else: self.PM=False


#***************************   FUNCTIONS   ****************************

#-------------------
# Collisions solver
#-------------------

def collision_solver(sim_pointer, collision):
	sim = sim_pointer.contents
	
	# get colliding particles
	p1,p2=ps[collision.p1],ps[collision.p2]

	# collider labels
	collider_labels = [code_to_label(p1.params['code']),code_to_label(p2.params['code'])]
	
	# Collisions with the Sun are treated as perfect accretion sinks.
	if collision.p1==0:
		write_events_file(sim.t,'collision',p1,p2)
		ps[0].m+=p2.m
		write_events_file_outcome([ps[0]])
		return 2
	if collision.p2==0:
		write_events_file(sim.t,'collision',p1,p2)
		ps[0].m+=p1.m
		write_events_file_outcome([ps[0]])
		return 1

	# Gas giants are designated perfect-accretion sinks.
	if collision.p1<=NGG:
		write_events_file(sim.t,'collision',p1,p2)
		merge(p1,p2)
		write_events_file_outcome([p1])
		return 2
	if collision.p2<=NGG:
		write_events_file(sim.t,'collision',p1,p2)
		merge(p2,p1)
		write_events_file_outcome([p2])
		return 1
	
	# Test-particle debris collisions are resolved by perfect merging.
	if collision.p2>=sim.N_active:
		write_events_file(sim.t,'collision',p1,p2)
		merge(p1,p2)
		write_events_file_outcome([p1])
		return 2
	if collision.p1>=sim.N_active:
		write_events_file(sim.t,'collision',p1,p2)
		merge(p2,p1)
		write_events_file_outcome([p2])
		return 1	

	# Reject spurious direct-geometry contacts that are too far outside contact.
	xrel=np.asarray([p1.x-p2.x,p1.y-p2.y,p1.z-p2.z])
	if np.sqrt(xrel.dot(xrel))>FALSE_COLLISION_RADIUS_FACTOR*(p1.r+p2.r): return 0
	
	indeces=[collision.p1,collision.p2]	
	coll_p=get_coll_params(p1,p2)
	
	# save snapshot at the collision time and event file
	collision_snapshot(ps)
	write_events_file(sim.t,'collision',p1,p2)
	
	print('\n{}'.format('-'*80))
	print(' Collision detected between:  {} - {}\n'.format(collider_labels[0],collider_labels[1]))
	print('\nCollision parameters: ')
	print('  - v0 = {:.2f}'.format(coll_p[0]))
	print('  - alpha = {:.1f}'.format(coll_p[1]))
	print('  - mtot = {:.2f} [MEAR]'.format(coll_p[2]/MEAR))
	print('  - m1/m2 = {:.2f}'.format(coll_p[3]))
	print('  - wf1 = {:.4f} [%]'.format(coll_p[4]))
	print('  - wf2 = {:.4f} [%]'.format(coll_p[5]))
	
	# collision CoM position and velocity
	x1,v1,m1,th1,R1=np.asarray([p1.x,p1.y,p1.z]),np.asarray([p1.vx,p1.vy,p1.vz]),p1.m,p1.theta,p1.r
	x2,v2,m2,th2,R2=np.asarray([p2.x,p2.y,p2.z]),np.asarray([p2.vx,p2.vy,p2.vz]),p2.m,p2.theta,p2.r
	
	mtot=m1+m2
	xCoM=(x1*m1+x2*m2)/mtot
	vCoM=(v1*m1+v2*m2)/mtot
	rcol=np.sqrt(xCoM.dot(xCoM))
	thcol=np.arctan2(xCoM[1],xCoM[0])		# coll point projected angle on the x-y plane
	inccol=np.pi/2.-np.arccos(xCoM[2]/rcol)	# coll point "inclination"

	# check mass order
	p1_lt_p2=False
	if m1<m2: p1_lt_p2=True

	# interpolate SPH table and find the 2 largest bodies
	largest = interpolate_SPHtable(coll_p)
	
	# put the more massive one first
	if largest[0][2]<largest[1][2]: largest[0],largest[1] = largest[1],largest[0]

	# Classify macroscopic survivors from the two interpolated remnants.
	Nbig = get_Nbig(largest[0][2],largest[1][2],coll_p[3])
	for i in range(Nbig):
		if largest[i][2]*mtot<MFM*MEAR: Nbig=Nbig-1
	survivors = []
	for i in range(Nbig): survivors.append(largest[i])
	
	# Enforce survivor/debris water-budget limits before creating fragments.
	wtot_mtot=(coll_p[4]+coll_p[3]*coll_p[5])/(1.+coll_p[3])/100.
	if Nbig>0: survivors=water_conservation(survivors,wtot_mtot)
	
	# Residual mass and water budgets are assigned to the debris reservoir.
	mfr,mwfr = 1.,1.
	for surv in survivors:
		mfr-=surv[2]
		mwfr-=surv[3]
	
	mwtot=mtot*wtot_mtot
	print('\n Mass and Water check:')
	print('--- before collision ---')
	print(' TOT: m={:.4f} MEAR'.format(mtot/MEAR))
	print('      w={:.2e} MEAR (wf:{:.2f}%)'.format(mwtot/MEAR,100.*wtot_mtot))
	print(' - T: m={:.4f} MEAR ({:.2f}%)'.format(mtot/(coll_p[3]+1.)/MEAR,100./(coll_p[3]+1.)))
	print('      w={:.2e} MEAR ({:.2f}% from this body with wf:{:.3f}%)'.format(coll_p[4]*mtot*1./(coll_p[3]+1.)/MEAR/100,coll_p[4]*1./(coll_p[3]+1.)/wtot_mtot,coll_p[4]))
	print(' - P: m={:.5f} MEAR ({:.3f}%)'.format(mtot*coll_p[3]/(coll_p[3]+1.)/MEAR,100.*coll_p[3]/(coll_p[3]+1.)))
	print('      w={:.2e} MEAR ({:.3f}% from this body with wf:{:.3f}%)'.format(coll_p[5]*mtot*coll_p[3]/(coll_p[3]+1.)/MEAR/100,coll_p[5]*coll_p[3]/(coll_p[3]+1.)/wtot_mtot,coll_p[5]))
	print('--- after collision ---')
	for i in range(Nbig):
		print(' - S{}: m={:.4f} MEAR ({:.2f}%)'.format(i+1,mtot*survivors[i][2]/MEAR,100.*survivors[i][2]))
		print('       w={:.2e} MEAR ({:.2f}% - wf:{:.2f}%)'.format(survivors[i][3]*mwtot/MEAR,100.*survivors[i][3],survivors[i][3]*wtot_mtot*100/survivors[i][2]))
	print(' - fr: m={:.5f} MEAR ({:.3f}%)'.format(mtot*mfr/MEAR,100.*mfr))
	if mfr>0: print('       w={:.2e} MEAR ({:.3f}% - wf:{:.3f}%)'.format(mwfr*mwtot/MEAR,100.*mwfr,mwfr*wtot_mtot*100/mfr))
	print(' ')

	# Convert SPH spherical position/velocity components to Cartesian vectors.
	for i in range(Nbig):
		survivors[i][0]=from_sph_to_cart(survivors[i][0])
		survivors[i][1]=from_sph_to_cart(survivors[i][1])

	# Convert survivor/debris water masses to bulk water fractions in percent.
	for i in range(Nbig): survivors[i][3]=100.*survivors[i][3]*wtot_mtot/survivors[i][2]
	if mfr==0.: wffr=0.
	else: wffr=100.*mwfr*wtot_mtot/mfr
			
	# Convert remnant and debris masses from fractions of M_tot to solar masses.
	for i in range(Nbig): survivors[i][2]=survivors[i][2]*mtot
	mfr=mfr*mtot

	# Choose the number of debris clusters from the SPH debris budget, then
	# retain only the configured fraction as resolved N-body debris. Applying the
	# factor here, rather than to raw cluster weights, prevents later mass
	# normalization from algebraically cancelling the dust prescription.
	Nfr=int(mfr/(MFM*MEAR))
	mfr=mfr*DUST_MASS_RETENTION_FACTOR
	if mfr==0.:
		Nfr=0
		wffr=0.
	debris=[]
	
	if Nfr>0:
	
		# Interpolate neighbouring SPH debris catalogues and cluster in velocity space.
		debris = interpolate_SPHcatalogue(coll_p,Nbig,Nfr)

		# Normalize clustered debris masses to the retained debris budget M_fr.
		mtotdb=0.
		for deb in debris: mtotdb+=deb[2]
		for deb in debris: deb[2]=deb[2]*mfr/mtotdb
		
		# Scale clustered debris water fractions to conserve the debris water budget.
		adj_factor=0.
		for deb in debris: adj_factor+=deb[2]*deb[3]
		adj_factor=wffr*mfr/adj_factor
		for deb in debris: deb[3]=adj_factor*deb[3]

		debris_plot=SHOW_DEBRIS_DIAGNOSTIC_PLOT
		if debris_plot:	
	
			md=mfr/Nfr
			drad=get_radius(md,wffr)
			HR=rcol*np.power(mtot/3.,1./3)
			Vesc=np.sqrt(2.*mtot/get_radius(mtot,0.))
			vd=Vesc*1.05
			thds=np.linspace(0,2.*np.pi,num=Nfr,endpoint=False)+rn()*2.*np.pi
			debris_noSPH=[]
			for i in range(Nfr):
				versor=np.asarray([np.cos(thds[i]),np.sin(thds[i]),0.])
				debris_noSPH.append([versor*HR,versor*vd,md,wffr])
			
			if Nbig==1: survivors.append(survivors[-1])
				
			Rfr=(-survivors[0][0]*survivors[0][2]-survivors[1][0]*survivors[1][2])/mfr
			Vfr=(-survivors[0][1]*survivors[0][2]-survivors[1][1]*survivors[1][2])/mfr
	
			import matplotlib.pyplot as plt
			DT=1.e-2
			fig, axs = plt.subplots(2,figsize=(6,7),gridspec_kw={'height_ratios': [3,1]})
			
			corr=1e-1/REAR	# plotted radius is 10x the real one
			s1,s2,sfr=get_radius(survivors[0][2],0)*corr,get_radius(survivors[1][2],0)*corr,get_radius(mfr,0)*corr
			drad=get_radius(md,wffr)*corr
			axs[0].scatter(survivors[0][0][0],survivors[0][0][1],s=s1,alpha=0.5,c='r')
			axs[0].scatter(survivors[1][0][0],survivors[1][0][1],s=s2,alpha=0.5,c='r')
			axs[0].plot([survivors[0][0][0],survivors[0][0][0]+DT*survivors[0][1][0]],[survivors[0][0][1],survivors[0][0][1]+DT*survivors[0][1][1]],c='r',lw=2)
			axs[0].plot([survivors[1][0][0],survivors[1][0][0]+DT*survivors[1][1][0]],[survivors[1][0][1],survivors[1][0][1]+DT*survivors[1][1][1]],c='r',lw=2)
			axs[0].plot([0],[0],'.k',ms=3)
			for deb in debris_noSPH:
				axs[0].scatter(deb[0][0],deb[0][1],s=drad*corr,alpha=0.5,c='b')
				axs[0].plot([deb[0][0],deb[0][0]+DT*deb[1][0]],[deb[0][1],deb[0][1]+DT*deb[1][1]],c='b',lw=1)
			for deb in debris:
				axs[0].scatter(deb[0][0],deb[0][1],s=drad*corr,alpha=0.5,c='r')
				axs[0].plot([deb[0][0],deb[0][0]+DT*deb[1][0]],[deb[0][1],deb[0][1]+DT*deb[1][1]],c='r',lw=1)
			
			
			axs[1].scatter(survivors[0][0][0],survivors[0][0][2],s=s1,alpha=0.5,c='r')
			axs[1].scatter(survivors[1][0][0],survivors[1][0][2],s=s2,alpha=0.5,c='r')
			axs[1].plot([survivors[0][0][0],survivors[0][0][0]+DT*survivors[0][1][0]],[survivors[0][0][2],survivors[0][0][2]+DT*survivors[0][1][2]],c='r',lw=2)
			axs[1].plot([survivors[1][0][0],survivors[1][0][0]+DT*survivors[1][1][0]],[survivors[1][0][2],survivors[1][0][2]+DT*survivors[1][1][2]],c='r',lw=2)
			axs[1].plot([0],[0],'.k',ms=3)
			for deb in debris_noSPH:
				axs[1].scatter(deb[0][0],deb[0][2],s=drad*corr,alpha=0.5,c='b')
				axs[1].plot([deb[0][0],deb[0][0]+DT*deb[1][0]],[deb[0][2],deb[0][2]+DT*deb[1][2]],c='b',lw=1)
			for deb in debris:
				axs[1].scatter(deb[0][0],deb[0][2],s=drad*corr,alpha=0.5,c='r')
				axs[1].plot([deb[0][0],deb[0][0]+DT*deb[1][0]],[deb[0][2],deb[0][2]+DT*deb[1][2]],c='r',lw=1)
			
			axs[0].set_xlim(-1.e-2/2,1.e-2/2)
			axs[0].set_ylim(-1.e-2/2,1.e-2/2)
			axs[1].set_xlim(-1.e-2/2,1.e-2/2)
			axs[1].set_ylim(-1.e-2/3/2,1.e-2/3/2)
			axs[0].set_xlabel('x [AU]')
			axs[0].set_ylabel('y [AU]')
			axs[1].set_xlabel('x [AU]')
			axs[1].set_ylabel('z [AU]')
			plt.tight_layout()
			plt.show()
			if Nbig==1: survivors.pop(1)

	# Immediate survivor-survivor reaccretion test from the paper.
	if Nbig==2:
		drel=survivors[0][0]-survivors[1][0]
		vrel=survivors[0][1]-survivors[1][1]
		vrel2=vrel.dot(vrel)
		vesc2=2.*(survivors[0][2]+survivors[1][2])/np.sqrt(drel.dot(drel))
		if vrel2<vesc2:
			ms1=survivors[0][2]
			ms2=survivors[1][2]
			m=ms1+ms2
			x=(survivors[0][0]*ms1+survivors[1][0]*ms2)/m
			v=(survivors[0][1]*ms1+survivors[1][1]*ms2)/m
			wf=(survivors[0][3]*ms1+survivors[1][3]*ms2)/m
			Nbig=1
			survivors=[[x,v,m,wf]]
	
	if Nfr>0:
		# Immediate debris-survivor reaccretion test from the paper.
		reaccretion=True
		while reaccretion:
			reaccretion=False
			for i,surv in enumerate(survivors):
				for j,deb in enumerate(debris):
					drel=surv[0]-deb[0]
					vrel=surv[1]-deb[1]
					vrel2=vrel.dot(vrel)
					vesc2=2.*(surv[2]+deb[2])/np.sqrt(drel.dot(drel))	
					if vrel2<vesc2:
						reaccretion=True
						msur,mdeb=surv[2],deb[2]
						m=msur+mdeb
						x=(surv[0]*msur+deb[0]*mdeb)/m
						v=(surv[1]*msur+deb[1]*mdeb)/m
						wf=(surv[3]*msur+deb[3]*mdeb)/m
						debris.pop(j)
						survivors[i]=[x,v,m,wf]
						print('\n debris {} has been reaccreted by survivor {}\n'.format(j,i))
						break
		
		# Re-test survivors after debris accretion changes their masses and velocities.
		if Nbig==2 and Nfr!=len(debris):
			drel=survivors[0][0]-survivors[1][0]
			vrel=survivors[0][1]-survivors[1][1]
			vrel2=vrel.dot(vrel)
			vesc2=2.*(survivors[0][2]+survivors[1][2])/np.sqrt(drel.dot(drel))
			if vrel2<vesc2:
				ms1=survivors[0][2]
				ms2=survivors[1][2]
				m=ms1+ms2
				x=(survivors[0][0]*ms1+survivors[1][0]*ms2)/m
				v=(survivors[0][1]*ms1+survivors[1][1]*ms2)/m
				wf=(survivors[0][3]*ms1+survivors[1][3]*ms2)/m
				Nbig=1
				survivors=[[x,v,m,wf]]
				
				# Repeat debris reaccretion after the survivor-survivor merger.
				reaccretion=True
				while reaccretion:
					reaccretion=False
					for i,surv in enumerate(survivors):
						for j,deb in enumerate(debris):
							drel=surv[0]-deb[0]
							vrel=surv[1]-deb[1]
							vrel2=vrel.dot(vrel)
							vesc2=2.*(surv[2]+deb[2])/np.sqrt(drel.dot(drel))	
							if vrel2<vesc2:
								reaccretion=True
								msur,mdeb=surv[2],deb[2]
								m=msur+mdeb
								x=(surv[0]*msur+deb[0]*mdeb)/m
								v=(surv[1]*msur+deb[1]*mdeb)/m
								wf=(surv[3]*msur+deb[3]*mdeb)/m
								debris.pop(j)
								survivors[i]=[x,v,m,wf]
								print('\n debris {} has been reaccreted by survivor {}\n'.format(j,i))
								break

	plot=SHOW_COLLISION_GEOMETRY_PLOT
	if plot:	
		import matplotlib.pyplot as plt
		DT=1.e-3
		dg=0.2
		corr=3.e6
		fig = plt.figure(figsize=(10,10))
		ax = plt.axes(projection='3d')

		xl=0.
		for sur in survivors:
			r=get_radius(sur[2],sur[3])
			ax.scatter3D(sur[0][0],sur[0][1],sur[0][2],s=r*corr,alpha=0.1,c='b')
			ax.plot3D([sur[0][0],sur[0][0]+DT*sur[1][0]],[sur[0][1],sur[0][1]+DT*sur[1][1]],[sur[0][2],sur[0][2]+DT*sur[1][2]],c='b',lw=0.1)
			if abs(sur[0][0]+DT*sur[1][0])>xl: xl=abs(sur[0][0]+DT*sur[1][0])
			if abs(sur[0][1]+DT*sur[1][1])>xl: xl=abs(sur[0][1]+DT*sur[1][1])
			if abs(sur[0][2]+DT*sur[1][2])>xl: xl=abs(sur[0][2]+DT*sur[1][2])

		for deb in debris:
			r=get_radius(deb[2],deb[3])
			ax.scatter3D(deb[0][0],deb[0][1],deb[0][2],s=r*corr,alpha=0.1,c='r')
			ax.plot3D([deb[0][0],deb[0][0]+DT*deb[1][0]],[deb[0][1],deb[0][1]+DT*deb[1][1]],[deb[0][2],deb[0][2]+DT*deb[1][2]],c='r',lw=0.1)
			if abs(deb[0][0]+DT*deb[1][0])>xl: xl=abs(deb[0][0]+DT*deb[1][0])
			if abs(deb[0][1]+DT*deb[1][1])>xl: xl=abs(deb[0][1]+DT*deb[1][1])
			if abs(deb[0][2]+DT*deb[1][2])>xl: xl=abs(deb[0][2]+DT*deb[1][2])

	# Recentre the generated post-impact products before frame rotation.
	Nbig=len(survivors)
	Nfr=len(debris)
	xsCoM,vsCoM,msCoM=np.zeros(3),np.zeros(3),0.
	for surv in survivors:
		xsCoM+=surv[0]*surv[2]
		vsCoM+=surv[1]*surv[2]
		msCoM+=surv[2]
	for deb in debris:
		xsCoM+=deb[0]*deb[2]
		vsCoM+=deb[1]*deb[2]
		msCoM+=deb[2]
	if msCoM!=0.:
		xsCoM/=msCoM
		vsCoM/=msCoM
	for i in range(Nbig):
		survivors[i][0]=survivors[i][0]-xsCoM
		survivors[i][1]=survivors[i][1]-vsCoM
	for i in range(Nfr):
		debris[i][0]=debris[i][0]-xsCoM
		debris[i][1]=debris[i][1]-vsCoM
		
	if plot:
		ax.scatter3D(xsCoM[0],xsCoM[1],xsCoM[2],s=r*corr,alpha=0.1,c='k',marker='x')
		ax.plot3D([xsCoM[0],xsCoM[0]+DT*vsCoM[0]],[xsCoM[1],xsCoM[1]+DT*vsCoM[1]],[xsCoM[2],xsCoM[2]+DT*vsCoM[2]],c='k',lw=0.1)
		for sur in survivors:
			r=get_radius(sur[2],sur[3])
			ax.scatter3D(sur[0][0],sur[0][1],sur[0][2],s=r*corr,alpha=1,c='b')
			ax.plot3D([sur[0][0],sur[0][0]+DT*sur[1][0]],[sur[0][1],sur[0][1]+DT*sur[1][1]],[sur[0][2],sur[0][2]+DT*sur[1][2]],c='b',lw=1)
			if abs(sur[0][0]+DT*sur[1][0])>xl: xl=abs(sur[0][0]+DT*sur[1][0])
			if abs(sur[0][1]+DT*sur[1][1])>xl: xl=abs(sur[0][1]+DT*sur[1][1])
			if abs(sur[0][2]+DT*sur[1][2])>xl: xl=abs(sur[0][2]+DT*sur[1][2])

		for deb in debris:
			r=get_radius(deb[2],deb[3])
			ax.scatter3D(deb[0][0],deb[0][1],deb[0][2],s=r*corr,alpha=1,c='r')
			ax.plot3D([deb[0][0],deb[0][0]+DT*deb[1][0]],[deb[0][1],deb[0][1]+DT*deb[1][1]],[deb[0][2],deb[0][2]+DT*deb[1][2]],c='r',lw=1)
			if abs(deb[0][0]+DT*deb[1][0])>xl: xl=abs(deb[0][0]+DT*deb[1][0])
			if abs(deb[0][1]+DT*deb[1][1])>xl: xl=abs(deb[0][1]+DT*deb[1][1])
			if abs(deb[0][2]+DT*deb[1][2])>xl: xl=abs(deb[0][2]+DT*deb[1][2])
		
		xl=xl*1.2
		ax.set_xlim(-xl,xl)
		ax.set_ylim(-xl,xl)
		ax.set_zlim(-xl,xl)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('y [AU]')
		ax.set_zlabel('z [AU]')
		plt.tight_layout()
		plt.show()


	# Determine the deterministic rotation from the SPH collision frame to REBOUND.
	chi,psi,h=angle_SPH_Rebound([x1,v1,m1],[x2,v2,m2],R1,R2)

	# Rotate survivors into the global frame and overwrite the colliders.
	for i in range(Nbig):
		# Rotate the local product state and translate by the collision CoM.
		if h[2]>0:
			survivors[i][0]=Ry(survivors[i][0],np.pi)
			survivors[i][1]=Ry(survivors[i][1],np.pi)
		survivors[i][0]=Rz(Rx(survivors[i][0],np.pi/2.-chi),psi)+xCoM
		survivors[i][1]=Rz(Rx(survivors[i][1],np.pi/2.-chi),psi)+vCoM
		
		# Update mass, water fraction, and radius for the survivor.
		sim.particles[indeces[i]].x=survivors[i][0][0]
		sim.particles[indeces[i]].y=survivors[i][0][1]
		sim.particles[indeces[i]].z=survivors[i][0][2]
		sim.particles[indeces[i]].vx=survivors[i][1][0]
		sim.particles[indeces[i]].vy=survivors[i][1][1]
		sim.particles[indeces[i]].vz=survivors[i][1][2]
		sim.particles[indeces[i]].m=survivors[i][2]
		sim.particles[indeces[i]].params['wf']=survivors[i][3]
		sim.particles[indeces[i]].r=get_radius(survivors[i][2],survivors[i][3])
	
	# Work around REBOUND return-code handling for one-survivor outcomes.
	if Nbig==1:
		sim.particles[indeces[1]].x=sim.particles[indeces[0]].x
		sim.particles[indeces[1]].y=sim.particles[indeces[0]].y
		sim.particles[indeces[1]].z=sim.particles[indeces[0]].z
		sim.particles[indeces[1]].vx=sim.particles[indeces[0]].vx
		sim.particles[indeces[1]].vy=sim.particles[indeces[0]].vy
		sim.particles[indeces[1]].vz=sim.particles[indeces[0]].vz
		sim.particles[indeces[1]].m=sim.particles[indeces[0]].m
		sim.particles[indeces[1]].params['wf']=sim.particles[indeces[0]].params['wf']
		sim.particles[indeces[1]].r=sim.particles[indeces[0]].r	

	# Preserve the label of the more massive pre-impact body.
	if Nbig==1 and p1_lt_p2: sim.particles[indeces[0]].params['code']=sim.particles[indeces[1]].params['code']
		
	# Add clustered debris after survivors.  The activation threshold is kept as
	# an explicit reproducibility knob because it controls which debris bodies
	# mutually gravitate in REBOUND.
	Nps=sim.N
	lab_deb=code_to_label(ps[-1].params['code'])
	last_deb=-1
	debris_labels=[]
	if lab_deb[0]=="D": last_deb=int(lab_deb[1:])

	if Nfr>0:
		debris = sorted(debris, key=lambda deb: deb[2])
		number_of_lines = sum(deb[2] >= ACTIVE_DEBRIS_MASS_MSUN for deb in debris)
		# Keep the paper's stated activation policy; see review notes on REBOUND
		# N_active indexing before changing this to include pre-existing bodies.
		sim.N_active=number_of_lines

	for i in range(Nfr):
		# Rotate the local product state and translate by the collision CoM.
		if h[2]>0:
			debris[i][0]=Ry(debris[i][0],np.pi)
			debris[i][1]=Ry(debris[i][1],np.pi)
		debris[i][0]=Rz(Rx(debris[i][0],np.pi/2.-chi),psi)+xCoM
		debris[i][1]=Rz(Rx(debris[i][1],np.pi/2.-chi),psi)+vCoM
			
		# Add the debris body and assign its passive water-fraction parameter.
		sim.add(m=debris[i][2],r=get_radius(debris[i][2],debris[i][3]),x=debris[i][0][0],y=debris[i][0][1],z=debris[i][0][2],vx=debris[i][1][0],vy=debris[i][1][1],vz=debris[i][1][2])
		sim.particles[i+Nps].params['wf']=debris[i][3]

		debris_lab = "D"+str(i+1+last_deb)
		sim.particles[i+Nps].params['code']=label_to_code(debris_lab)
		debris_labels.append(debris_lab)

	print('\n Post-collisional objects')
	print('\tm [MEAR]  wf [%]  a [AU]  e [1]   inc [deg]')
	for i in range(Nbig):
		spi=sim.particles[indeces[i]]
		print('sur-{}\t {:.4f}   {:.4f}  {:.4f}  {:.4f}   {:.4f}'.format(i+1,spi.m/MEAR,spi.params['wf'],spi.a,spi.e,spi.inc*180/np.pi))
	for i in range(Nfr):
		spi=sim.particles[sim.N-Nfr+i]
		print('deb-{}\t {:.4f}   {:.4f}  {:.4f}  {:.4f}   {:.4f}'.format(i+1,spi.m/MEAR,spi.params['wf'],spi.a,spi.e,spi.inc*180/np.pi))
	print()
	
	# Persist the collision record and human-readable outcome logs.
	save_collision(coll_p,xCoM,vCoM,chi,psi,h,survivors,mfr,wffr,Nfr,collider_labels,debris_labels)

	# save outcome in events file
	outcome = []
	for i in range(Nbig): outcome.append(sim.particles[indeces[i]])
	for i in range(Nfr): outcome.append(sim.particles[sim.N-Nfr+i])
	write_events_file_outcome(outcome)
	
	# print collision event in the progress file
	if save_collision_outcome: save_col_out(sim.t,Nbig,Nfr,mfr,wffr)
	
	if Nbig==0: return 3
	if Nbig==1: return 2
	if Nbig==2: return 0

def merge(pa,pb):
	"""Merge pb into pa, conserving mass, linear momentum, and water mass."""
	ma,mb=pa.m,pb.m
	mab=ma+mb
	try:
		wf=(pa.params['wf']*ma+pb.params['wf']*mb)/mab
	except Exception:
		wf=None
	pa.x=(ma*pa.x+mb*pb.x)/mab
	pa.y=(ma*pa.y+mb*pb.y)/mab
	pa.z=(ma*pa.z+mb*pb.z)/mab
	pa.vx=(ma*pa.vx+mb*pb.vx)/mab
	pa.vy=(ma*pa.vy+mb*pb.vy)/mab
	pa.vz=(ma*pa.vz+mb*pb.vz)/mab
	pa.m=mab
	if wf is not None:
		pa.params['wf']=wf
		try:
			primary_label=code_to_label(pa.params['code'])
		except Exception:
			primary_label=None
		if primary_label not in ("SUN","JUP","SAT"):
			pa.r=get_radius(pa.m,pa.params['wf'])


def water_conservation(survs,wtot_mtot):
	# Dry impacts carry no water budget to conserve.
	if wtot_mtot==0.:
		for i in range(len(survs)): survs[i][3]=0.
		return survs
	# Survivor water mass cannot exceed survivor mass.
	mfr,wfr=1.,1.
	for i in range(len(survs)):
		if survs[i][3]>survs[i][2]/wtot_mtot: survs[i][3]=survs[i][2]/wtot_mtot
		mfr=mfr-survs[i][2]
		wfr=wfr-survs[i][3]
	# Any remaining debris water mass cannot exceed the debris mass.
	if wfr>mfr/wtot_mtot:
		C=(1.-mfr/wtot_mtot)/(1.-wfr)
		for i in range(len(survs)): survs[i][3]=C*survs[i][3]
	return survs

def get_coll_params(p1,p2):
	m1,m2=p1.m,p2.m
	xrel=np.asarray([p1.x-p2.x,p1.y-p2.y,p1.z-p2.z])
	vrel=np.asarray([p1.vx-p2.vx,p1.vy-p2.vy,p1.vz-p2.vz])
	xmod=np.sqrt(xrel.dot(xrel))
	vmod=np.sqrt(vrel.dot(vrel))
	
	mtot=m1+m2
	gamma=min(m1/m2,m2/m1)
	vesc=np.sqrt(2.*mtot/xmod)
	v0=vmod/vesc
	alpha=np.arccos(abs(np.dot(xrel/xmod,vrel/vmod)))*180/np.pi #[deg]
	if m1>m2: wft,wfp=p1.params['wf'],p2.params['wf']
	else: wft,wfp=p2.params['wf'],p1.params['wf']

	return v0,alpha,mtot,gamma,wft,wfp

def get_Nbig(m1,m2,g):
	tar=1/(1.+g)
	pro=g*tar
	# CC
	if m1<tar*0.1: return 0		# Catastrophic Collision
	elif m2<pro*0.1: return 1	# Projectile accretion or destruction
	else: return 2				# Hit-and-run
	
def from_sph_to_cart(vc):
	return np.asarray([vc[0]*np.cos(vc[1])*np.sin(vc[2]),vc[0]*np.sin(vc[1])*np.sin(vc[2]),vc[0]*np.cos(vc[2])])

def Rx(v,t):
	ct,st=np.cos(t),np.sin(t)
	return np.asarray([v[0],ct*v[1]-st*v[2],st*v[1]-ct*v[2]])

def Ry(v,t):
	ct,st=np.cos(t),np.sin(t)
	return np.asarray([ct*v[0]+st*v[2],v[1],-st*v[0]+ct*v[2]])
	
def Rz(v,t):
	ct,st=np.cos(t),np.sin(t)
	return np.asarray([ct*v[0]-st*v[1],st*v[0]+ct*v[1],v[2]])

def angle_SPH_Rebound(l1,l2,R1,R2):
	
	all_steps_plot=SHOW_FRAME_MAPPING_STEPS
	if all_steps_plot:
		import matplotlib.pyplot as plt
		fig, axs = plt.subplots(3,4,figsize=[16,9])
		dt=1.e-3
		c=['r','b']
		R=[R1,R2]
		ax=axs[0,0]
		for i,l in enumerate([l1,l2]):
			ax.plot(l[0][0],l[0][1],'.'+c[i])
			ax.plot([l[0][0],l[0][0]+l[1][0]*dt],[l[0][1],l[0][1]+l[1][1]*dt],c[i])
			circle = plt.Circle((l[0][0],l[0][1]), R[i], color=c[i],alpha=0.5)
			ax.add_patch(circle)
		ax.set_xlim(-0.3425,-0.341)
		ax.set_ylim(0.9542,0.9557)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('y [AU]')
		ax=axs[0,1]
		for i,l in enumerate([l1,l2]):
			ax.plot(l[0][0],l[0][2],'.'+c[i])
			ax.plot([l[0][0],l[0][0]+l[1][0]*dt],[l[0][2],l[0][2]+l[1][2]*dt],c[i])
			circle = plt.Circle((l[0][0],l[0][2]), R[i], color=c[i],alpha=0.5)
			ax.add_patch(circle)
		ax.set_xlim(-0.3425,-0.341)
		ax.set_ylim(-0.000795,0.000605)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('z [AU]')
		
		print(np.sqrt(l1[1].dot(l1[1])),c[0])	
		print(np.sqrt(l2[1].dot(l2[1])),c[1])
		print('v',l1[1],c[0])	
		print('v',l2[1],c[1])	

	# move to SoC of smaller body
	if l1[2]>l2[2]:
		r0=l1[0]-l2[0]
		v0=l1[1]-l2[1]
		c=['b','r']
		R=[R2,R1]
	else:
		r0=l2[0]-l1[0]
		v0=l2[1]-l1[1]	
	
	if all_steps_plot:
		ax=axs[0,2]
		ax.plot(0,0,'.'+c[0])
		ax.plot(r0[0],r0[1],'.'+c[1])
		ax.plot([r0[0],r0[0]+v0[0]*dt],[r0[1],r0[1]+v0[1]*dt],c[1])
		circle = plt.Circle((0,0), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((r0[0],r0[1]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.0003,0.0003)
		ax.set_ylim(-0.00045,0.00015)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('y [AU]')
		ax=axs[0,3]
		ax.plot(0,0,'.'+c[0])
		ax.plot(r0[0],r0[2],'.'+c[1])
		ax.plot([r0[0],r0[0]+v0[0]*dt],[r0[2],r0[2]+v0[2]*dt],c[1])
		circle = plt.Circle((0,0), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((r0[0],r0[2]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.0003,0.0003)
		ax.set_ylim(-0.00045,0.00015)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('z [AU]')
		
		#3D
		#ax = plt.axes(projection='3d')
		#ax.scatter3D(0,0,0,c=c[0])
		#ax.scatter3D(r0[0],r0[1],r0[2],c=c[1])
		#u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
		#x = np.cos(u)*np.sin(v)*R1
		#y = np.sin(u)*np.sin(v)*R1
		#z = np.cos(v)*R1
		#ax.plot_wireframe(x, y, z, color=c[0])
		#u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
		#x = np.cos(u)*np.sin(v)*R2+r0[0]
		#y = np.sin(u)*np.sin(v)*R2+r0[1]
		#z = np.cos(v)*R2+r0[2]
		#ax.plot_wireframe(x, y, z, color=c[1])
		#ax.plot3D([r0[0],r0[0]+v0[0]*dt],[r0[1],r0[1]+v0[1]*dt],[r0[2],r0[2]+v0[2]*dt],c=c[1])
		#ax.set_xlim(-0.0003,0.0003)
		#ax.set_ylim(-0.00045,0.00015)
		#ax.set_zlim(-0.00045,0.00015)
		#plt.show()
	
	# rotate the system so that h=[0,0,+h]
	h=np.cross(r0,v0)
	phi_norm2=h[0]*h[0]+h[1]*h[1]
	if phi_norm2!=0: phi=np.arccos(h[0]/np.sqrt(phi_norm2))
	else: phi=0.
	th=np.arccos(h[2]/np.sqrt(h.dot(h)))
	
	r0=Ry(Rz(r0,-phi),-th)
	v0=Ry(Rz(v0,-phi),-th)

	if all_steps_plot:
		ax=axs[1,0]
		ax.plot(0,0,'.'+c[0])
		ax.plot(r0[0],r0[1],'.'+c[1])
		ax.plot([r0[0],r0[0]+v0[0]*dt],[r0[1],r0[1]+v0[1]*dt],c[1])
		circle = plt.Circle((0,0), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((r0[0],r0[1]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.0003,0.0003)
		ax.set_ylim(-0.00045,0.00015)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('y [AU]')


	# orbital elements
	r0_mod=np.sqrt(r0.dot(r0))
	v0_mod=np.sqrt(v0.dot(v0))
	mtot=l1[2]+l2[2]
	a=1./(v0_mod*v0_mod/mtot-2./r0_mod)
	e=np.sqrt(1.+h.dot(h)/mtot/a)
	f0=-np.arccos(((a*(e*e-1)/r0_mod)-1.)/e)
	
	# rotating so that f=0 for y=0 (omega=0)
	th0=np.arctan2(r0[1],r0[0])
	omega=th0-f0
	R0=Rz(r0,-omega)
	V0=Rz(v0,-omega)

	if all_steps_plot:
		ax=axs[1,0]
		ax.plot([-np.cos(omega),np.cos(omega)],[-np.sin(omega),np.sin(omega)],ls='dotted',c='k')
		ax=axs[1,1]
		ax.plot(0,0,'.'+c[0])
		ax.plot(R0[0],R0[1],'.'+c[1])
		ax.plot([R0[0],R0[0]+V0[0]*dt],[R0[1],R0[1]+V0[1]*dt],c[1])
		circle = plt.Circle((0,0), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((R0[0],R0[1]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.0003,0.0003)
		ax.set_ylim(-0.00015,0.00045)
		ax.plot([-1,1],[0,0],ls='dotted',c='k')
	
	
	# back-tracing the colliding bodies to d=5(R1+R2) and get the velocity angle wrt x-axis (delta)
	rmin=5.*(R1+R2)
	fd=-np.arccos(((a/rmin)*(e*e-1.)-1.)/e)
	cfd,sfd=np.cos(fd),np.sin(fd)
	phid=np.arctan2(1.+e*cfd,e*sfd)

	# getting the y_versor of the SPH frame (velocity versor of the bigger body)
	SPHy=np.asarray([np.cos(fd+phid),np.sin(fd+phid),0.])
	SPHy=Rz(Ry(Rz(SPHy,omega),th),phi)
	chi=np.arccos(SPHy[2])
	psi=np.arctan2(-SPHy[0],SPHy[1])

	if all_steps_plot:
		Rd=np.asarray([cfd,sfd,0.])*rmin
		Vd=np.asarray([np.cos(fd+phid),np.sin(fd+phid),0.])*np.sqrt(mtot*((2./rmin)+(1./a)))
		ax=axs[1,2]
		ax.plot(0,0,'.'+c[0])
		ax.plot(R0[0],R0[1],'.'+c[1])
		ax.plot([R0[0],R0[0]+V0[0]*dt],[R0[1],R0[1]+V0[1]*dt],c[1],ls='dotted')
		circle = plt.Circle((0,0), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((R0[0],R0[1]), R[1], color=c[1],alpha=0.1)
		ax.add_patch(circle)
		ax.plot(Rd[0],Rd[1],'.'+c[1])
		ax.plot([Rd[0],Rd[0]+Vd[0]*dt],[Rd[1],Rd[1]+Vd[1]*dt],c[1])
		circle = plt.Circle((Rd[0],Rd[1]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.0005,0.0005)
		ax.set_ylim(-0.0005,0.0005)
		ax.plot([-1,1],[0,0],ls='dotted',c='k')
		flim=np.arccos(-1./e)
		fs=np.linspace(-flim+1e-5,flim,1000)
		rs=a*(e*e-1.)/(1.+e*np.cos(fs))
		ax.plot(np.cos(fs)*rs,np.sin(fs)*rs,ls='--',c='k',lw=1)

					
		ax=axs[1,3]
		ax.plot(0,0,'.'+c[0])
		R0=Rz(R0,omega)
		V0=Rz(V0,omega)
		Rd=Rz(Rd,omega)
		Vd=Rz(Vd,omega)
		ax.plot(R0[0],R0[1],'.'+c[1])
		ax.plot([R0[0],R0[0]+V0[0]*dt],[R0[1],R0[1]+V0[1]*dt],c[1],ls='dotted')
		circle = plt.Circle((0,0), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((R0[0],R0[1]), R[1], color=c[1],alpha=0.1)
		ax.add_patch(circle)
		ax.plot(Rd[0],Rd[1],'.'+c[1])
		ax.plot([Rd[0],Rd[0]+Vd[0]*dt],[Rd[1],Rd[1]+Vd[1]*dt],c[1])
		circle = plt.Circle((Rd[0],Rd[1]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.0005,0.0005)
		ax.set_ylim(-0.0005,0.0005)
		ax.plot([-np.cos(omega),np.cos(omega)],[-np.sin(omega),np.sin(omega)],ls='dotted',c='k')
		flim=np.arccos(-1./e)
		fs=np.linspace(-flim+1e-5,flim,1000)
		rs=a*(e*e-1.)/(1.+e*np.cos(fs))
		ax.plot(np.cos(fs+omega)*rs,np.sin(fs+omega)*rs,ls='--',c='k',lw=1)	
		
	
		R0=Rz(Ry(R0,th),phi)
		V0=Rz(Ry(V0,th),phi)
		Rd=Rz(Ry(Rd,th),phi)
		Vd=Rz(Ry(Vd,th),phi)
		ax=axs[2,0]
		ax.plot(0,0,'.'+c[0])
		ax.plot(R0[0],R0[1],'.'+c[1])
		ax.plot([R0[0],R0[0]+V0[0]*dt],[R0[1],R0[1]+V0[1]*dt],c[1],ls='dotted')
		circle = plt.Circle((0,0), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((R0[0],R0[1]), R[1], color=c[1],alpha=0.1)
		ax.add_patch(circle)
		ax.plot(Rd[0],Rd[1],'.'+c[1])
		ax.plot([Rd[0],Rd[0]+Vd[0]*dt],[Rd[1],Rd[1]+Vd[1]*dt],c[1])
		circle = plt.Circle((Rd[0],Rd[1]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.0005,0.0005)
		ax.set_ylim(-0.0005,0.0005)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('y [AU]')
		
		ax=axs[2,1]
		ax.plot(0,0,'.'+c[0])
		ax.plot(R0[0],R0[2],'.'+c[1])
		ax.plot([R0[0],R0[0]+V0[0]*dt],[R0[2],R0[2]+V0[2]*dt],c[1],ls='dotted')
		circle = plt.Circle((0,0), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((R0[0],R0[2]), R[1], color=c[1],alpha=0.1)
		ax.add_patch(circle)
		ax.plot(Rd[0],Rd[2],'.'+c[1])
		ax.plot([Rd[0],Rd[0]+Vd[0]*dt],[Rd[2],Rd[2]+Vd[2]*dt],c[1])
		circle = plt.Circle((Rd[0],Rd[2]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.0005,0.0005)
		ax.set_ylim(-0.0005,0.0005)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('z [AU]')
		
		mtot=l1[2]+l2[2]
		xCoM=(l1[0]*l1[2]+l2[0]*l2[2])/mtot
		vCoM=(l1[1]*l1[2]+l2[1]*l2[2])/mtot
		dX=np.asarray([1.1e-3,1e-4,0])
		if l1[2]>l2[2]:
			x1=Rd*l2[2]/mtot+xCoM+dX
			v1=Vd*l2[2]/mtot+vCoM
			x2=-Rd*l1[2]/mtot+xCoM+dX
			v2=-Vd*l1[2]/mtot+vCoM
			c=['r','b']
			R=[R1,R2]
		else:
			x1=-Rd*l1[2]/mtot+xCoM+dX
			v1=-Vd*l1[2]/mtot+vCoM
			x2=Rd*l2[2]/mtot+xCoM+dX
			v2=Vd*l2[2]/mtot+vCoM

		ax=axs[2,2]
		ax.plot(l1[0][0],l1[0][1],'.'+c[0])
		ax.plot(l2[0][0],l2[0][1],'.'+c[1])
		circle = plt.Circle((l1[0][0],l1[0][1]), R1, color=c[0],alpha=0.1)
		ax.add_patch(circle)
		circle = plt.Circle((l2[0][0],l2[0][1]), R2, color=c[1],alpha=0.1)
		ax.add_patch(circle)
		ax.plot(x1[0],x1[1],'.'+c[0])
		ax.plot(x2[0],x2[1],'.'+c[1])
		ax.plot([x1[0],x1[0]+v1[0]*dt],[x1[1],x1[1]+v1[1]*dt],c[0])
		ax.plot([x2[0],x2[0]+v2[0]*dt],[x2[1],x2[1]+v2[1]*dt],c[1])
		circle = plt.Circle((x1[0],x1[1]), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((x2[0],x2[1]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.3415,-0.34)
		ax.set_ylim(0.9542,0.9557)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('y [AU]')
		
		ax=axs[2,3]
		ax.plot(l1[0][0],l1[0][2],'.'+c[0])
		ax.plot(l2[0][0],l2[0][2],'.'+c[1])
		circle = plt.Circle((l1[0][0],l1[0][2]), R1, color=c[0],alpha=0.1)
		ax.add_patch(circle)
		circle = plt.Circle((l2[0][0],l2[0][2]), R2, color=c[1],alpha=0.1)
		ax.add_patch(circle)
		ax.plot(x1[0],x1[2],'.'+c[0])
		ax.plot(x2[0],x2[2],'.'+c[1])
		ax.plot([x1[0],x1[0]+v1[0]*dt],[x1[2],x1[2]+v1[2]*dt],c[0])
		ax.plot([x2[0],x2[0]+v2[0]*dt],[x2[2],x2[2]+v2[2]*dt],c[1])
		circle = plt.Circle((x1[0],x1[2]), R[0], color=c[0],alpha=0.5)
		ax.add_patch(circle)
		circle = plt.Circle((x2[0],x2[2]), R[1], color=c[1],alpha=0.5)
		ax.add_patch(circle)
		ax.set_xlim(-0.3415,-0.34)
		ax.set_ylim(-0.000795,0.000605)
		ax.set_xlabel('x [AU]')
		ax.set_ylabel('z [AU]')
		
		print(np.sqrt(v1.dot(v1)),c[0])	
		print(np.sqrt(v2.dot(v2)),c[1])	
		print('v',v1,c[0])	
		print('v',v2,c[1])	
				

		print('\na = ',a,' [AU]')			 	   
		print('e = ',e)
		print('f_0 = ',f0/np.pi,' [pi]')
		print('theta_0 = ',th0/np.pi,' [pi]')
		print('omega = ',omega/np.pi,' [pi]')
		print('\nSPHy versor: ',SPHy)
		print('  chi: {:.3f} pi'.format(chi/np.pi))
		print('  psi: {:.3f} pi'.format(psi/np.pi))
		plt.tight_layout()	
		plt.show()
	
	return chi,psi,h


#-------------------
# SPH table interpolator
#-------------------

def interpolate_SPHtable(params):
	
	# Select the two neighbouring SPH grid nodes and the interpolation weight per axis.
	ind_w=[get_interppoint_indices_and_weight(i,x) for i,x in enumerate(params)]
	
	if PRINT_INTERPOLATION_POINTS:
		for i,x in enumerate(ind_w): print(params[i],'  in  ',allpar[i],'  ->  ',x)
	
	# Gather the 2^6 neighbouring SPH table payloads.
	points=[]
	for p1 in ind_w[0][0]:
		for p2 in ind_w[1][0]:
			for p3 in ind_w[2][0]:
				for p4 in ind_w[3][0]:
					for p5 in ind_w[4][0]:
						for p6 in ind_w[5][0]:
							code='{}{}{}{}{}{}'.format(p1,p2,p3,p4,p5,p6)
							nr=get_index_from_code(code)-1
							if SPHcat[nr].code!=code:
								print('\n>>> WRONG SPH collision selected! <<<\n')
								sys.exit()
							points.append([SPHcat[nr].largest,SPHcat[nr].code])
	
	for j in range(6): points=[interpolate(points[i*2],points[i*2+1],ind_w[5-j][1]) for i in range(int(len(points)/2))]
	return points[0][0]
	
def get_interppoint_indices_and_weight(i,x,extrapol=True):
	# Use logarithmic interpolation along M_tot (axis i=2), as in the paper.
	if i==2: log=True
	else: log=False
	
	for j,xj in enumerate(allpar[i]):
		if x<xj:
			if j==0: ind=[0,1]
			else: ind=[j-1,j]
			break
		ind=[len(allpar[i])-2,len(allpar[i])-1]
	dx=weight(x,allpar[i][ind[0]],allpar[i][ind[1]],log=log)
	if not extrapol:
		if dx<0.: dx=0.
		elif dx>1.: dx=1.
	return [ind,dx]

def weight(x,xa,xb,log=False):
	if log: return np.log10(x/xa)/np.log10(xb/xa)
	else: return (x-xa)/(xb-xa)
	
def interpolate(pa,pb,dx):
	# Adjacent binary interpolations must combine matching mixed-radix prefixes.
	if pa[1][:-1]!=pb[1][:-1]:
		print('\n>>> WRONG SPH collision coupling! <<<\n')
		sys.exit()
	
	# --- 1st largest ---
	ya,yb=pa[0][0],pb[0][0]
	
	# Skip crashed endpoints by returning the usable neighbouring payload.
	if ya[2]==-1.: return pb[0],pa[1][:-1]
	if yb[2]==-1.: return pa[0],pa[1][:-1]
	
	# Interpolate the first remnant in radius/speed and circular angular coordinates.
	r=[lin_interpol(ya[0][0],yb[0][0],dx),interpol_angle(ya[0][1],yb[0][1],dx),interpol_angle(ya[0][2],yb[0][2],dx)]
	v=[lin_interpol(ya[1][0],yb[1][0],dx),interpol_angle(ya[1][1],yb[1][1],dx),interpol_angle(ya[1][2],yb[1][2],dx)]
	m=lin_interpol(ya[2],yb[2],dx)
	gwf=lin_interpol(ya[3],yb[3],dx)
	largest=[[r,v,m,gwf]]
	
	# --- 2nd largest ---
	ya,yb=pa[0][1],pb[0][1]
	
	# Interpolate consistently when the lower endpoint is perfect merging.
	if ya[2]==-1.:								# pa: PM
		if yb[2]==-1.: largest.append(ya)		#pa & pb PM	
		else:
			if dx<=0.: largest.append(ya)		#extrapolation->PM
			elif dx>=1.: largest.append(yb)		#extrapolation->yb
			else:								#interpolate with a PM
				r=[lin_interpol(0.,yb[0][0],dx),yb[0][1],yb[0][2]]
				v=[lin_interpol(0.,yb[1][0],dx),yb[1][1],yb[1][2]]
				m=lin_interpol(0.,yb[2],dx)
				gwf=yb[3]
				largest.append([r,v,m,gwf])
		adjust_result(largest)
		return largest,pa[1][:-1]
	
	# Interpolate consistently when the upper endpoint is perfect merging.
	if yb[2]==-1.:								# pb: PM
		if dx<=0.: largest.append(ya)			#extrapolation->ya
		elif dx>=1.: largest.append(yb)			#extrapolation->PM
		else:									#interpolate with a PM
			r=[lin_interpol(ya[0][0],0.,dx),ya[0][1],ya[0][2]]
			v=[lin_interpol(ya[1][0],0.,dx),ya[1][1],ya[1][2]]
			m=lin_interpol(ya[2],0.,dx)
			gwf=ya[3]
			largest.append([r,v,m,gwf])
		adjust_result(largest)
		return largest,pa[1][:-1]
	
	# Interpolate the second remnant when both endpoints contain it.
	r=[lin_interpol(ya[0][0],yb[0][0],dx),interpol_angle(ya[0][1],yb[0][1],dx),interpol_angle(ya[0][2],yb[0][2],dx)]
	v=[lin_interpol(ya[1][0],yb[1][0],dx),interpol_angle(ya[1][1],yb[1][1],dx),interpol_angle(ya[1][2],yb[1][2],dx)]
	m=lin_interpol(ya[2],yb[2],dx)
	gwf=lin_interpol(ya[3],yb[3],dx)
	largest.append([r,v,m,gwf])
	adjust_result(largest)
	return largest,pa[1][:-1]
	
def lin_interpol(a,b,dx,log=False):
	if log: return np.power(10.,lin_interpol(np.log10(a),np.log10(b),dx))
	return a*(1.-dx)+b*dx

def interpol_angle(a,b,dx):
	return np.arctan2(lin_interpol(np.sin(a),np.sin(b),dx),lin_interpol(np.cos(a),np.cos(b),dx))

def adjust_result(largest):
	# Keep remnant ordering and physical bounds after interpolation/extrapolation.
	if largest[0][2]<largest[1][2]: largest=[largest[1],largest[0]]

	if largest[0][2]<0:
		largest[0][2]=1./SPHRES
		largest[1][2]=1./SPHRES
	
	if largest[1][2]<0:
		if largest[0][2]>=1. or largest[1][2]==-1.:
			largest[0][2]=1.
			largest=[largest[0],[[-1.,-1.,-1.],[-1.,-1.,-1.],-1.,-1.]]
		else: largest[1][2]=1./SPHRES

	if largest[0][2]+largest[1][2]>1.:
		corr=largest[0][2]+largest[1][2]
		largest[0][2],largest[1][2]=largest[0][2]/corr,largest[1][2]/corr
	
	if largest[1][2]==-1. and largest[0][2]!=1.: largest[0][2]=1.

	for i in range(2):
		for j in range(2):
			if largest[i][j][0]<0.: largest[i][j][0]=0.
			
	if largest[0][2]==1.: largest[0][3]=1.
	else:
		for i in range(2):
			if largest[i][3]<0: largest[i][3]=0.
		if largest[0][3]+largest[1][3]>1.:
			corr=largest[0][3]+largest[1][3]
			largest[0][3],largest[1][3]=largest[0][3]/corr,largest[1][3]/corr

def get_radius(m,wf):
	# Radius-mass relation used in the paper; water-fraction dependence is neglected.
	C=1.008*REAR
	S=0.279
	R=C*np.power(m/MEAR,S)
	return R	# radius [AU]


#-------------------
# SPH Debris catalogue interpolator
#-------------------


def interpolate_SPHcatalogue(params,Nbig,Ncl):

	plot=SHOW_DEBRIS_DIAGNOSTIC_PLOT
	if plot:				
		import matplotlib.pyplot as plt
		from mpl_toolkits import mplot3d
		import pylab	
		
		Ncolors=64		
		cm = pylab.get_cmap('gist_rainbow')
		col=[cm(1.*i/Ncolors) for i in range(Ncolors)]
		ci=-1
    	
		fig = plt.figure(figsize=(10,10))
		ax = plt.axes(projection='3d')
		ax.set_xlabel('vx [AU/yr/2$\\pi$]')
		ax.set_ylabel('vy [AU/yr/2$\\pi$]')
		ax.set_zlabel('vz [AU/yr/2$\\pi$]')   



	# Select the two neighbouring SPH grid nodes and the interpolation weight per axis.
	ind_w=[get_interppoint_indices_and_weight(i,x,extrapol=False) for i,x in enumerate(params)]
	
	# Concatenate weighted debris samples from non-crashed hypercube corners.
	fragments=[]

	for i1,p1 in enumerate(ind_w[0][0]):
		w1=ind_w[0][1]*(2*i1-1)+1-i1
		for i2,p2 in enumerate(ind_w[1][0]):
			w2=ind_w[1][1]*(2*i2-1)+1-i2
			for i3,p3 in enumerate(ind_w[2][0]):
				w3=ind_w[2][1]*(2*i3-1)+1-i3
				for i4,p4 in enumerate(ind_w[3][0]):
					w4=ind_w[3][1]*(2*i4-1)+1-i4
					for i5,p5 in enumerate(ind_w[4][0]):
						w5=ind_w[4][1]*(2*i5-1)+1-i5
						for i6,p6 in enumerate(ind_w[5][0]):
							w6=ind_w[5][1]*(2*i6-1)+1-i6
							w=w1*w2*w3*w4*w5*w6
							
							if plot: ci+=1
							
							if w==0.: continue
							code='{}{}{}{}{}{}'.format(p1,p2,p3,p4,p5,p6)
							nr=get_index_from_code(code)-1
							if SPHcat[nr].crashed: continue
							corner_Nbig = max(SPHcat[nr].Nbig, 0)
							for val in SPHcat[nr].all[corner_Nbig:]:
								fragments.append([val[0],val[1],val[2],val[3],val[4],val[5],val[7]*w,val[8]]) # x,y,z,vx,vy,vz,w*m/mtot,wf [REBOUND units]
						
							if plot:
								lab='{}  -  {:.4f}'.format(code,w)
								ax.scatter3D(SPHcat[nr].all[corner_Nbig:,0],SPHcat[nr].all[corner_Nbig:,1],SPHcat[nr].all[corner_Nbig:,2],s=0.1,c=col[ci],alpha=1,zorder=1)
								ax.scatter3D(0,0,0,s=10,c=col[ci],alpha=1,label=lab,zorder=0)						
	if plot:
		plt.legend()
		plt.show()
	
	# Save the interpolated debris sample as a collision diagnostic.
	save_debris(fragments)
	
	# Cluster in velocity space using SPH mass fractions as sample weights.
	clusters=[]
	fragments=np.asarray(fragments)
	vfr=fragments[:,[3,4,5]]
	mfr=fragments[:,6]
	res=KMeans(n_clusters=Ncl,random_state=KMEANS_RANDOM_STATE).fit(vfr,sample_weight=mfr)
	
	for i in range(Ncl):
		m=0.
		xvmw=np.zeros(8)
		for j,lab in enumerate(res.labels_):
			if int(lab)==i:
				m+=fragments[j,6]
				xvmw+=fragments[j,:]*fragments[j,6]
		xvmw=xvmw/m
		# Keep raw cluster masses here. The dust retention factor is applied once
		# to the target debris budget in collision_solver before normalization.
		clusters.append([xvmw[:3],xvmw[3:6],m,xvmw[7]])
	
	plot=SHOW_DEBRIS_DIAGNOSTIC_PLOT
	if plot:				
		import matplotlib.pyplot as plt
		from mpl_toolkits import mplot3d
		import pylab	
		
		Ncolors=Ncl		
		cm = pylab.get_cmap('gist_rainbow')
		col=[cm(1.*i/Ncolors) for i in range(Ncolors)]
		cc=[col[int(i)] for i in res.labels_]

		fig = plt.figure(figsize=(10,10))
		ax = plt.axes(projection='3d')
		ax.set_xlabel('vx [AU/yr/2$\\pi$]')
		ax.set_ylabel('vy [AU/yr/2$\\pi$]')
		ax.set_zlabel('vz [AU/yr/2$\\pi$]')   
		dt=0.01
		ax.scatter3D(fragments[:,0],fragments[:,1], fragments[:,2], s=0.1, c=cc,alpha=0.1,zorder=0)
		for i in range(Ncl):
			ax.scatter3D(clusters[i][0][0],clusters[i][0][1],clusters[i][0][2], s=7, c='k',alpha=1,zorder=1)
			ax.plot3D([clusters[i][0][0],clusters[i][0][0]+dt*clusters[i][1][0]],[clusters[i][0][1],clusters[i][0][1]+dt*clusters[i][1][1]],[clusters[i][0][2],clusters[i][0][2]+dt*clusters[i][1][2]],c='k')
		plt.show()
	
	return clusters


#-------------------
# SPH catalogue
#-------------------

def load_SPHcat(fname):
	with open(fname,'r') as f: lines=f.readlines()
	_ = lines.pop(0)
	return [SPHcol(line) for line in lines]

def get_index_from_code(code):
	# returns SPHcol.id given SPH.code
	val=0
	for i in range(len(bases)-1): val=(val+int(code[i]))*(int(bases[i+1]))
	return val+int(code[5])+1

#-------------------
# remove particles
#-------------------

def remove_ps():
	"""Apply the open-boundary removal criteria used for the production run."""
	for i in reversed(range(NGG+1,sim.N)):
		pa,pe=ps[i].a,ps[i].e
		if pe>=REMOVE_ECCENTRICITY_MIN:
			write_events_file(sim.t,f'removed -> e>={REMOVE_ECCENTRICITY_MIN:g}',ps[i],[])
			sim.remove(i)
			write_events_file_outcome([])
			continue
		if pa>REMOVE_A_MAX_AU:
			write_events_file(sim.t,f'removed -> a>{REMOVE_A_MAX_AU:g}',ps[i],[])
			sim.remove(i)
			write_events_file_outcome([])
			continue
		if pa<REMOVE_A_MIN_AU:
			write_events_file(sim.t,f'merged_into_Sun -> a<{REMOVE_A_MIN_AU:g}',ps[0],ps[i])
			ps[0].m+=ps[i].m
			sim.remove(i)
			write_events_file_outcome([ps[0]])
			continue
		if pa*(1.-pe)<REMOVE_Q_MIN_AU:
			write_events_file(sim.t,f'merged_into_Sun -> perihelion<{REMOVE_Q_MIN_AU:g}',ps[0],ps[i])
			ps[0].m+=ps[i].m
			sim.remove(i)
			write_events_file_outcome([ps[0]])
			continue

#-------------------
# particles label
#-------------------

def label_to_code(label):
	if label=="SUN": return 0
	elif label=="JUP": return 1.1
	elif label=="SAT": return 2.1
	elif label[0]=="E": return float(label[1:])+0.2
	elif label[0]=="P": return float(label[1:])+0.3
	elif label[0]=="D": return float(label[1:])+0.4
	else: raise Exception("Wrong label")

def code_to_label(code):
	if code==0: return "SUN"
	elif code==1.1: return "JUP"
	elif code==2.1: return "SAT"
	elif str(code)[-1]=="2": return "E"+str(code)[:-2]
	elif str(code)[-1]=="3": return "P"+str(code)[:-2]
	elif str(code)[-1]=="4": return "D"+str(code)[:-2]
	else: raise Exception("Wrong code")

#-------------------
# output files
#-------------------

def initialize_collisions_file():
	with open(coll_file,'w+') as f:
		f.write('# time [yr]\tv0 [v_esc]\talpha [degrees]\tmtot [MEAR]\tgamma [1]\twft [%]\twfp [%]')
		f.write('\tx_CoM [AU]\ty_CoM [AU]\tz_CoM [AU]\tvx_CoM [AU/yr/2pi]\tvy_CoM [AU/yr/2pi]\tvz_CoM [AU/yr/2pi]')
		f.write('\tchi [rad]\tpsi [rad]\thz [AU^2/yr/2pi]')
		f.write('\tNbig [1]\tm1 [MEAR]\twf1 [%]\tm2 [MEAR]\twf2 [%]\tmfr [MEAR]\twffr [%]')
		f.write('\tNdebris\tlabel1\tlabel2\tlabel_debris\n')

def save_collision(params,xCoM,vCoM,chi,psi,h,survs,mfr,wffr,Ndebris,collider_labels,debris_labels):
	# position and velocity of the collision's CoM are given in the CoM system of coordinate of the simulation
	v0,alpha,mtot,gamma,wft,wfp=params
	with open(coll_file,'a') as f:
		f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(sim.t/2./np.pi,v0,alpha,mtot/MEAR,gamma,wft,wfp))
		for val in xCoM: f.write('\t{}'.format(val))
		for val in vCoM: f.write('\t{}'.format(val))
		f.write('\t{}\t{}\t{}'.format(chi,psi,h[2]))
		Nbig=len(survs)
		if Nbig==0: f.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(Nbig,0,0,0,0,mfr/MEAR,wffr))
		if Nbig==1: f.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(Nbig,survs[0][2]/MEAR,survs[0][3],0,0,mfr/MEAR,wffr))
		if Nbig==2: f.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(Nbig,survs[0][2]/MEAR,survs[0][3],survs[1][2]/MEAR,survs[1][3],mfr/MEAR,wffr))
		if Ndebris==0: f.write('\t{}\t{}\t{}\t{}\n'.format(Ndebris,collider_labels[0],collider_labels[1],'Nan'))
		elif Ndebris==1: f.write('\t{}\t{}\t{}\t{}\n'.format(Ndebris,collider_labels[0],collider_labels[1],debris_labels[0]))
		else: f.write('\t{}\t{}\t{}\t{}\n'.format(Ndebris,collider_labels[0],collider_labels[1],debris_labels[0]+'-'+debris_labels[-1]))

def save_data(t,ps,path):
	sim.move_to_hel()
	with open('{}/{}.t{}'.format(path,path,int(np.around(t))),'w+') as f:
		f.write('# m [MSUN]\tR [AU]\ta [AU]\te [1]\tinc [rad]\tOmega [rad]\tomega [rad]\tM [rad]\twf [%]\tactive [bool]\tcode [str]\n')
		line='{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'
		f.write(line.format(ps[0].m,ps[0].r,0.,0.,0.,0.,0.,0.,0.,1,"SUN"))
		for p in ps[1:sim.N_active]: f.write(line.format(p.m,p.r,p.a,p.e,p.inc,p.Omega,p.omega,p.M,p.params['wf'],1,code_to_label(p.params['code'])))
		for p in ps[sim.N_active:]: f.write(line.format(p.m,p.r,p.a,p.e,p.inc,p.Omega,p.omega,p.M,p.params['wf'],0,code_to_label(p.params['code'])))
	sim.move_to_com()
	
def write_events_file(t,ty,p1,p2):
	with open(events_file,'a+') as f: 
		f.write('-'*50+'\n')
		f.write('TYPE: {}\n'.format(ty))
		f.write('TIME: {} yr\n'.format(t/2./np.pi))
		f.write('INVOLVED PARTICLES  (label, mass [MEAR], wf [%])\n')
		f.write('{}\t{}\t{}\n'.format(code_to_label(p1.params['code']),p1.m/MEAR,p1.params['wf']))
		if not p2==[]: f.write('{}\t{}\t{}\n'.format(code_to_label(p2.params['code']),p2.m/MEAR,p2.params['wf']))
		else: f.write('-\t-\t-\n')

def write_events_file_outcome(outcome):
	with open(events_file,'a+') as f: 
		f.write('RESULTING PARTICLES  (label, mass [MEAR], wf [%])\n')
		if not outcome==[]:
			for p in outcome: f.write('{}\t{}\t{}\n'.format(code_to_label(p.params['code']),p.m/MEAR,p.params['wf']))
		else: f.write('-\t-\t-\n')
	
def collision_snapshot(ps):
	previous_files=glob.glob(scenario+'/*.snapshot')
	collN=len(previous_files)+1
	new_snap='{}/collision_{}.snapshot'.format(scenario,collN)
	sim.move_to_hel()
	
	with open(new_snap,'w+') as f:
		f.write('# m [MSUN]\tR [AU]\ta [AU]\te [1]\tinc [rad]\tOmega [rad]\tomega [rad]\tM [rad]\twf [%]')
		f.write('\tx [AU]\ty [AU]\tz [AU]\tvx [AU/yr/2pi]\tvy [AU/yr/2pi]\tvz [AU/yr/2pi]')
		f.write('\tactive [bool]\tcode [str]\n')
		line='{}'+'\t{}'*16+'\n'
		f.write(line.format(ps[0].m,ps[0].r,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1,"SUN"))
		for p in ps[1:sim.N_active]: f.write(line.format(p.m,p.r,p.a,p.e,p.inc,p.Omega,p.omega,p.M,p.params['wf'],p.x,p.y,p.z,p.vx,p.vy,p.vz,1,code_to_label(p.params['code'])))
		for p in ps[sim.N_active:]: f.write(line.format(p.m,p.r,p.a,p.e,p.inc,p.Omega,p.omega,p.M,p.params['wf'],p.x,p.y,p.z,p.vx,p.vy,p.vz,0,code_to_label(p.params['code'])))

	sim.move_to_com()
	
def save_progr(t,Np,t1,t0,dE):
	with open(sp_file,'a+') as f: f.write(' - t={:.0f} yr   N={}   partial: {:.2f} s   running: {:.2f} s   err_rel={:.2e}\n'.format(t,Np,t1,t0,dE))

def save_col_out(t,Nbig,Nfr,mfr,wffr):
	with open(sp_file,'a+') as f: f.write('  ~~~ Collision at t={} yr:   Nbig={}   Nfr={}   mfr={:.4f} MEAR   wffr={:.2e} %\n'.format(t/np.pi/2.,Nbig,Nfr,mfr/MEAR,wffr))

def save_debris(frag):
	previous_files=glob.glob(scenario+'/*.debris')
	collN=len(previous_files)+1
	new_deb='{}/collision_{}.debris'.format(scenario,collN)
	
	with open(new_deb,'w+') as f:
		f.write('# x[AU]\ty[AU]\tz[AU]\tvx[AU/yr/2pi]\tvy[AU/yr/2pi]\tvz[AU/yr/2pi]\tw*m/mtot[1]\twf[%]\n')
		for fr in frag: f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(fr[0],fr[1],fr[2],fr[3],fr[4],fr[5],fr[6],fr[7]))


#----------------------------------------------------------------------
#----------------------------------------------------------------------


#***************   INITIALIZE SIMULATION OUTPUT FILES   ***************

SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

# old file warning
if new_sim_warning:
	import os
	import glob
	old_files=glob.glob(scenario+'/*.dat')+glob.glob(scenario+'/*.snapshot')
	if old_files!=[]:
		print('\n\n {}\n WARNING!!!\n \n {}\n'.format('~'*50,'~'*50))
		print(' Previous simulation file detected!\n')
		for f in old_files: print(f)
		user=str(input('\nRemove old files? (y,n) '))
		if user=='y':
			for f in old_files: os.remove(f)

# collisions list file
coll_file='{}/coll.dat'.format(scenario)
if glob.glob(coll_file)==[]: initialize_collisions_file()

# events file
events_file='{}/events.dat'.format(scenario)

# save progress file
sp_file='{}/progress.{}'.format(scenario,scenario)


#***********************   SIMULATION SETUP   *************************
											     
sim = rebound.Simulation()

# Integrator options from the manuscript defaults.
sim.integrator = "mercurius"
sim.ri_mercurius.hillfac = MERCURIUS_HILLFAC
sim.dt=dt
sim.ri_ias15.min_dt = IAS15_MIN_DT_FACTOR * sim.dt

# Collision handling and boundary policy.
sim.collision = "direct"
sim.collision_resolve_keep_sorted = 1
sim.boundary = "none"
sim.track_energy_offset = 1

# Custom collision resolver for non-special impacts.
sim.collision_resolve = collision_solver
min_btd = MIN_BACKTRACK_DISTANCE_HILL_RADII

# REBOUNDX carries passive per-particle parameters such as water fraction.
rebx = reboundx.Extras(sim)

# SPH outcome catalogue used by the collision resolver.
SPHcat = load_SPHcat(SPH_TABLE_PATH)

# Particle-removal cadence.
rem_freq = REMOVAL_CHECK_FREQUENCY_STEPS

# Test particles feel active bodies but not each other.
sim.testparticle_type = 1

#----------------------------------------------------------------------
#----------------------------------------------------------------------

#*****************************   INPUT   ******************************

starting_file = '{}/{}.{}'.format(scenario,scenario,t0)

start=np.loadtxt(starting_file,usecols=np.arange(0,9))
hs=np.loadtxt(starting_file,usecols=-1,dtype=str)

# --- Sun
sim.add(m=start[0,0],r=start[0,1])
sim.particles[0].params['wf']=0.
sim.particles[0].params['code']=label_to_code(hs[0])

# --- Planets/Embryos/Planetesimals
for i in range(1,len(start)):
	sim.add(m=start[i,0],r=start[i,1],a=start[i,2],e=start[i,3],inc=start[i,4],Omega=start[i,5],omega=start[i,6],M=start[i,7],hash=hs[i])
	sim.particles[i].params['wf']=start[i,8]
	sim.particles[i].params['code']=label_to_code(hs[i])

# --- Active Particles
sim.N_active=sum(np.loadtxt(starting_file,usecols=-2,dtype=int))

ps = sim.particles
sim.move_to_com()
E0 = sim.calculate_energy() # Calculate initial energy 

#----------------------------------------------------------------------
#----------------------------------------------------------------------		

#*****************************   OUTPUT   *****************************

if t0=='start': tmin=0.
if t0[0]=='t': tmin=int(t0[1:])*np.pi*2.

sim.t=tmin
tend=tmin+Dt*2.*np.pi
times = np.linspace(tmin,tend,Nsaves+1)[1:]

import time
clock0=time.time()

print(' - t={:.0f} yr   N={}   partial: {:.2f} s   running: {:.2f} s'.format(np.round(sim.t/2./np.pi),len(ps),0,0))
if save_progess: save_progr(np.round(sim.t/2./np.pi),len(ps),0,0,0)

for t in times:
	clock1=time.time()
	
	# removing particles stepping 
	tnext=sim.t+rem_freq*sim.dt
	while tnext<t:
		sim.integrate(tnext)
		remove_ps() 	# remove particles
		tnext+=rem_freq*sim.dt
	
	# saving data stepping 
	sim.integrate(t)
	remove_ps() 	# remove particles

	if save_checkpoints: save_data(sim.t/np.pi/2.,ps,scenario)
	
	clock2=time.time()
	E1 = sim.calculate_energy()
	
	print(' - t={:.0f} yr   N={}   partial: {:.2f} s   running: {:.2f} s   err_rel={:.2e}'.format(np.round(sim.t/2./np.pi),len(ps),clock2-clock1,clock2-clock0,(E1-E0)/E0))
	if save_progess: save_progr(np.round(sim.t/2./np.pi),len(ps),clock2-clock1,clock2-clock0,(E1-E0)/E0)
	
