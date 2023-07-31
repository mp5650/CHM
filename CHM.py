import numpy as np
import dedalus.public as d3
import time
import pathlib
import h5py
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly = (2*np.pi, 2*np.pi)
Nx, Ny = (256, 256)
Beta = 8.0
Viscosity = 1.5e-8
Friction = 1.2e-3
dealias = 3/2
stop_sim_time = 50
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis))
b = dist.Field(name='b', bases=(xbasis,ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP(['psi', 'vx', 'vy', 'q'])
problem.parameters['Bt'] = Beta
problem.parameters['Mu'] = Viscosity
problem.parameters['Al'] = Friction
problem.parameters['Sq2Al'] = np.sqrt(2.0*Friction)
problem.parameters['Ly'] = Ly
problem.parameters['Lx'] = Lx
problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A))"
problem.add_equation("dt(q) + Mu*Lap(Lap(q)) + Al*q - Bt*dy(psi) = -(vx*dx(q) + vy*dy(q))")
problem.add_equation("q - Lap(psi) = 0", condition="(nx!=0) or (ny!=0)")
problem.add_equation("psi = 0", condition="(nx==0) and (ny==0)")
problem.add_equation("vy - dx(psi) = 0")
problem.add_equation("vx + dy(psi) = 0")
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


# Initial conditions
if not pathlib.Path('restart.h5').exists():
    q = solver.state['q']
    #ff = forcing(1.0)
    q['c'] = np.sin(3)
    # For above, need to replace initial condition to be a choice of mode rather than forcing function

    # Timestepping and output
    dt = timestep
    stop_sim_time = 3600.1
    fh_mode = 'overwrite'

else:
    # Restart
    #write, last_dt = solver.load_state('restart.h5', -1)
    logger.info("Loading solver state from: restart.h5")

    with h5py.File('restart.h5', mode='r') as f:
        write = f['scales']['write_number'][-1]
        last_dt = f['scales']['timestep'][-1]
        solver.iteration = solver.inital_iteration = f['scales']['iteration'][-1]

        solver.sim_time = solver.initial_sim_time = f['scales']['sim_time'][-1]
        logger.info("Loading iteration: {}".format(solver.iteration))
        logger.info("Loading write: {}".format(write))
        logger.info("Loading sim time: {}".format(solver.sim_time))
        logger.info("Loading timestep: {}".format(last_dt))

        q = solver.state['q']
        psi = solver.state['psi']
        vx = solver.state['vx']
        vy = solver.state['vy']

        last_q = f['tasks']['q'][-1,:,:]
        # Note: I'm not really sure what the conventions for the DFT used in dedalus are, so I use numpy instead

        np_kx = np.fft.fftfreq(Nx, Lx/Nx/2/np.pi)
        np_ky = np.fft.rfftfreq(Ny, Ly/Ny/2/np.pi)

        np_kxg, np_kyg = np.meshgrid(np_kx, np_ky, indexing='ij')
        np_k2 = np_kxg**2 + np_kyg**2
        invlap_np = np.zeros(np_k2.shape)
        invlap_np[np_k2>0] = -1.0 / np_k2[np_k2>0]

        last_qfft = np.fft.rfft2(last_q)
        last_psifft = invlap_np*last_qfft
        last_vxfft = 1j*np_kyg*last_psifft
        last_vyfft = -1j*np_kxg*last_psifft

        last_psi = np.fft.irfft2(last_psifft)
        last_vx = np.fft.irfft2(last_vxfft)
        last_vy = np.fft.irfft2(last_vyfft)

        gshape = domain.dist.grid_layout.local_shape(scales=1)
        gslice = domain.dist.grid_layout.slices(scales=1)

        q['g'] = last_q[gslice]
        psi['g'] = last_psi[gslice]
        vx['g'] = last_vx[gslice]
        vy['g'] = last_vy[gslice]


    # Timestepping and output
    dt = last_dt
    stop_sim_time = 3600.1
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = stop_sim_time
# Analysis
fh_fixer = solver.evaluator.add_file_handler('fh_fixer', sim_dt=10000.0, max_writes=10, mode=fh_mode)
fh_fixer.add_system(solver.state)
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=snapshotStep, max_writes=8, mode=fh_mode)
snapshots.add_task('q', layout='g', name='q')

# CFL
CFL = d3.CFL(solver, initial_dt=dt, cadence=10, safety=1.0,
                     max_change=1.5, min_change=0.1, max_dt=max_timestep, threshold=0.05)
CFL.add_velocities(('vx', 'vy'))

# Flow properties
output_cadence = 10

flow = d3.GlobalFlowProperty(solver, cadence=output_cadence)
flow.add_property("vx*vx + vy*vy", name='Energy')
flow.add_property("q*q", name='Enstrophy')
curr_time = time.time()

# Main loop
startup_iter = 10
try:
    logger.info('Starting loop')
    start_time = time.time()
    start_iter = solver.iteration
    curr_iter = solver.iteration

    while solver.proceed:
        if solver.iteration - start_iter > 10:
            dt = CFL.compute_dt()

        forcing_func.args = [dt]
        dt = solver.step(dt)
        if (solver.iteration-2) % output_cadence == 0:
            next_time = time.time()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Average timestep (ms): %f' % ((next_time-curr_time) * 1000.0 / (solver.iteration - curr_iter)))
            logger.info('Max v^2 = %f' % flow.max('Energy'))
            logger.info('Average v^2 = %f' % flow.volume_average('Energy'))
            logger.info('Max q^2 = %f' % flow.max('Enstrophy'))
            curr_time = next_time
            curr_iter = solver.iteration
            if solver.iteration - start_iter > 100:
                output_cadence = 100
            if not np.isfinite(flow.max('Enstrophy')):
                raise Exception('NaN encountered')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
