"""Quick test of TVB simulation"""
import numpy as np
from tvb.simulator.lab import *

print("Setting up connectivity...")
conn = connectivity.Connectivity.from_file()
print(f"Connectivity loaded: {conn.weights.shape}")

# Downsample to 48 nodes
indices = np.linspace(0, len(conn.weights) - 1, 48, dtype=int)
conn.weights = conn.weights[indices][:, indices]
conn.tract_lengths = conn.tract_lengths[indices][:, indices]
conn.region_labels = conn.region_labels[indices]
conn.centres = conn.centres[indices]
conn.configure()
print(f"Downsampled to 48 nodes")

print("Setting up model...")
model = models.ReducedWongWang()

print("Setting up coupling...")
coup = coupling.Linear(a=np.array([0.0152]))

print("Setting up integrator...")
heunint = integrators.HeunStochastic(
    dt=2**-4,
    noise=noise.Additive(nsig=np.array([0.001]))
)

print("Setting up monitors...")
mon = monitors.TemporalAverage(period=800.0)  # 0.8 seconds in ms

print("Creating simulator...")
sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coup,
    integrator=heunint,
    monitors=[mon]
)

print("Configuring simulator...")
sim.configure()

print("Running simulation for 10 seconds...")
results = []
for data in sim(simulation_length=10000.0):  # 10 seconds
    if data[0][1] is not None:
        results.append(data[0][1])
        print(f"  Progress: {len(results)} chunks")

print(f"\nSimulation complete! Got {len(results)} data chunks")
if results:
    concatenated = np.concatenate(results)
    print(f"Final shape: {concatenated.shape}")
