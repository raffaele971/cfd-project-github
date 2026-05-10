### Driven cavity solvers

This repository contains several studies related to the classic lid-driven cavity problem, a fundamental benchmark in Computational Fluid Dynamics (CFD). The lid-driven cavity involves an incompressible, viscous fluid inside a container with three stationary walls and a top wall (lid) moving at constant velocity. This motion induces the formation of complex recirculating flows and vortices within the cavity, making it an essential test case for evaluating and validating numerical methods for solving the Navier-Stokes equations.

📂 Repository Contents
The repository is organized into the following main case studies:

1. Square (or Rectangular) Lid-Driven Cavity
Description: Simulation of the classic lid-driven cavity flow, both in square and rectangular geometries.
Numerical Approach: The fluid flow is solved using the streamfunction-vorticity (ψ-ζ) formulation, which is well-suited for investigating vorticity dynamics and provides deeper insight into the underlying coherent structures and vortex formations within the cavity.
Objective: To analyze the primary and secondary (corner) vortices, benchmark numerical methods, and compare results with reference solutions in literature.
2. Passive Scalar Transport in Lid-Driven Cavity
Description: Extension of the classical cavity problem by introducing the passive transport of a scalar quantity (such as temperature or concentration) within the flow.
Purpose: This study allows for the exploration of both advection and diffusion phenomena in the presence of complex flow patterns, making it a valuable test for coupled simulation strategies (fluid + scalar transport).
Applications: Understanding scalar transport is crucial for various engineering applications, such as heat transfer, pollutant dispersion, and mixing processes.

🧑‍🔬 Goals and Applications
Validate CFD codes and numerical schemes for incompressible viscous flows.
Investigate the characteristics of vortex dynamics and scalar mixing in confined geometries.
Provide well-documented cases for didactic and research purposes, facilitating reproducibility and further exploration.
 vortical structures.
