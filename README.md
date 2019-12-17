# README

This project demonstrates a method for finding the solenoidal part of a vector
field.

This is something I found useful, but it is pretty simple mathematically. Still,
in my very limited experience with incompressible fluid simulations, I haven't
come across anyone else using this approach, so I thought it was worth putting
a small test project together.

## Motivation

Any (nicely behaved) vector field can be broken into an irrotational and a
solenoidal part. Finding the irrotational part of a vector field is important in
fluid simulations of incompressible fluids. Since an incompressible fluid cannot
have a divergence in the velocity field, every timestep the irrotational part of
the velocity field must be removed. One common way of doing this is by first
calculating a "pressure" field by using a Poisson solver and then subtracting
the gradient of the pressure from the velocity field.

If the pressure is used to remove the irrotational part of the velocity field,
then ideally the pressure field will be defined on a staggered grid relative to
the velocity field. However, this makes determining the correct boundary
conditions for the pressure field very difficult, since the boundaries of the
pressure field are not aligned with those of the velocity field. The method
demonstrated in this project demonstrates how the intermediate pressure field
can be avoided altogether, making it much easier to enforce boundary conditions
on the velocity field.

## Overview

If a vector field `F` can be broken into an irrotational part `D` and a
solenoidal part `A`, then the following conditions must hold: `curl D = 0`,
`div A = 0`, and `F = A + D`. Note that on a discrete lattice, these equations
will often overdetermine `A` and `D`, depending on the boundary conditions, and
so it will not be possible to exactly satisfy these conditions at all points on
the lattice. By looking at the quantity `curl (curl F)` (see the provided file
`math.tex`), an expression for `A` directly in terms of `F` can be found,
without any intermediate potential necessary.

One downside of using this method is that there is a lot of effective viscosity
added to the simulation.

