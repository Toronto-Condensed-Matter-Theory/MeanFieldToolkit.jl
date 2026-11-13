# Examples

Two kinds of examples, covering the Hubbard model, the Heisenberg model with fermionic spinons, and the slave-boson
t-J model on the square, triangular, honeycomb and kagome lattices. All of them are unrestricted mean-field theory:
every spin channel on every site and nearest-neighbour bond (and, for t-J, every singlet and triplet pairing channel)
is a free order parameter, in unit cells large enough for the natural magnetic orders.

## `MinimalExamples/`: one model on one lattice, spelled out

Plain scripts, no helper functions, each under 100 lines: build the `UnitCell`, the `Param`s and the model, call
`SolveMFT!` from an explicit starting point, then raise the temperature step by step.

| Script | What it does |
|---|---|
| `<Lattice>Hubbard.jl` | Hubbard model at half filling: magnetic order, and its melting with temperature (Néel or ordering temperature, free energy). |
| `<Lattice>Heisenberg.jl` | Heisenberg model with fermionic spinons: the classical ordered state, a spinon-hopping (spin-liquid) ansatz, and the ordering temperature. |
| `<Lattice>tJ.jl` | Slave-boson t-J model at one doping: the pairing pattern on each bond (d-wave, d + id, …) and the pairing temperature. |
| `SquareAltermagnet.jl` | Néel order with sublattice-dependent third-neighbour hopping: d-wave spin splitting with zero net moment. |
| `KagomeFluxScan.jl` | Free energy of kagome spinon states with a fixed flux pattern [ϕ, ϕ, π − 2ϕ] at several temperatures. |

`<Lattice>` is `Square`, `Triangular`, `Honeycomb` or `Kagome`. Run them from the repository root:

```sh
julia --project=Sample -e 'using Pkg; Pkg.instantiate()'     # once
julia --project=Sample Sample/MinimalExamples/SquareHubbard.jl
```

## `Notebooks/`: one Pluto notebook per lattice

`Square.jl`, `Triangular.jl`, `Honeycomb.jl` and `Kagome.jl` run all three models on that lattice (scans in U and in
doping, finite temperature) in about a minute. They use the short functions in `common.jl`, which wrap exactly the loops
written out in `MinimalExamples/`. Open them with [Pluto](https://plutojl.org):

```julia
using Pluto; Pluto.run()        # then open Sample/Notebooks/Square.jl
```

The first cell activates and instantiates the `Sample` environment.

## Notes

- `GetMFTFreeEnergy(mft)` returns the free energy per site F = E − TS, the quantity to compare between phases at T > 0.
