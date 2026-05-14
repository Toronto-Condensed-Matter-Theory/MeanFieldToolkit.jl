module MFTResume
    export ResumeMFT!, ReadMFT

    using FixedPointToolkit, Logging, LinearAlgebra, JLD2

    using ..MeanFieldToolkit.MFTIter: MFTIterator
    using ..MeanFieldToolkit.MFTRun: SolveMFT!


@doc """
```julia
ResumeMFT!(fileName::String ; Update::Function = SimpleMixing, max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) --> SelfCons 
```
Resumes a mean-field simulation from a checkpoint file.
This requires a full checkpoint file, such as one produced with `debug = true` in `SolveMFT!`.
- If `Update` is passed, then the update function is used to perform the self-consistency update.
- If `max_iter` is passed, then the maximum number of iterations is set to `max_iter`.
- If `tol` is passed, then the tolerance for convergence is set to `tol`.
- If `checkpoint_interval` is passed, then the checkpoint interval is set to `checkpoint_interval`.
"""
    function ResumeMFT!(fileName::String ; Update::Function = SimpleMixing, max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) :: SelfCons 

        checkpoint      =   load(fileName)
        if !all(haskey(checkpoint, key) for key in ("inputs", "outputs", "function args", "Update kwargs", "Self-consistency params"))
            error("ResumeMFT! requires a full checkpoint file. Re-run SolveMFT! with debug = true to preserve resumable checkpoints.")
        end

        SelfConsParams  =   Dict{Symbol, Real}(:tol => tol, :max_iter => max_iter, :checkpoint_interval => checkpoint_interval)
        
        SC              =   ContinueFixedPoint!(fileName, MFTIterator, Update, SelfConsParams)
        @info "Completed!"

        return SC
    end


@doc """
```julia
ReadMFT(fileName::String) --> Dict{String, Any}
```
Reads a mean-field simulation from either a full checkpoint file or a compact output file.
Returns a dictionary containing the following keys:
- `Convergence`: The norm of the difference between the input and the output at the last iteration.
- `Expectations`: The expectation values of the order parameters in the last iteration.
- `Iterations`: The number of iterations performed.
- `MFT`: The mean-field theory object used for the simulation when available, otherwise `nothing`.
"""
    function ReadMFT(fileName::String) 

        f   =   load(fileName)

        if all(haskey(f, key) for key in ("inputs", "outputs", "function args"))
            return Dict("Convergence" => norm(f["inputs"][end] - f["outputs"][end]) / sqrt(length(f["inputs"][end])),
                        "Expectations" => f["outputs"][end],
                        "Iterations" => length(f["inputs"]),
                        "MFT" => f["function args"][1])
        end

        return Dict("Convergence" => get(f, "Convergence", nothing),
                    "Expectations" => get(f, "Outputs", nothing),
                    "Iterations" => get(f, "Iterations", nothing),
                    "MFT" => nothing)
    end

end