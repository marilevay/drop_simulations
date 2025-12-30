using DelimitedFiles, CSV, DataFrames, Printf, LinearAlgebra, FileIO
using Statistics, SpecialPolynomials, Dates, FastGaussQuadrature
using FilePathsBase, Glob
using ProgressMeter
using TerminalLoggers
using Plots
include("sweep_simulation.jl")
include("sweep_utils.jl")

function height_plots(all_north_poles_list, all_south_poles_list, all_h_list, all_times_list, folder_name)
    plot_path = joinpath(folder_name, "height_plots.png")
    plt = Plots.plot(all_times_list, [getindex.(all_north_poles_list, 1),
                                getindex.(all_south_poles_list, 1),
                                all_h_list],
               label=["North Pole" "South Pole" "Center of Mass"],
               xlabel="Non-Dimensional Time", ylabel="Height (in Radii)",
               linewidth=2)

    savefig(plt, plot_path)
    return plot_path
end

function maximum_radius_plot(all_maximum_radii_list, all_times_list, folder_name)
    plot_path = joinpath(folder_name, "max_radius_plot.png")
    plt = Plots.plot(all_times_list, all_maximum_radii_list,
               label="Max Radius", xlabel="Non-Dimensional Time",
               ylabel="Maximum Radius (ND)", linewidth=2)
    savefig(plt, plot_path)
    return plot_path
end

function amplitudes_over_time_plot(all_amps_list, all_times_list, folder_name, n_thetas)
    plot_path = joinpath(folder_name, "amplitudes_over_time_plot.png")
    amps_mat = reduce(hcat, all_amps_list)

    plt = Plots.plot(xlabel="Non-Dimensional Time", ylabel="Amplitude")

    @showprogress 1 "Plotting mode amplitudes..." for i in 1:(n_thetas - 1)
        Plots.plot!(plt, all_times_list, amps_mat[i, :], label="Mode $i")
    end

    savefig(plt, plot_path)
    return plot_path
end

function resolve_dimensionless(; Bond, Web, Ohn, R_in_CGS, g_in_CGS, rho_in_CGS, sigma_in_CGS, nu_in_GCS, V_in_CGS)
    # Check if all dimensionless numbers are provided
    if !isnothing(Bond) && !isnothing(Web) && !isnothing(Ohn)
        return Bond, Web, Ohn
    end

    # Otherwise, check if all physical parameters are provided
    if all(!isnothing, [R_in_CGS, g_in_CGS, rho_in_CGS, sigma_in_CGS, nu_in_GCS, V_in_CGS])
        Bo = rho_in_CGS * g_in_CGS * R_in_CGS^2 / sigma_in_CGS
        We = rho_in_CGS * V_in_CGS^2 * R_in_CGS / sigma_in_CGS
        Oh = nu_in_GCS * sqrt(rho_in_CGS / (sigma_in_CGS * R_in_CGS))
        return Bo, We, Oh
    end

    error("You must provide either all of (Bond, Web, Ohn) or all of (R_in_CGS, g_in_CGS, rho_in_CGS, sigma_in_CGS, nu_in_GCS, V_in_CGS).")
end

function plot_and_save(R_in_CGS; g_in_CGS=9.8, T_end=10, n_thetas=40, n_sampling_time_L_mode=16,
                        Bond=nothing, Web=nothing, Ohn=nothing, rho_in_CGS=nothing, 
                        sigma_in_CGS=nothing, nu_in_GCS=nothing, V_in_CGS=nothing)

    """
    Function that runs the simulation and saves all results and plots to a designated path on your computer.

    Inputs:
    T_end (int): Duration of Simulation (in non-dimensional-time).
    n_thetas (int): Total number of sampled angles.
    n_sampling_time_L_mode (int): Number of sampling points in the time domain for each Legendre mode.

    either:
    Bond (float or Vector{float}): Bond number, non-dimensional gravity. If you want to define from R_in_CGS, sigma_in_CGS, rho_in_CGS, and g_in_CGS, ignore.
        If want to set it to 0: Input 0
    Web (float or Vector{float}): Weber number, non-dimensional velocity. If you want to define from R_in_CGS, sigma_in_CGS, rho_in_CGS, and V_in_CGS, ignore.
        If want to set it to 0: Input 0   
    Ohn (float or Vector{float}): Ohnesorghe number, non-dimensional viscosity. If you want to define from R_in_CGS, sigma_in_CGS, rho_in_CGS, and nu_in_CGS, ignore.
        If want to set it to 0: Input 0
    (and all others = nothing).

    or:
    rho_in_CGS (float): Density of chosen liquid in g/cm^3. If want to run simulations based on non-dimensional variables (Bo, Oh, We) only, ignore.
    sigma_in_CGS (float): Surface tension of chosen liquid in dyne/cm. If want to run simulations based on non-dimensional variables (Bo, Oh, We) only, ignore.
    nu_in_GCS (float): in cm^2/s. If want to run simulations based on non-dimensional variables (Bo, Oh, We) only, ignore.
    g_in_CGS (float): Gravity in cm/s^2.
    R_in_CGS (float): Radius of droplet in cm.
    V_in_CGS (float): Velocity of the droplet's center of mass in cm/s. If want to run simulations based on non-dimensional variables (Bo, Oh, We) only, ignore.


    Outputs: 
    Saves all plots and data to a designated path on your computer.
    """

    println("""
    Parameters selected:
        - R_in_CGS: $R_in_CGS
        - g_in_CGS: $g_in_CGS
        - T_end: $T_end
        - n_thetas: $n_thetas
        - n_sampling_time_L_mode: $n_sampling_time_L_mode
        - Bond: $Bond
        - Web: $Web
        - Ohn: $Ohn
        - rho_in_CGS: $rho_in_CGS
        - sigma_in_CGS: $sigma_in_CGS
        - nu_in_GCS: $nu_in_GCS
        - V_in_CGS: $V_in_CGS
    """)

    # Resolve dimensionless numbers or error if not enough info
    Bo, We, Oh = resolve_dimensionless(
        Bond=Bond, Web=Web, Ohn=Ohn,
        R_in_CGS=R_in_CGS, g_in_CGS=g_in_CGS, rho_in_CGS=rho_in_CGS,
        sigma_in_CGS=sigma_in_CGS, nu_in_GCS=nu_in_GCS, V_in_CGS=V_in_CGS
    )
            
    pb = Progress(length(Web), desc="Running simulation steps", barlen=30)

    # Legendre roots and cos(theta)
    roots, _ = gausslegendre(n_thetas)
    cos_theta_vec = vcat([1.0], reverse(roots))
    theta_vec = acos.(cos_theta_vec)
    q_last = findfirst(x -> x <= 0, cos_theta_vec)
    next!(pb)

    # File paths (create directory only once)
    # Helper function to build the directory name
    function build_desktop_path(; R_in_CGS=nothing, n_thetas=nothing, Bo=nothing, Oh=nothing, We=nothing)
        parts = ["Mode_Convergence"]
        if !isnothing(R_in_CGS)
            push!(parts, "R=$(R_in_CGS)")
        end
        if !isnothing(n_thetas)
            push!(parts, "modes=$(n_thetas)")
        end
        if !isnothing(Bo)
            push!(parts, "Bo=$(Bo)")
        end
        if !isnothing(Oh)
            push!(parts, "Oh=$(Oh)")
        end
        if !isnothing(We)
            push!(parts, "We=$(We)")
        end
        return join(parts, "_")
    end

    # Usage in your function:
    desktop_path = build_desktop_path(R_in_CGS=R_in_CGS, n_thetas=n_thetas, Bo=Bo, Oh=Oh, We=nothing)

    if !isdir(desktop_path)
        mkpath(desktop_path)
    end

    csv_main = joinpath(desktop_path, "simulation_results.csv")
    csv_header = ["R_in_CGS", "V_in_CGS", 
                "Bo", "We", "Oh",
                "Alpha Coefficient of restitution", "Center Coefficient of restitution", 
                "Contact time ND", "Contact time CGS",
                "Min north pole height ND", "Min north pole height CGS",
                "Time min north pole height ND", "Time min north pole height CGS",
                "Max radius ND", "Max radius CGS",
                "Time of max radius ND", "Time of max radius CGS",
                "Max contact radius ND", "Max contact radius CGS",
                "Time max contact radius ND", "Time max contact radius CGS",
                "Max radial projection ND", "Max radial projection CGS",
                "Time max radial projection ND", "Time max radial projection CGS",
                "Min side height ND", "Min side height CGS",
                "Time min side height ND", "Time min side height CGS"]

    # Write header if file does not exist
    if !isfile(csv_main)
        open(csv_main, "w") do io
            writedlm(io, [csv_header], ',')
        end
    end

    # Step 4: Loop over Weber numbers
    for We in Web
        folder_name = joinpath(desktop_path, "simulation_We=$(We)_Oh=$(Oh)_Bo=$(Bo)_modes=$(n_thetas)")

        if !isdir(folder_name)
            mkpath(folder_name)
        end

        next!(pb)

        # Compute V_in_CGS if needed
        this_We = isnothing(We) ? rho_in_CGS * V_in_CGS^2 * R_in_CGS / sigma_in_CGS : We

        # Units and initial conditions
        unit_length = R_in_CGS
        unit_time = isnothing(R_in_CGS) ? sqrt(rho_in_CGS / sigma_in_CGS) : sqrt(rho_in_CGS * R_in_CGS^3 / sigma_in_CGS)
        H = 1.0
        V = -sqrt(this_We)

        # Run simulation
        sim = running_simulation(n_thetas, n_sampling_time_L_mode, T_end, H, V, Bo, theta_vec, Oh)
        all_amps_list, all_amps_vel_list, all_press_list, all_h_list, all_v_list,
        all_m_list, all_north_poles_list, all_south_poles_list,
        all_maximum_radii_list, all_times_list_original = sim
        all_times_list = all_times_list_original[1:length(all_m_list)]

        # Arrays for CSV export
        all_amps_array = hcat(all_amps_list...)
        all_amps_vel_array = hcat(all_amps_vel_list...)
        all_press_array = hcat(all_press_list...)
        all_h_array = reshape(all_h_list, :, 1)
        all_v_array = reshape(all_v_list, :, 1)
        all_m_array = reshape(all_m_list, :, 1)
        all_times_array = reshape(all_times_list, :, 1)
        all_north_poles_array = reshape(all_north_poles_list, :, 1)
        all_south_poles_array = reshape(all_south_poles_list, :, 1)
        all_max_radii_array = reshape(all_maximum_radii_list, :, 1)

        # Calculate metrics
        center_coef_rest = center_coefficient_of_restitution(all_v_list, all_m_list)
        alpha_coef_rest, contact_time_nd, contact_time_ind = alpha_coefficient_of_restitution(
            all_h_list, all_v_list, all_m_list, all_times_list, Bo, this_We)
        min_north_pole_h_nd, time_min_north_pole_h_nd = min_north_pole_height(all_north_poles_list, all_times_list)
        max_radius_nd, time_max_radius_nd = max_radius_over_t(all_maximum_radii_list, contact_time_ind, all_times_list)
        max_contact_radius_nd, time_max_contact_radius_nd = maximum_contact_radius(all_amps_list, all_m_list, all_times_list, n_thetas, theta_vec)
        max_radial_project_nd, time_max_radial_project_nd = max_radial_projection(all_amps_list, n_thetas, all_times_list, theta_vec)
        min_side_h_nd, time_min_side_h_nd = min_side_height(all_amps_list, n_thetas, all_times_list, all_h_list, theta_vec)

        resolve_factor(val_nd, factor) = isnothing(factor) ? val_nd : val_nd * factor

        contact_time_cgs           = resolve_factor(contact_time_nd, unit_time)
        min_north_pole_h_cgs       = resolve_factor(min_north_pole_h_nd, R_in_CGS)
        time_min_north_pole_h_cgs  = resolve_factor(time_min_north_pole_h_nd, unit_time)
        max_radius_cgs             = resolve_factor(max_radius_nd, R_in_CGS)
        time_max_radius_cgs        = resolve_factor(time_max_radius_nd, unit_time)
        max_contact_radius_cgs     = resolve_factor(max_contact_radius_nd, R_in_CGS)
        time_max_contact_radius_cgs= resolve_factor(time_max_contact_radius_nd, unit_time)
        max_radial_project_cgs     = resolve_factor(max_radial_project_nd, R_in_CGS)
        time_max_radial_project_cgs= resolve_factor(time_max_radial_project_nd, unit_time)
        min_side_h_cgs             = resolve_factor(min_side_h_nd, R_in_CGS)
        time_min_side_h_cgs        = resolve_factor(time_min_side_h_nd, unit_time)

        summary_row = [R_in_CGS, V_in_CGS, Bo, this_We, Oh, alpha_coef_rest, center_coef_rest,
                    contact_time_nd, contact_time_cgs,
                    min_north_pole_h_nd, min_north_pole_h_cgs,
                    time_min_north_pole_h_nd, time_min_north_pole_h_cgs,
                    max_radius_nd, max_radius_cgs,
                    time_max_radius_nd, time_max_radius_cgs,
                    max_contact_radius_nd, max_contact_radius_cgs,
                    time_max_contact_radius_nd, time_max_contact_radius_cgs,
                    max_radial_project_nd, max_radial_project_cgs,
                    time_max_radial_project_nd, time_max_radial_project_cgs,
                    min_side_h_nd, min_side_h_cgs,
                    time_min_side_h_nd, time_min_side_h_cgs]
        
        print("SUMMARY ROW: ", summary_row, "\n")

        open(csv_main, "a") do io
            print("Row appended to CSV")
            writedlm(io, [summary_row], ',')
        end

        # Save other outputs with We in the filename if needed
        writedlm(joinpath(folder_name, "all_amps_We=$(this_We).csv"), all_amps_array', ',')
        writedlm(joinpath(folder_name, "all_amps_vel_We=$(this_We).csv"), all_amps_vel_array', ',')
        writedlm(joinpath(folder_name, "all_press_We=$(this_We).csv"), all_press_array', ',')

        extra_csv = joinpath(folder_name, "simulation_results_extra.csv")
        extra_csv_header = ["Times", "H", "V", "M", "North Pole", "South Pole", "Maximum Radius at t"]
        extra_data = hcat(all_times_array, all_h_array, all_v_array, all_m_array,
                        all_north_poles_array, all_south_poles_array, all_max_radii_array)
        
                        if !isfile(extra_csv)
            open(extra_csv, "w") do io
                writedlm(io, [extra_csv_header], ',')
            end
        end
        open(extra_csv, "a") do io
            writedlm(io, extra_data, ',')
        end

        # Save plots with We in the filename if needed
        height_plots(all_north_poles_list, all_south_poles_list, all_h_list, all_times_list, folder_name)
        maximum_radius_plot(all_maximum_radii_list, all_times_list, folder_name)
        amplitudes_over_time_plot(all_amps_list, all_times_list, folder_name, n_thetas)
    end

    println("Simulation and saving complete.")
    return ["All Files Saved"]
end

# Example call (select a column from the CSV file to use as input):
function main()
    R_in_CGS = nothing
    #Bo_list = CSV.File("CSV/File/Path"; select=[:Bo]).Bo
    We_list = CSV.File("CSV/File/Path"; select=[:We]).We
    
    # Run the simulation and save results
    plot_and_save(R_in_CGS; g_in_CGS=9.8, T_end=10, n_thetas=40, n_sampling_time_L_mode=16,
                            Bond=0, Web=We_list, Ohn=0, rho_in_CGS=nothing, 
                            sigma_in_CGS=nothing, nu_in_GCS=nothing, V_in_CGS=nothing)
end

main()
