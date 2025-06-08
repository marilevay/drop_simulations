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

function plot_and_save(R_in_CGS; g_in_CGS=9.8, T_end=10, n_thetas=40, n_sampling_time_L_mode=16,
                        Bond=nothing, Web=nothing, Ohn=nothing, rho_in_CGS=nothing, 
                        sigma_in_CGS=nothing, nu_in_GCS=nothing, V_in_CGS=nothing)

    println("Starting simulation...")
    pb = Progress(8, desc="Running simulation steps", barlen=30)

    # Step 1: Legendre roots and cos(theta)
    roots, _ = gausslegendre(n_thetas)
    cos_theta_vec = vcat([1.0], reverse(roots))
    theta_vec = acos.(cos_theta_vec)
    q_last = findfirst(x -> x <= 0, cos_theta_vec)
    next!(pb)

    # Step 2: Dimensionless parameters
    Bo = isnothing(Bond) ? rho_in_CGS * g_in_CGS * R_in_CGS^2 / sigma_in_CGS : Bond
    We = isnothing(Web) ? rho_in_CGS * V_in_CGS^2 * R_in_CGS / sigma_in_CGS : Web
    V_in_CGS = isnothing(V_in_CGS) ? sqrt((We * sigma_in_CGS) / (rho_in_CGS * R_in_CGS)) : V_in_CGS
    Oh = isnothing(Ohn) ? nu_in_GCS * sqrt(rho_in_CGS / (sigma_in_CGS * R_in_CGS)) : Ohn
    next!(pb)

    # Step 3: File paths
    desktop_path = "Mode_Convergence_R=$(R_in_CGS)_modes=$(n_thetas)_Web=$(We)"
    folder_name = joinpath(desktop_path, "simulation_We=$(We)_Oh=$(Oh)_Bo=$(Bo)_modes=$(n_thetas)")
    mkpath(folder_name)
    next!(pb)

    # Step 4: Units and initial conditions
    unit_length = R_in_CGS
    unit_time = sqrt(rho_in_CGS * R_in_CGS^3 / sigma_in_CGS)
    H = 1.0
    V = -sqrt(We)
    next!(pb)

    # Step 5: Run simulation
    sim = running_simulation(n_thetas, n_sampling_time_L_mode, T_end, H, V, Bo, theta_vec, Oh)
    all_amps_list, all_amps_vel_list, all_press_list, all_h_list, all_v_list,
    all_m_list, all_north_poles_list, all_south_poles_list,
    all_maximum_radii_list, all_times_list_original = sim
    all_times_list = all_times_list_original[1:length(all_m_list)]
    next!(pb)

    # Step 6: Arrays for CSV export
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
    next!(pb)

    # Step 7: Calculate metrics
    center_coef_rest = center_coefficient_of_restitution(all_v_list, all_m_list)
    alpha_coef_rest, contact_time_nd, contact_time_ind = alpha_coefficient_of_restitution(
        all_h_list, all_v_list, all_m_list, all_times_list, Bo, We)
    min_north_pole_h_nd, time_min_north_pole_h_nd = min_north_pole_height(all_north_poles_list, all_times_list)
    max_radius_nd, time_max_radius_nd = max_radius_over_t(all_maximum_radii_list, contact_time_ind, all_times_list)
    max_contact_radius_nd, time_max_contact_radius_nd = maximum_contact_radius(all_amps_list, all_m_list, all_times_list, n_thetas, theta_vec)
    max_radial_project_nd, time_max_radial_project_nd = max_radial_projection(all_amps_list, n_thetas, all_times_list, theta_vec)
    min_side_h_nd, time_min_side_h_nd = min_side_height(all_amps_list, n_thetas, all_times_list, all_h_list, theta_vec)
    next!(pb)

    # Step 8: Generate plots and save results
    height_plots(all_north_poles_list, all_south_poles_list, all_h_list, all_times_list, folder_name)
    maximum_radius_plot(all_maximum_radii_list, all_times_list, folder_name)
    amplitudes_over_time_plot(all_amps_list, all_times_list, folder_name, n_thetas)

    contact_time_cgs = contact_time_nd * unit_time
    min_north_pole_h_cgs = min_north_pole_h_nd * R_in_CGS
    time_min_north_pole_h_cgs = time_min_north_pole_h_nd * unit_time
    max_radius_cgs = max_radius_nd * R_in_CGS
    time_max_radius_cgs = time_max_radius_nd * unit_time
    max_contact_radius_cgs = max_contact_radius_nd * R_in_CGS
    time_max_contact_radius_cgs = time_max_contact_radius_nd * unit_time
    max_radial_project_cgs = max_radial_project_nd * R_in_CGS
    time_max_radial_project_cgs = time_max_radial_project_nd * unit_time
    min_side_h_cgs = min_side_h_nd * R_in_CGS
    time_min_side_h_cgs = time_min_side_h_nd * unit_time

    csv_main = joinpath(desktop_path, "simulation_results.csv")
    summary_row = [R_in_CGS, V_in_CGS, Bo, We, Oh, alpha_coef_rest, center_coef_rest,
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

    open(csv_main, "a") do io
        writedlm(io, [summary_row], ',')
    end

    writedlm(joinpath(folder_name, "all_amps.csv"), all_amps_array', ',')
    writedlm(joinpath(folder_name, "all_amps_vel.csv"), all_amps_vel_array', ',')
    writedlm(joinpath(folder_name, "all_press.csv"), all_press_array', ',')

    extra_data = hcat(all_times_array, all_h_array, all_v_array, all_m_array,
                      all_north_poles_array, all_south_poles_array, all_max_radii_array)
    writedlm(joinpath(folder_name, "simulation_results_extra.csv"), extra_data, ',')

    println("Simulation and saving complete.")
    return ["All Files Saved"]
end

# Example call:
function main()
    R_in_CGS = 0.1                # Radius in cm
    g_in_CGS = 9.8                # Gravity in cm/s²
    T_end = 10                    # Duration of simulation (non-dimensional time)
    n_thetas = 40                 # Number of sampled angles
    n_sampling_time_L_mode = 16  # Sampling time for L mode

    # Run the simulation and save results
    plot_and_save(R_in_CGS;
                  g_in_CGS=g_in_CGS,
                  T_end=T_end,
                  n_thetas=n_thetas,
                  n_sampling_time_L_mode=n_sampling_time_L_mode,
                  rho_in_CGS=1.0,
                  sigma_in_CGS=72.0,
                  nu_in_GCS=0.01,
                  V_in_CGS=30.0)
end

# Call the main function if this file is run directly
main()
