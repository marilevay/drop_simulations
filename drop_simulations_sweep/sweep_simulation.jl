using DelimitedFiles, CSV, DataFrames, Printf, LinearAlgebra, FileIO
using Statistics, SpecialPolynomials, Dates
using FilePathsBase, Glob
include("sweep_utils.jl")

function running_simulation(n_thetas, n_sampling_time_L_mode, T_end, H, V, Bo, theta_vec, Oh)
    delta_time_max = (2π / sqrt(n_thetas * (n_thetas + 2) * (n_thetas - 1)) / n_sampling_time_L_mode)
    n_times_base = ceil(Int, T_end / delta_time_max)
    all_times_list = collect(0:delta_time_max:(n_times_base * delta_time_max))

    amps_prev_vec = zeros(n_thetas - 1)
    amps_vel_prev_vec = zeros(n_thetas - 1)
    h_prev = H
    v_prev = V
    m_prev = 0

    all_amps_list = [amps_prev_vec]
    all_amps_vel_list = [amps_vel_prev_vec]
    all_press_list = [zeros(n_thetas + 1)]
    all_h_list = [h_prev]
    all_v_list = [v_prev]
    all_m_list = [0]
    all_north_poles_list = [height_poles(amps_prev_vec, H, n_thetas)[1]]
    all_south_poles_list = [height_poles(amps_prev_vec, H, n_thetas)[2]]
    all_maximum_radii_list = [max_radius_at_each_t(amps_prev_vec, n_thetas, theta_vec)]

    ind_time = 1
    lift_off = false
    count = 0


    while all_times_list[ind_time] < all_times_list[end]
    
        if length(all_m_list) > 2 && all_m_list[end] == 0 && all_m_list[end-1] > 0
            lift_off = true
        end
        if lift_off
            count += 1
            if count == 50
                break
            end
        end

        delta_t = all_times_list[ind_time+1] - all_times_list[ind_time]

        sol_vec_m_prev, err_m_prev = try_q(amps_prev_vec, amps_vel_prev_vec, h_prev, v_prev, m_prev, delta_t, n_thetas, Bo, Oh)
        sol_vec_m_prev_m1, err_m_prev_m1 = try_q(amps_prev_vec, amps_vel_prev_vec, h_prev, v_prev, m_prev-1, delta_t, n_thetas, Bo, Oh)

        if err_m_prev_m1 < err_m_prev
            sol_vec_m_prev_m2, err_m_prev_m2 = try_q(amps_prev_vec, amps_vel_prev_vec, h_prev, v_prev, m_prev-2, delta_t, n_thetas, Bo, Oh)
            if err_m_prev_m2 < err_m_prev_m1
                time_insert = (all_times_list[ind_time] + all_times_list[ind_time+1]) / 2
                insert!(all_times_list, ind_time+1, time_insert)
            else
                ind_time += 1
                m_prev -= 1
                amps_prev_vec = sol_vec_m_prev_m1[1:n_thetas-1]
                amps_vel_prev_vec = sol_vec_m_prev_m1[n_thetas:2n_thetas-2]
                press_prev_vec = sol_vec_m_prev_m1[2n_thetas-1:end-2]
                h_prev = sol_vec_m_prev_m1[end-1]
                v_prev = sol_vec_m_prev_m1[end]
                push!(all_amps_list, amps_prev_vec)
                push!(all_amps_vel_list, amps_vel_prev_vec)
                push!(all_press_list, press_prev_vec)
                push!(all_h_list, h_prev)
                push!(all_v_list, v_prev)
                push!(all_m_list, m_prev)
                push!(all_north_poles_list, height_poles(amps_prev_vec, h_prev, n_thetas)[1])
                push!(all_south_poles_list, height_poles(amps_prev_vec, h_prev, n_thetas)[2])
                push!(all_maximum_radii_list, max_radius_at_each_t(amps_prev_vec, n_thetas, theta_vec))
            end
        else
            sol_vec_m_prev_p1, err_m_prev_p1 = try_q(amps_prev_vec, amps_vel_prev_vec, h_prev, v_prev, m_prev+1, delta_t, n_thetas, Bo, Oh)
            if err_m_prev_p1 < err_m_prev
                sol_vec_m_prev_p2, err_m_prev_p2 = try_q(amps_prev_vec, amps_vel_prev_vec, h_prev, v_prev, m_prev+2, delta_t, n_thetas, Bo, Oh)
                if err_m_prev_p2 < err_m_prev_p1
                    time_insert = (all_times_list[ind_time] + all_times_list[ind_time+1]) / 2
                    insert!(all_times_list, ind_time+1, time_insert)
                else
                    ind_time += 1
                    m_prev += 1
                    amps_prev_vec = sol_vec_m_prev_p1[1:n_thetas-1]
                    amps_vel_prev_vec = sol_vec_m_prev_p1[n_thetas:2n_thetas-2]
                    press_prev_vec = sol_vec_m_prev_p1[2n_thetas-1:end-2]
                    h_prev = sol_vec_m_prev_p1[end-1]
                    v_prev = sol_vec_m_prev_p1[end]
                    push!(all_amps_list, amps_prev_vec)
                    push!(all_amps_vel_list, amps_vel_prev_vec)
                    push!(all_press_list, press_prev_vec)
                    push!(all_h_list, h_prev)
                    push!(all_v_list, v_prev)
                    push!(all_m_list, m_prev)
                    push!(all_north_poles_list, height_poles(amps_prev_vec, h_prev, n_thetas)[1])
                    push!(all_south_poles_list, height_poles(amps_prev_vec, h_prev, n_thetas)[2])
                    push!(all_maximum_radii_list, max_radius_at_each_t(amps_prev_vec, n_thetas, theta_vec))
                end
            elseif isinf(err_m_prev_p1) && isinf(err_m_prev) && isinf(err_m_prev_m1)
                time_insert = (all_times_list[ind_time] + all_times_list[ind_time+1]) / 2
                insert!(all_times_list, ind_time+1, time_insert)
            else
                ind_time += 1
                amps_prev_vec = sol_vec_m_prev[1:n_thetas-1]
                amps_vel_prev_vec = sol_vec_m_prev[n_thetas:2n_thetas-2]
                press_prev_vec = sol_vec_m_prev[2n_thetas-1:end-2]
                h_prev = sol_vec_m_prev[end-1]
                v_prev = sol_vec_m_prev[end]
                push!(all_amps_list, amps_prev_vec)
                push!(all_amps_vel_list, amps_vel_prev_vec)
                push!(all_press_list, press_prev_vec)
                push!(all_h_list, h_prev)
                push!(all_v_list, v_prev)
                push!(all_m_list, m_prev)
                push!(all_north_poles_list, height_poles(amps_prev_vec, h_prev, n_thetas)[1])
                push!(all_south_poles_list, height_poles(amps_prev_vec, h_prev, n_thetas)[2])
                push!(all_maximum_radii_list, max_radius_at_each_t(amps_prev_vec, n_thetas, theta_vec))
            end
        end
    end

    return (all_amps_list, all_amps_vel_list, all_press_list, all_h_list, all_v_list,
            all_m_list, all_north_poles_list, all_south_poles_list, all_maximum_radii_list,
            all_times_list)
end
