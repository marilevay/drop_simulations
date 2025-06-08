using LinearAlgebra
using SpecialFunctions
using FastGaussQuadrature
using DelimitedFiles
using ProgressMeter
using Printf
using LegendrePolynomials
using ColorSchemes

# For 3D plotting and animation
using Plots
using GeometryBasics: Point3f0, TriangleFace, Mesh 
using GLMakie  
using CairoMakie
using Statistics
using Animations
using Colors
plotlyjs() 

function Legendre(x::Real, n::Int64, n_thetas::Int64)
  """
  Function that evaluates nth Legendre polynomial at x using the LegendrePolynomials package.

  Inputs:
  - x (Real): X-value at which Legendre polynomial is evaluated. The Real abstract type in Julia includes both Int, Float64.
  - n (Int): Number of Legendre polynomial to be evaluated
- n_thetas (Int): Total number of sampled angles (used to avoid numerical errors)

  Output:
  - value (Float64): Value of nth Legendre polynomial at x
  """
    if n == n_thetas && x != 1
        # To avoid numerical errors very close to zero
        value = 0.0
    else
        value = LegendrePolynomials.Pl(x, n)
    end
    return value
end

function matrix(delta_t::Float64, q::Int, n_thetas::Int, cos_theta_vec::Vector{Float64}, Oh::Float64)
    # First Block Row
    I_ntheta1 = Matrix{Float64}(I, n_thetas - 1, n_thetas - 1)
    block1 = hcat(I_ntheta1,
                  -delta_t * I_ntheta1,
                  zeros(n_thetas - 1, n_thetas + 1),
                  zeros(n_thetas - 1, 1),
                  zeros(n_thetas - 1, 1))

    # Second Block Row
    C_list = [(l - 1) * l * (l + 2) for l in 2:n_thetas]
    C_mat = Diagonal(C_list)

    D_list = [1 + delta_t * 2 * Oh * (l - 1) * (2l + 1) for l in 2:n_thetas]
    D_mat = Diagonal(D_list)

    E_list = [l for l in 2:n_thetas]
    E_mat = hcat(zeros(n_thetas - 1, 2), Diagonal(E_list))

    block2 = hcat(delta_t * C_mat,
                  D_mat,
                  delta_t * E_mat,
                  zeros(n_thetas - 1, 1),
                  zeros(n_thetas - 1, 1))

    # Combine first and second block rows
    M_mat = vcat(block1, block2)

    # Third and Fourth Block Rows (Contact conditions)
    FH_mat = zeros(n_thetas + 1, n_thetas + 1)
    for ind_angle in 1:(n_thetas + 1)
        for ind_poly in 1:(n_thetas + 1)
            FH_mat[ind_angle, ind_poly] = Legendre(cos_theta_vec[ind_angle], ind_poly - 1, n_thetas)
        end
    end

    F_mat = FH_mat[1:q, 3:end]  # 3:end because 1-based indexing (cols 2+)
    H_mat = FH_mat[(q + 1):end, :]

    G_vec = -1.0 ./ cos_theta_vec[1:q]
    G_mat = reshape(G_vec, q, 1)

    third_block_row = hcat(F_mat,
                           zeros(q, n_thetas - 1),
                           zeros(q, n_thetas + 1),
                           G_mat,
                           zeros(q, 1))

    fourth_block_row = hcat(zeros(n_thetas + 1 - q, n_thetas - 1),
                            zeros(n_thetas + 1 - q, n_thetas - 1),
                            H_mat,
                            zeros(n_thetas + 1 - q, 1),
                            zeros(n_thetas + 1 - q, 1))

    M_mat = vcat(M_mat, third_block_row, fourth_block_row)

    # Fifth Block Row (center of mass height)
    fifth_block_row = hcat(zeros(1, n_thetas - 1),
                           zeros(1, n_thetas - 1),
                           zeros(1, n_thetas + 1),
                           [1.0],
                           [-delta_t])

    M_mat = vcat(M_mat, fifth_block_row)

    # Sixth Block Row (center of mass velocity)
    K = zeros(1, n_thetas + 1)
    K[1, 2] = -1  # index 2 corresponds to index 1 in Python

    sixth_block_row = hcat(zeros(1, n_thetas - 1),
                           zeros(1, n_thetas - 1),
                           delta_t * K,
                           [0.0],
                           [1.0])

    M_mat = vcat(M_mat, sixth_block_row)

    return M_mat
end


function error(amplitudes_vec, h, q, n_thetas, cos_theta_vec)
    """
    Function that calculates the error of q contact points, which represents the difference between the vertical components of the contact points at k and k+1.

    Inputs:
    amplitudes_vec (vector): Vector of amplitudes for q contact points at k+1.
    h (float): Height of the center of mass at k+1.
    q (int): Number of guessed contact points.
    n_thetas (int): Total number of sampled angles.
    cos_theta_vec (vector): Vector of cos(theta) values for the sampled angles.

    Outputs:
    err (float): Error at k+1 with q contact points.
    """

    if q < 0
        return Inf
    end

    if q == 0
        all_verticals_list = Float64[]
        for ind_theta in 1:(n_thetas+1)
            cos_theta = cos_theta_vec[ind_theta] 
            height = (1 + sum(amplitudes_vec[l] * Legendre(cos_theta, l+2, n_thetas) for l in 1:(n_thetas-1))) * cos_theta # Calculate the vertical component of the contact point
            push!(all_verticals_list, height)
        end
        for vertical in all_verticals_list
            if vertical > h
                return Inf
            end
        end
        return 0.0
    end

    # Variables to store sums
    sum_q = 1.0
    sum_qp1 = 1.0

    # Extracting the cos(theta) values at q (= q-1 in list index) and q+1 (= q in list index)
    cos_theta_q = cos_theta_vec[q]
    cos_theta_q_plus_1 = cos_theta_vec[q+1]

    # Calculating the sums for the vertical components of the contact points because the sums are based on the Legendre polynomial properties
    for ind_poly in 1:(n_thetas-1)
        sum_q += amplitudes_vec[ind_poly] * Legendre(cos_theta_q, ind_poly+2, n_thetas)
        sum_qp1 += amplitudes_vec[ind_poly] * Legendre(cos_theta_q_plus_1, ind_poly+2, n_thetas)
    end

    # Calculating the vertical components of each point
    vertical_q = sum_q * cos_theta_q
    vertical_qp1 = sum_qp1 * cos_theta_q_plus_1

    # Regular error calculation
    err1 = abs(vertical_q - vertical_qp1)
    err2 = 0.0 # Err2 for checking whether droplet crosses surface (physically impossible)
    for ind_theta in (q+1):(n_thetas+1)
        rad = 1.0
        for ind_poly in 2:n_thetas
            rad += amplitudes_vec[ind_poly-1] * Legendre(cos_theta_vec[ind_theta], ind_poly, n_thetas)
        end
        if rad * cos_theta_vec[ind_theta] > h
            err2 = Inf
            break
        end
    end
    err = maximum([err1, err2]) # Return highest error

    return err
end

function try_q(amps_prev_vec::Vector{Float64}, amps_vel_prev_vec::Vector{Float64},
               h_prev::Float64, v_prev::Float64,
               q::Int, delta_t::Float64, n_thetas::Int,
               Bo::Float64, Oh::Float64)
    """
    Function that "tries" q contact points by solving the system of equations.

    Inputs:
    amps_prev_vec (vector): Vector of amplitudes at the previous time (k).
    amps_vel_prev_vec (vector): Vector of velocities of amplitudes at the previous time (k).
    h_prev (float): Height of the center of mass at the previous time (k).
    v_prev (float): Velocity of the center of mass at the previous time (k).
    q (int): Number of guessed contact points.
    delta_t (float): Time step (in non-dimensional time).
    n_thetas (int): Total number of sampled angles.
    Bo (float): Bond number, representing the ratio of gravitational forces to surface tension forces.
    Oh (float): Ohnesorge number, representing the ratio of viscous forces to inertial forces.

    Outputs:
    sol_vec (vector): Vector of the solution of matrix at q contact points at time k+1
    err (float): Error at k+1 with q contact points.
    """

    # Sampling angles
    roots, weights = gausslegendre(n_thetas)
    cos_theta_vec = vcat(1.0, reverse(roots))
    q_last = findfirst(x -> x <= 0, cos_theta_vec)

    if q < 0
        # Physically invalid case: return previous state + infinite error
        sol_vec = vcat(amps_prev_vec,
                       amps_vel_prev_vec,
                       zeros(n_thetas + 1),
                       [h_prev],
                       [v_prev])
        err = Inf

    elseif q == 0
        # No contact case
        RHS_modes_vec = vcat(amps_prev_vec, amps_vel_prev_vec)
        RHS_h_vec = [h_prev, v_prev - delta_t * Bo]

        # Extract submatrices
        full_mat = matrix(delta_t, q, n_thetas, cos_theta_vec, Oh)
        M_modes_mat = full_mat[1:(2n_thetas - 2), 1:(2n_thetas - 2)]
        M_h_mat = full_mat[end-1:end, end-1:end]

        # Solve
        sol_modes_vec = M_modes_mat \ RHS_modes_vec
        sol_h_vec = M_h_mat \ RHS_h_vec

        sol_vec = vcat(sol_modes_vec,
                       zeros(n_thetas + 1),
                       sol_h_vec)

        err = error(sol_modes_vec[1:(n_thetas - 1)],
                    sol_h_vec[1],
                    q, n_thetas, cos_theta_vec)

    elseif q <= q_last
        RHS_vec = vcat(amps_prev_vec,
                       amps_vel_prev_vec,
                       -ones(q),
                       zeros(n_thetas + 1 - q),
                       [h_prev],
                       [v_prev - delta_t * Bo])

        M_mat = matrix(delta_t, q, n_thetas, cos_theta_vec, Oh)
        sol_vec = M_mat \ RHS_vec

        err = error(sol_vec[1:(n_thetas - 1)],
                    sol_vec[end - 1],
                    q, n_thetas, cos_theta_vec)

    else
        # Invalid deformation assumption; return previous state + infinite error
        sol_vec = vcat(amps_prev_vec,
                       amps_vel_prev_vec,
                       zeros(n_thetas + 1),
                       [h_prev],
                       [v_prev])
        err = Inf
    end

    return sol_vec, err
end

function height_poles(all_amps_vec, h, n_thetas)

    """
    Function that evaluates the deformed droplet's north and south pole radii.

    Inputs:
    amps_prev_vec (vector): Vector of amplitudes at a single point in time.
    h (float): Height of the center of mass.
    n_thetas (int): Total number of sampled angles.

    Outputs:
    north_pole_radius (vector): Radius at the north pole.
    south_pole_radius (vector): Radius at the south pole.

    """

    #Setting undeformed radii
    north_pole_radius = 1.0
    south_pole_radius = 1.0

    #Adding deformations from the Legendre modes
    for i in 2:(n_thetas-1)
        north_pole_radius += all_amps_vec[i-1] * Legendre(-1, i, n_thetas)
        south_pole_radius += all_amps_vec[i-1] * Legendre(1, i, n_thetas)
    end

    #Obtaining height
    north_pole_height = north_pole_radius * 1.0 + h
    south_pole_height = south_pole_radius * -1.0 + h

    return [north_pole_height, south_pole_height]
end

function min_north_pole_height(all_north_poles_list, all_times_list)
    min_north_pole, min_north_pole_ind = findmin(all_north_poles_list)
    time_of_min_north_pole = all_times_list[min_north_pole_ind]
    return [min_north_pole, time_of_min_north_pole]
end

function max_radius_at_each_t(all_amps_vec, n_thetas, theta_vec)
    """
    Function that calculates the maximum radius of the droplet at a single point in time.

    Inputs:
    amps_prev_vec (vector): Vector of amplitudes at a single point in time.

    Outputs:
    radius_of_largest_deformation (vector): Radius of the angle, where largest radius is achieved.

    """
    radii = ones(n_thetas)  # Initial undeformed radii

    for i in 2:(n_thetas - 1)
        for ind_angles in 1:n_thetas
            radii[ind_angles] += all_amps_vec[i - 1] * Legendre(cos(theta_vec[ind_angles]), i, n_thetas)
        end
    end

    return maximum(radii)
end

function max_radius_over_t(all_max_radii, contact_time_index, all_times_list)
    idx = Int(floor(contact_time_index))  # or use `round`, depending on desired behavior
    relevant_max_radii = all_max_radii[1:idx]
    max_radius, max_radius_index = findmax(relevant_max_radii)
    time_of_max_radius = all_times_list[max_radius_index]
    return [max_radius, time_of_max_radius]
end

function alpha_coefficient_of_restitution(all_h_list, all_v_list, all_m_list, all_time_list, Bo, We)
    final_velocity, final_height, contact_time, index = nothing, nothing, nothing, nothing

    for i in 2:(length(all_m_list)-1)
        if all_m_list[i] == 0 && all_m_list[i-1] > 0
            index = i
            break
        end
    end

    if index === nothing
        return [nothing, nothing, nothing]
    end

    final_velocity = (all_v_list[index] + all_v_list[index-1]) / 2
    final_height = (all_h_list[index] + all_h_list[index-1]) / 2
    contact_time = (all_time_list[index] + all_time_list[index-1]) / 2
    last_time_in_contact = index

    coeff_rest_squared = ((Bo * (final_height - 1)) + 0.5 * final_velocity^2) / (0.5 * We)
    coeff_rest_squared_float = sqrt(abs(coeff_rest_squared))

    return [coeff_rest_squared_float, contact_time, last_time_in_contact]
end

function center_coefficient_of_restitution(all_v_list, all_m_list)
    final_velocity, initial_velocity, index = nothing, nothing, nothing

    for ind in 1:length(all_m_list)-1
        if all_m_list[ind] == 0 && all_m_list[ind+1] != 0
            initial_velocity = (all_v_list[ind] + all_v_list[ind+1]) / 2
            break
        end
    end

    for i in 2:(length(all_m_list)-1)
        if all_m_list[i] == 0 && all_m_list[i-1] > 0
            index = i
            break
        end
    end

    if index === nothing
        return nothing
    end

    final_velocity = (all_v_list[index] + all_v_list[index-1]) / 2
    coef_rest = -final_velocity / initial_velocity

    return coef_rest
end

function maximum_contact_radius(all_amps_list, all_m_list, all_times_list, n_thetas, theta_vec)
    contact_radii = Float64[]

    for time in eachindex(all_amps_list)
        m = all_m_list[time]
        if m >= length(theta_vec)  # guard against overflow
            push!(contact_radii, 0.0)
            continue
        end

        radii_vec = ones(2)

        for i in 2:(n_thetas - 1)
            radii_vec[1] += all_amps_list[time][i - 1] * Legendre(cos(theta_vec[m + 1]), i, n_thetas)
            radii_vec[2] += all_amps_list[time][i - 1] * Legendre(cos(theta_vec[m + 2]), i, n_thetas)
        end

        x_1 = radii_vec[1] * sin(theta_vec[m + 1])
        x_2 = radii_vec[2] * sin(theta_vec[m + 2])
        push!(contact_radii, (x_1 + x_2) / 2)
    end

    max_contact_radius, max_idx = findmax(contact_radii)
    time_of_max_radius = all_times_list[max_idx]
    return [max_contact_radius, time_of_max_radius]
end

function max_radial_projection(all_amps_list, n_thetas_int, all_times_list, theta_vec)
    all_amps_array = all_amps_list
    max_horizontal_projection = []

    for time_ind in eachindex(all_amps_array)
        all_horiz_comps = []
        thetas = vcat(theta_vec, π)
        radii = ones(length(thetas))

        for ind_amplitude in 2:(n_thetas_int - 1)
            for ind_angles in 1:n_thetas_int
                radii[ind_angles] += all_amps_array[time_ind][ind_amplitude - 1] * Legendre(cos(thetas[ind_angles]), ind_amplitude, n_thetas_int)
            end
        end

        for i in 1:length(thetas)
            push!(all_horiz_comps, radii[i] * sin(thetas[i]))
        end

        push!(max_horizontal_projection, maximum(all_horiz_comps))
    end

    max_of_radial_projections, max_radial_projection_index = findmax(max_horizontal_projection)
    time_of_max_view = all_times_list[max_radial_projection_index]

    return [max_of_radial_projections, time_of_max_view]
end

function min_side_height(all_amps_list, n_thetas_int, all_times_list, all_h_list, theta_vec)
    all_amps_array = all_amps_list
    highest_side_view = []

    for time_ind in eachindex(all_amps_array)
        all_heights = []
        thetas = vcat(theta_vec, π)
        radii = ones(length(thetas))

        for ind_amplitude in 2:(n_thetas_int - 1)
            for ind_angles in 1:n_thetas_int
                radii[ind_angles] += all_amps_array[time_ind][ind_amplitude - 1] * Legendre(cos(thetas[ind_angles]), ind_amplitude, n_thetas_int)
            end
        end

        for i in 1:length(thetas)
            y = all_h_list[time_ind] - (radii[i] * cos(thetas[i]))
            push!(all_heights, y)
        end

        push!(highest_side_view, maximum(all_heights))
    end

    min_of_highest_side_view, min_side_view_index = findmin(highest_side_view)
    time_of_min_view = all_times_list[min_side_view_index]

    return [min_of_highest_side_view, time_of_min_view]
end