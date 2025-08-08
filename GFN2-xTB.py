'''
References:
[GFN1] : https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.7b00118
[GFN2] : pubs.acs.org/doi/pdf/10.1021/acs.jctc.8b01176

'''
from slater_constants import gaussian_exponents, gaussian_linear_coeficients, slater_exponents, number_of_gaussians
from file_formats import parse_coord_file, parse_xyz_file
from D4_constants import covalent_radii as D4_covalent_radii
from math import sqrt, e, pi, log10
from GFN2_constants import number_of_subshells,angular_momentum_of_subshell,reference_occupations,dimensional_components_of_angular_momentum, effective_nuclear_charge, repulsion_alpha, H_CN, self_energy,electro_negativity, Huckel_covalent_radii,k_poly,K_AB,principal_quantum_number


h2o_coord_file = '''
$coord
 0.00000000000000      0.00000000000000     -0.73578586109551      o
 1.44183152868459      0.00000000000000      0.36789293054775      h
-1.44183152868459      0.00000000000000      0.36789293054775      h
$end
'''
caffine_xyz_file = '''
24

C          1.07317        0.04885       -0.07573
N          2.51365        0.01256       -0.07580
C          3.35199        1.09592       -0.07533
N          4.61898        0.73028       -0.07549
C          4.57907       -0.63144       -0.07531
C          3.30131       -1.10256       -0.07524
C          2.98068       -2.48687       -0.07377
O          1.82530       -2.90038       -0.07577
N          4.11440       -3.30433       -0.06936
C          5.45174       -2.85618       -0.07235
O          6.38934       -3.65965       -0.07232
N          5.66240       -1.47682       -0.07487
C          7.00947       -0.93648       -0.07524
C          3.92063       -4.74093       -0.06158
H          0.73398        1.08786       -0.07503
H          0.71239       -0.45698        0.82335
H          0.71240       -0.45580       -0.97549
H          2.99301        2.11762       -0.07478
H          7.76531       -1.72634       -0.07591
H          7.14864       -0.32182        0.81969
H          7.14802       -0.32076       -0.96953
H          2.86501       -5.02316       -0.05833
H          4.40233       -5.15920        0.82837
H          4.40017       -5.16929       -0.94780
'''
H = 0
He = 1
x = 0
y = 1
z = 2

def euclidean_distance_squared(position_A: list[float], position_B: list[float]) -> float:
    assert len(position_A) == 3 and len(position_B) == 3, f"postions should have 3 coordiantes exactly! not {len(position_A)}, {len(position_B)}"
    x_diff = position_A[x]-position_B[x]
    y_diff = position_A[y]-position_B[y]
    z_diff = position_A[z]-position_B[z]
    return x_diff**2 + y_diff**2 + z_diff**2

def euclidean_distance(position_A: list[float], position_B: list[float]) -> float:
    return sqrt(euclidean_distance_squared(position_A, position_B))

def get_orbitals(atoms: list[int]) -> list[tuple[int]]:
    orbitals = []
    for atom_idx,atom in enumerate(atoms):
        for subshell in range(number_of_subshells[atom]):
            l = angular_momentum_of_subshell[atom][subshell] 
            for orbital in range(l*2+1):
                orbitals.append((atom_idx,atom,subshell,orbital))
    return orbitals

def get_orbitals_cartesian(atoms: list[int]) -> list[tuple[int]]:
    orbitals = []
    for atom_idx,atom in enumerate(atoms):
        for subshell in range(number_of_subshells[atom]):
            l = angular_momentum_of_subshell[atom][subshell] 
            for orbital in range(((l+1)*(l+2)//2)):
                orbitals.append((atom_idx,atom,subshell,orbital))
    return orbitals

def get_square_matrix(n: int, default_element=0.0) -> list[list[float]]:
    matrix = []
    for _ in range(n):
        row = []
        for _ in range(n):
            row.append(default_element)
        matrix.append(row)
    return matrix

def density_initial_guess(atoms: list[int]):
    orbitals = get_orbitals(atoms)
    occs = get_square_matrix(len(orbitals))
    for idx,(_,atom,subshell,orbital) in enumerate(orbitals):
        l = angular_momentum_of_subshell[atom][subshell] 
        orbitals_in_subshell = l*2+1 
        electrons_in_subshell = reference_occupations[atom][subshell]
        electrons_per_orbital = electrons_in_subshell/orbitals_in_subshell
        occs[idx][idx] = electrons_per_orbital
    return occs

def build_overlap_dipol_quadrupol(element_ids, positions, intcut): 
    pyramidal_numbers = [
        [     1     ],
        [    1,1    ],
        [   1,2,1   ],
        [  1,3,3,1  ],
        [ 1,4,6,4,1 ]
    ]
    dfactorial = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0]
    trafo = [
        [1], # s
        [1,1,1], # p
        [1,1,1,sqrt(3.),sqrt(3.),sqrt(3.)], # d
        [1.0, 1.0, 1.0, sqrt(5.0), sqrt(5.0), sqrt(5.0), sqrt(5.0), sqrt(5.0), sqrt(5.0), sqrt(15.0)] # f
    ]
    x = 0
    y = 1
    z = 2
    xx = 0
    yy = 1
    zz = 2
    xy = 3
    xz = 4
    yz = 5
    orbitals = get_orbitals(atoms)
    slater_0th_moment = get_square_matrix(len(orbitals))
    slater_1st_moment = get_square_matrix(len(orbitals), default_element=[0,0,0])
    slater_2nd_moment = get_square_matrix(len(orbitals), default_element=[0,0,0,0,0,0])
    for orbital_idx_A,(atom_idx_A,atom_A,subshell_A,orbital_A) in enumerate(orbitals):
        l_A = angular_momentum_of_subshell[atom_A][subshell_A]
        number_of_gaussians_A = number_of_gaussians[atom_A][subshell_A]
        slater_exponent_A = slater_exponents[atom_A][subshell_A]
        n_A = principal_quantum_number[atom_A][subshell_A]
        l_A_dims = dimensional_components_of_angular_momentum[l_A][orbital_A]
        position_A = positions[atom_idx_A]
        for orbital_idx_B,(atom_idx_B,atom_B,subshell_B,orbital_B) in enumerate(orbitals):
            l_B = angular_momentum_of_subshell[atom_B][subshell_B]
            number_of_gaussians_B = number_of_gaussians[atom_B][subshell_B]
            slater_exponent_B = slater_exponents[atom_B][subshell_B]
            n_B = principal_quantum_number[atom_B][subshell_B]
            l_B_dims = dimensional_components_of_angular_momentum[l_B][orbital_B]
            position_B = positions[atom_idx_B]
            euclidean_distance_AB_squared = (position_A[0]-position_B[0])**2+(position_A[1]-position_B[1])**2+(position_A[2]-position_B[2])**2
            if euclidean_distance_AB_squared > 2000:
                continue
            if orbital_idx_A == orbital_idx_B:
                slater_0th_moment[orbital_idx_A][orbital_idx_B] = 1
                continue
            for gaussian_A in range(number_of_gaussians_A):
                gaussian_exponent_A = gaussian_exponents[number_of_gaussians_A][n_A][l_A][gaussian_A]
                unnormalized_linear_coeficient_A = gaussian_linear_coeficients[number_of_gaussians_A][n_A][l_A][gaussian_A]*slater_exponent_A**2
                normalization_factor_A = ((2/pi)*gaussian_exponent_A)**(3/4) * sqrt(4*gaussian_exponent_A)**l_A/ sqrt(dfactorial[l_A])
                linear_coeficient_A = unnormalized_linear_coeficient_A*normalization_factor_A*trafo[l_A][orbital_A]
                for gaussian_B in range(number_of_gaussians_B):
                    gaussian_exponent_B = gaussian_exponents[number_of_gaussians_B][n_B][l_B][gaussian_B]
                    unnormalized_linear_coeficient_B = gaussian_linear_coeficients[number_of_gaussians_B][n_B][l_B][gaussian_B]*slater_exponent_B**2
                    normalization_factor_B = ((2/pi)*gaussian_exponent_B)**(3/4) * sqrt(4*gaussian_exponent_B)**l_B/ sqrt(dfactorial[l_B])
                    linear_coeficient_B = unnormalized_linear_coeficient_B*normalization_factor_B*trafo[l_B][orbital_B]
                    combined_gaussian_exponent = gaussian_exponent_A+gaussian_exponent_B
                    gaussian_product_exponent = (gaussian_exponent_A*gaussian_exponent_B)/combined_gaussian_exponent
                    estimate = euclidean_distance_AB_squared*gaussian_product_exponent 
                    if estimate > intcut:
                        continue
                    gaussian_product_center = [
                            (gaussian_exponent_A*position_A[x]+gaussian_exponent_B*position_B[x])/combined_gaussian_exponent,
                            (gaussian_exponent_A*position_A[y]+gaussian_exponent_B*position_B[y])/combined_gaussian_exponent,
                            (gaussian_exponent_A*position_A[z]+gaussian_exponent_B*position_B[z])/combined_gaussian_exponent
                    ]
                    gaussian_overlap = e**(-euclidean_distance_AB_squared*gaussian_product_exponent) * (pi / combined_gaussian_exponent)**(3/2)
                    prefactor = gaussian_overlap * linear_coeficient_A * linear_coeficient_B

                    gaussian_0th_moment_components = [0,0,0]
                    gaussian_1st_moment_components = [0,0,0]
                    gaussian_2nd_moment_components = [0,0,0]
                    for dimension in [x,y,z]:
                        l_A_dim = l_A_dims[dimension]
                        l_B_dim = l_B_dims[dimension]
                        max_l = max(l_A_dim, l_B_dim)
                        min_l = min(l_A_dim, l_B_dim)
                        v_A = [0]*(max_l+1)
                        v_B = [0]*(max_l+1)
                        v_A[l_A_dim] = 1.0
                        v_B[l_B_dim] = 1.0
                        product_center_relative_to_position_A = gaussian_product_center[dimension]-position_A[dimension]
                        product_center_relative_to_position_B = gaussian_product_center[dimension]-position_B[dimension]

                        # horizontal shifts
                        for i,pyr in enumerate(pyramidal_numbers[l_A_dim][:-1]):
                            v_A[i] += pyr*(product_center_relative_to_position_A**(l_A_dim-i))*v_A[l_A_dim]
                        for i,pyr in enumerate(pyramidal_numbers[l_B_dim][:-1]):
                            v_B[i] += pyr*(product_center_relative_to_position_B**(l_B_dim-i))*v_B[l_B_dim]

                        # form product
                        product = [0]*(max_l+min_l+1)
                        for i in range(min_l+1):
                            product[i+i] += v_A[i]*v_B[i]
                            for j in range(i+1,max_l+1):
                                product[i+j] += v_A[i] * v_B[j] + v_A[j] * v_B[i]

                        # 1d gaussian overlap
                        gaussian_overlap_dim_component = [0]*(l_A_dim+l_B_dim+3)
                        for l in range(0,(l_A_dim+l_B_dim+3),2):
                            half_l = l//2
                            gamma = 0.5/combined_gaussian_exponent
                            gaussian_overlap_dim_component[l] = gamma**half_l*dfactorial[half_l]
                        # 0th 1st and 2nd gaussian moment
                        for summed_l, prod in enumerate(product):
                            gaussian_0th_moment_components[dimension] += (
                                1*gaussian_product_center[dimension]**0 * gaussian_overlap_dim_component[summed_l+0]
                            ) * prod
                            gaussian_1st_moment_components[dimension] += (
                                1*gaussian_product_center[dimension]**1 * gaussian_overlap_dim_component[summed_l+0] + 
                                1*gaussian_product_center[dimension]**0 * gaussian_overlap_dim_component[summed_l+1]
                            ) * prod
                            gaussian_2nd_moment_components[dimension] += (
                                1*gaussian_product_center[dimension]**2 * gaussian_overlap_dim_component[summed_l+0] + 
                                2*gaussian_product_center[dimension]**1 * gaussian_overlap_dim_component[summed_l+1] + 
                                1*gaussian_product_center[dimension]**0 * gaussian_overlap_dim_component[summed_l+2]
                            ) * prod
                    
                    # contribute to the slater moments
                    slater_0th_moment[orbital_idx_A][orbital_idx_B] += prefactor * gaussian_0th_moment_components[x] * gaussian_0th_moment_components[y] * gaussian_0th_moment_components[z]

                    slater_1st_moment[orbital_idx_A][orbital_idx_B][x] += prefactor * gaussian_1st_moment_components[x] * gaussian_0th_moment_components[y] * gaussian_0th_moment_components[z]
                    slater_1st_moment[orbital_idx_A][orbital_idx_B][y] += prefactor * gaussian_0th_moment_components[x] * gaussian_1st_moment_components[y] * gaussian_0th_moment_components[z]
                    slater_1st_moment[orbital_idx_A][orbital_idx_B][z] += prefactor * gaussian_0th_moment_components[x] * gaussian_0th_moment_components[y] * gaussian_1st_moment_components[z]

                    slater_2nd_moment[orbital_idx_A][orbital_idx_B][xx] += prefactor * gaussian_2nd_moment_components[x] * gaussian_0th_moment_components[y] * gaussian_0th_moment_components[z]
                    slater_2nd_moment[orbital_idx_A][orbital_idx_B][yy] += prefactor * gaussian_0th_moment_components[x] * gaussian_2nd_moment_components[y] * gaussian_0th_moment_components[z]
                    slater_2nd_moment[orbital_idx_A][orbital_idx_B][zz] += prefactor * gaussian_0th_moment_components[x] * gaussian_0th_moment_components[y] * gaussian_2nd_moment_components[z]
                                                       
                    slater_2nd_moment[orbital_idx_A][orbital_idx_B][xy] += prefactor * gaussian_1st_moment_components[x] * gaussian_1st_moment_components[y] * gaussian_0th_moment_components[z]
                    slater_2nd_moment[orbital_idx_A][orbital_idx_B][xz] += prefactor * gaussian_1st_moment_components[x] * gaussian_0th_moment_components[y] * gaussian_1st_moment_components[z]
                    slater_2nd_moment[orbital_idx_A][orbital_idx_B][yz] += prefactor * gaussian_0th_moment_components[x] * gaussian_1st_moment_components[y] * gaussian_1st_moment_components[z]
    return slater_0th_moment, slater_1st_moment, slater_2nd_moment
# [GFN1, eq. 13] [GFN2, eq. 9]
def get_GFN2_repulsion_energy(atoms: list[int], positions : list[list[float]]) -> float:
    sum = 0
    for idx_A, (atom_A, position_A) in enumerate(zip(atoms, positions)):
        for idx_B, (atom_B, position_B) in enumerate(zip(atoms, positions)):
            if idx_A == idx_B:
                # The distance between an atom and it self will be zero resulting in division by zero if we do the calculation. 
                # Also an atom does not repulse itself!
                continue
            k_f = 3/2
            if atom_A in [H, He] and atom_B in [H, He]:
                k_f = 1
            R_AB = euclidean_distance(position_A, position_B)
            alpha_A = repulsion_alpha[atom_A]
            alpha_B = repulsion_alpha[atom_B]
            Y_A = effective_nuclear_charge[atom_A]
            Y_B = effective_nuclear_charge[atom_B]
            sum += ((Y_A*Y_B)/R_AB) * e ** ( - sqrt(alpha_A*alpha_B) * R_AB**k_f )
    E_rep = (1/2)*sum
    return E_rep

def get_coordination_numbers(atoms: list[int], positions: list[list[float]]) -> list[float]:
    Huckel_coordination_numbers = []
    for idx_A, atom_A in enumerate(atoms):
        CN_A = 0
        R_A = positions[idx_A]
        Rcov_A = D4_covalent_radii[atom_A]
        for idx_B, atom_B in enumerate(atoms):
            if idx_A == idx_B:
                continue
            R_B = positions[idx_B]
            Rcov_B = D4_covalent_radii[atom_A]
            R_AB = euclidean_distance(R_A,R_B)
            CN_A += (1 + e ** (-10*((4*(Rcov_A+Rcov_B))/(3*R_AB)-1)))**(-1) * (1 + e ** (-20*((4*(Rcov_A+Rcov_B+2))/(3*R_AB)-1)))**(-1)
        Huckel_coordination_numbers.append(CN_A)
    return Huckel_coordination_numbers

def huckel_matrix(atoms: list[int], positions: list[list[float]], overlap: list[list[float]]) -> list[list[float]]:
    orbitals = get_orbitals(atoms)
    H_EHT = get_square_matrix(len(orbitals))
    CN = get_coordination_numbers(atoms,positions)
    for orbital_idx, (atom_idx,atom,subshell,orbital) in enumerate(orbitals):
        CN_A = CN[atom_idx]
        H_A = self_energy[atom][subshell] # constant
        H_CN_A = H_CN[atom][subshell] # constant
        H_EHT[orbital_idx][orbital_idx] = H_A - H_CN_A*CN_A

    for idx_A,(atom_A_idx,atom_A,subshell_A,_) in enumerate(orbitals):
        l_A = angular_momentum_of_subshell[atom_A][subshell_A]
        EN_A = electro_negativity[atom_A]
        R_A = positions[atom_A_idx]
        Rcov_A = Huckel_covalent_radii[atom_A]
        k_poly_A = k_poly[atom_A][l_A]
        for idx_B,(atom_B_idx,atom_B,subshell_B,_) in enumerate(orbitals):
            if idx_A == idx_B:
                continue
            l_B = angular_momentum_of_subshell[atom_B][subshell_B]
            EN_B = electro_negativity[atom_B]
            R_B = positions[atom_B_idx]
            Rcov_B = Huckel_covalent_radii[atom_B]
            k_poly_B = k_poly[atom_B][l_B]
            K_ll = K_AB[l_A][l_B]
            delta_EN_squared = (EN_A-EN_B)**2
            k_EN = 0.02
            X = 1+k_EN*delta_EN_squared
            R_AB = euclidean_distance(R_A,R_B) 
            Rcov_AB = Rcov_A + Rcov_B 
            PI = (1+k_poly_A*sqrt(R_AB/Rcov_AB)) * (1+k_poly_B*sqrt(R_AB/Rcov_AB))
            slater_exp_A = slater_exponents[atom_A][l_A]
            slater_exp_B = slater_exponents[atom_B][l_B]
            Y = sqrt((2*sqrt(slater_exp_A*slater_exp_B)) / (slater_exp_A+slater_exp_B))
            H_nn = H_EHT[idx_A][idx_A]
            H_mm = H_EHT[idx_B][idx_B]
            S_nm = overlap[idx_A][idx_B]
            H_EHT[idx_A][idx_B] = K_ll*(1/2)*(H_nn+H_mm)*S_nm*Y*X*PI
    return H_EHT

def get_GFN2_Huckel_energy(P:list[list[int]], H:list[list[int]]):
    E_EHT = 0
    for shell_mu in range(len(P)):
        for shell_nu in range(len(P[0])):
            E_EHT += P[shell_nu][shell_mu] * H[shell_mu][shell_nu]
    return E_EHT

def get_error_squared_2d(A:list[list[int]], B:list[list[int]]):
    error = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            error += (A[i][j] - B[i][j])**2
    return error
def mulliken_population_analysis(*args):
    return 0,0
def build_SDQH0(atoms: list[int], positions : list[list[float]]):
    orbitals = get_orbitals(atoms)
    
    S = get_square_matrix(len(orbitals))
    for idx in range(len(orbitals)):
        S[idx][idx] = 1
    return S,0,0,0
def get_D4Prime_energy(*args):
    return 0
def get_GFN2_AES_energy(*args):
    return 0
def get_GFN_IES_energy(*args):
    return 0
def eigh(*args):
    return 0
def compute_density_matrix_from_fermi(*args):
    return 0
def get_GFN2_energy(atoms: list[int], positions : list[list[float]]) -> float:
    assert len(atoms) == len(positions), f"number of atoms ({len(atoms)}) and positions ({len(positions)}) should be the same!"
    P = density_initial_guess(atoms)
    accuracy = 1.0
    integral_cutoff = max(20.0, 25.0-10.0*log10(accuracy))
    S, D, Q = build_overlap_dipol_quadrupol(atoms, positions, integral_cutoff)
    H0_EHT = huckel_matrix(atoms, positions, S)
    charges_per_atom, charges_per_shell = mulliken_population_analysis(P,atoms)
    repulsion_energy = get_GFN2_repulsion_energy(atoms, positions)
    print(repulsion_energy)
    dispersion_energy = get_D4Prime_energy(..., charges_per_atom, positions)
    huckel_energy = get_GFN2_Huckel_energy(P, H0_EHT)
    print(huckel_energy)
    anisotropic_energy = get_GFN2_AES_energy(charges_per_shell, S, D, Q, positions)
    isotropic_energy = get_GFN_IES_energy(charges_per_atom, positions)
    total_energy = repulsion_energy +\
                    dispersion_energy +\
                    huckel_energy +\
                    anisotropic_energy +\
                    isotropic_energy
    return total_energy
    #energy_converged = False
    #densities_converged = False
    #while not (energy_converged and densities_converged):
    #    e = eigh(H, S) # HC=SCe 
    #    P_next = compute_density_matrix_from_fermi(e,...)
    #    charges_per_atom_next, charges_per_shell_next = mulliken_population_analysis(P_next,atoms)
    #    # ...
    #    # update everything with new charges
    #    # ...
    #    energy_converged = (total_energy-total_energy_next)**2 < tolerance
    #    densities_converged = get_error_squared_2d(P,P_next) < tolerance

if __name__ == "__main__" :
    atoms, positions = parse_xyz_file(caffine_xyz_file)
    print(get_GFN2_energy(atoms, positions))

