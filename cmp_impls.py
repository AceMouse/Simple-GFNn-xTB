import numpy as np
import glob
import argparse
import os
import math
from GFN2_xTB import build_overlap_dipol_quadrupol, get_coordination_numbers

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compare original xtb results with our own python implementation.")
parser.add_argument("directory", help="Path to the directory containing the binary files from xtb containing the arguments and result.")
args = parser.parse_args()

directory = args.directory
directory = os.path.abspath(directory)

def assert_compare(val, val_res, val_name, fn_name):
    print("\033[1;31m")
    print("Fortran:")
    print(f"{val_name}: ", val_res)

    print()

    print("Python:")
    print(f"{val_name}: ", val)

    print()

    print(f"[{fn_name}] {val_name} do not match")
    exit(1)

def is_equal(val, val_res, val_name, fn_name):
    eq = val == val_res
    if (not eq):
        assert_compare(val, val_res, val_name, fn_name)
    return True

def is_close(val, val_res, val_name, fn_name, rel_tol=1e-9, abs_tol=0.0):
    close = math.isclose(val, val_res, rel_tol=rel_tol, abs_tol=abs_tol)
    if not close:
        assert_compare(val, val_res, val_name, fn_name)
    return True

def is_array_equal(val, val_res, val_name, fn_name):
    eq = np.array_equal(val, val_res)
    if (not eq):
        assert_compare(val, val_res, val_name, fn_name)
    return True

def is_array_close(val, val_res, val_name, fn_name):
    eq = np.allclose(val, val_res)
    if (not eq):
        assert_compare(val, val_res, val_name, fn_name)
    return True

def compare(fo, py, label, force_equal=False, rtol=1e-02):
    fo = np.array(fo)
    py = np.array(py)
    if py.shape != fo.shape:
        print(f"{label}:")
        print(f"\tShape missmatch\nPython: {py.shape}\nFortran: {fo.shape}")
        return False
    equal = np.array_equal(py,fo)
    if equal:
        return True
    close = np.allclose(py,fo,rtol=rtol)
    if close and not force_equal:
        return True
    print("\033[0;31m", end='')
    if close:
        print(f"{label}: Is close but not equal!")
    else:
        print(f"{label}: Is not close!")
    print("\033[0;0m", end='')
    print(f"\tPython: \n{py}")
    print(f"\tFortran: \n{fo}")
    diff = ""
    diff_no_numbers = ""
    diff_multiple = ""
    diff_arr = py - fo
    max_before_newline = 800
    max_before_newline_no_numbers = 800
    max_before_newline_multiple = 800
    shape = py.shape
    last_index = ()
    count = 0
    closing = py.ndim
    different_idx = []
    for idx in np.ndindex(shape):
        if py.ndim > 1:
            if last_index:
                for dim in range(0,py.ndim-1):
                    closing += idx[dim] != last_index[dim]
                if closing > 0:
                    diff += f"{']'*closing}"
                    diff_no_numbers += f"{']'*closing}"
                    diff_multiple += f"{']'*closing}"
                    count = 0
            last_index = idx
        diff += f'{"\n"*(closing>0)}{" "*(py.ndim-closing)}{"["*closing}'
        diff_no_numbers += f'{"\n"*(closing>0)}{" "*(py.ndim-closing)}{"["*closing}'
        diff_multiple += f'{"\n"*(closing>0)}{" "*(py.ndim-closing)}{"["*closing}'
        if count == max_before_newline:
            diff += "\n"
            diff += " "*py.ndim
            count = 0
        if count == max_before_newline_no_numbers:
            diff_no_numbers += "\n"
            diff_no_numbers += " "*py.ndim
            count = 0
        if count == max_before_newline_multiple:
            diff_multiple += "\n"
            diff_multiple += " "*py.ndim
            count = 0
        closing = 0
        diff_val = diff_arr[idx]
        color = 0
        color_multiple = 0
        no_number = " "
        multiple = "        "
        if not np.allclose(fo[idx],py[idx],rtol=rtol):
            different_idx += [idx]
            if py[idx] != 0:
                mult_val = fo[idx]/py[idx]
            else:
                mult_val = 0
            multiple = f"{mult_val: >8.5f}"
            color_multiple = 32 if mult_val>0 else 31
            color = 32
            no_number = "+"
            if diff_val < 0:
                color = 31
                no_number = "-"
        pos_space = " " if py[idx] >= 0 else ""
        diff += f"\033[0;{color}m{f'{pos_space}{py[idx]} ': <32}\033[0;0m"
        diff_no_numbers += f"\033[0;{color}m{no_number} \033[0;0m"
        diff_multiple += f"\033[0;{color_multiple}m{multiple} \033[0;0m"
        #diff += f"{f'{pos_space}{py[idx]} ': <32} "
        #diff_no_numbers += f"{no_number} "
        #diff_multiple += f"{multiple} "

        # Add line breaks when a major axis changes (e.g., new row in 2D, new matrix in 3D)
        count += 1
    diff += "]"*py.ndim
    diff_no_numbers += "]"*py.ndim
    diff_multiple += "]"*py.ndim
    print(f"\tDiff: \n{diff}")
    print(f"\tDiff pattern: \n{diff_no_numbers}")
    print(f"\tDiff multiple: \n{diff_multiple}")
    print("\tDiff idxs (first 10): \n",list(different_idx)[:10])
    return False




        
def test_build_overlap_dipol_quadrupol():
    for i, file_path in enumerate(glob.glob(f'{directory}/build_SDQH0/*.bin')):
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            def read_reals(n=1):
                return np.fromfile(f, dtype=np.float64, count=n)

            nat = read_ints(1)[0]
            at1 = read_ints(1)[0]
            at = np.fromfile(f, dtype=np.int32, count=at1)
            nbf = read_ints(1)[0]
            nao = read_ints(1)[0]
            xyz1, xyz2 = read_ints(2)
            xyz = np.fromfile(f, dtype=np.float64, count=xyz1 * xyz2).reshape((xyz2, xyz1))
            trans1, trans2 = read_ints(2)
            trans = np.fromfile(f, dtype=np.float64, count=trans1 * trans2).reshape((trans2, trans1))
            selfEnergy1, selfEnergy2 = read_ints(2)
            selfEnergy = np.fromfile(f, dtype=np.float64, count=selfEnergy1 * selfEnergy2).reshape((selfEnergy2, selfEnergy1))
            intcut = read_reals(1)[0]
            caoshell1, caoshell2 = read_ints(2)
            caoshell = np.fromfile(f, dtype=np.int32, count=caoshell1 * caoshell2).reshape((caoshell2, caoshell1))
            saoshell1, saoshell2 = read_ints(2)
            saoshell = np.fromfile(f, dtype=np.int32, count=saoshell1 * saoshell2).reshape((saoshell2, saoshell1))
            nprim1 = read_ints(1)[0]
            nprim = np.fromfile(f, dtype=np.int32, count=nprim1)
            primcount1 = read_ints(1)[0]
            primcount = np.fromfile(f, dtype=np.int32, count=primcount1)
            alp1 = read_ints(1)[0]
            alp = np.fromfile(f, dtype=np.float64, count=alp1)
            cont1 = read_ints(1)[0]
            cont = np.fromfile(f, dtype=np.float64, count=cont1)


            sint_res1, sint_res2 = read_ints(2)
            sint_res = np.fromfile(f, dtype=np.float64, count=sint_res1 * sint_res2).reshape((sint_res2, sint_res1))
            dpint_res1, dpint_res2, dpint_res3 = read_ints(3)
            dpint_res = np.fromfile(f, dtype=np.float64, count=dpint_res1 * dpint_res2 * dpint_res3).reshape((dpint_res3, dpint_res2, dpint_res1))
            qpint_res1, qpint_res2, qpint_res3 = read_ints(3)
            qpint_res = np.fromfile(f, dtype=np.float64, count=qpint_res1 * qpint_res2 * qpint_res3).reshape((qpint_res3, qpint_res2, qpint_res1))
            H0_res1 = read_ints(1)[0]
            H0_res = np.fromfile(f, dtype=np.float64, count=H0_res1)
            H0_noovlp_res1 = read_ints(1)[0]
            H0_noovlp_res = np.fromfile(f, dtype=np.float64, count=H0_noovlp_res1)

            S, D, Q = build_overlap_dipol_quadrupol(at-1, xyz, intcut)
            compare(sint_res,  S, "[build_overlap_dipol_quadrupol] S")
            #compare(dpint_res, D, "[build_overlap_dipol_quadrupol] D")
            #compare(qpint_res, Q, "[build_overlap_dipol_quadrupol] Q")
        

    print("\033[0;32m", end='')
    print("matches! [build_overlap_dipol_quadrupol]")
    print("\033[0;0m", end='')



def test_coordination_number():
    fn_name = "coordination_number"
    for i, file_path in enumerate(glob.glob(f'{directory}/{fn_name}/*.bin')):
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            m = read_ints(1)[0]
            cn_res = np.fromfile(f, dtype=np.float64, count=m)

            from file_formats import parse_xyz_file
            with open("./data/caffeine.xyz", "r") as f:
                data = f.read()
                element_ids, positions = parse_xyz_file(data)
            #cn = GFN2_coordination_numbers_np(element_ids, positions)
            cn = get_coordination_numbers(element_ids, positions)

            #is_array_equal(cn, cn_res, "coordination numbers", fn_name)
            if not compare(cn_res,cn, "coordination numbers"):
                quit(0)

    print("\033[0;32m", end='')
    print(f"matches! [{fn_name}]")
    print("\033[0;0m", end='')




test_build_overlap_dipol_quadrupol()
test_coordination_number()
