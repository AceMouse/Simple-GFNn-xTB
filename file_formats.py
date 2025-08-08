from chemistry_constants import symbol_to_atom, angstrom_to_bohr_conversion_constant
def parse_coord_file(file: str) -> tuple[list[int], list[list[float]]]:
    atoms = []
    positions = []

    for line in file.split('\n'):
        words = line.split()
        if len(words) != 4:
            continue
        atoms += [symbol_to_atom[words[3]]-1]
        positions += [[float(w) for w in words[0:3]]]
    return atoms, positions
def parse_xyz_file(file: str) -> tuple[list[int], list[list[float]]]:
    atoms = []
    positions = []
    lines = file.split('\n')
    for line in lines[2:]:
        words = line.split()
        if len(words) != 4:
            continue
        atoms += [symbol_to_atom[words[0].lower()]-1]
        positions += [[float(w)*angstrom_to_bohr_conversion_constant for w in words[1:4]]]
    return atoms, positions
