{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }: {
    apps."x86_64-linux" = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";

      xtb = (self.packages."x86_64-linux".xtb.overrideAttrs (finalAttrs: previousAttrs: {
        patches = [
          ./nix/patches/xtb/log_utils.patch
          ./nix/patches/xtb/log_args_and_outputs.patch
        ];
      }));

      xtb_test_data = builtins.derivation {
        name = "xtb-test-data";
        system = "x86_64-linux";
        builder = "${pkgs.bash}/bin/bash";
        src = ./data/C200.xyz;
        args = ["-c" ''
          PATH=$PATH:${pkgs.coreutils}/bin
          mkdir -p ./calls/{build_SDQH0,coordination_number,dim_basis,dtrf2,electro,form_product,get_multiints,h0scal,horizontal_shift,multipole_3d,newBasisset,olapp}
          ${xtb}/bin/xtb $src
          mv calls $out
        ''];
      };

    in {
      "cmp-impls" = let
        python = (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
          numpy
          scipy
          cvxopt
        ]));
      in {
        type = "app";
        program = toString (pkgs.writeShellScript "cmp-impls" ''
          PYTHONPATH=${pkgs.lib.cleanSource ./. } exec ${python}/bin/python \
            ${./cmp_impls.py} ${xtb_test_data}
        '');
      };
    };

    packages."x86_64-linux" = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in rec {
      xtb = pkgs.callPackage ./nix/xtb.nix { inherit cpx numsa; };
      cpx = pkgs.callPackage ./nix/cpx.nix { inherit numsa; };
      numsa = pkgs.callPackage ./nix/numsa.nix {};
    };

    devShells."x86_64-linux".default = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
      packages = with pkgs; [
        pkgs.pyright
        (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
          numpy
          scipy
          cvxopt
        ]))
      ];
    };
  };
}
