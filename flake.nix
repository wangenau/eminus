# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
{
  description = "eminus - Pythonic electronic structure theory.";

  inputs = { nixpkgs.url = "nixpkgs/nixos-unstable"; };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # config.allowUnfree = true;
      };

      pyEnv = pkgs.python3.withPackages (p:
        with p; [
          ### basic ###
          numpy
          pip
          scipy
          ### dispersion ###
          simple-dftd3
          ### hdf5 ###
          h5py
          ### fods and libxc ###
          pyscf
          ### torch ###
          # torch-bin
          ### viewer ###
          # nglview is missing
          plotly
          ### dev ###
          furo
          jupyter
          matplotlib
          mypy
          pytest
          ruff
          sphinx
          sphinx-design
          sphinxcontrib-bibtex
        ]);

    in
    {
      devShells."${system}".default = with pkgs;
        mkShell {
          buildInputs = [ pyEnv ];

          shellHook = ''
            pip install -e . --prefix="$TMPDIR"
            export PYTHONPATH="$(pwd):$PYTHONPATH"
          '';
        };
    };
}
