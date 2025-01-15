# SPDX-FileCopyrightText: 2021 The eminus developers
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
          scipy
          uv
          ### dispersion ###
          simple-dftd3
          ### hdf5 ###
          h5py
          ### fods, libxc, and symmetry ###
          pyscf
          ### torch ###
          # torch-bin
          ### viewer ###
          # nglview is missing
          plotly
          ### dev ###
          coverage
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
            uv pip install -e . --link-mode=copy --prefix="$TMPDIR"
            export PYTHONPATH="$(pwd):$PYTHONPATH"
          '';
        };
    };
}
