# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
{
  description = "eminus - A pythonic plane wave density functional theory (DFT) code.";

  inputs = { nixpkgs.url = "nixpkgs/nixos-unstable"; };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # Required for torch-bin
      };

      pyEnv = pkgs.python3.withPackages (p:
        with p; [
          ### basic ###
          numpy
          pip
          scipy
          ### dispersion ###
          simple-dftd3
          ### fods and libxc ###
          pyscf
          ### torch ###
          torch-bin
          # torch if allowUnfree = true is undesired
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
          sphinx
          sphinx-design
          sphinxcontrib-bibtex
        ]);

    in
    {
      devShells."${system}".default = with pkgs;
        mkShell {
          buildInputs = [
            pyEnv
            ### dev ###
            ruff
          ];

          shellHook = ''
            pip install -e . --prefix "$TMPDIR"
            export PYTHONPATH="$(pwd):$PYTHONPATH"
            export MPLBACKEND="TKAgg"
          '';
        };
    };
}
