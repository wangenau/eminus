{
  description = "eminus - A plane wave density functional theory code.";

  inputs = { nixpkgs.url = "nixpkgs/nixos-23.05"; };

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
          ### fods and libxc ###
          pyscf
          ### torch ###
          torch-bin
          # torch if allowUnfree = true is undesired
          ### viewer ###
          jupyter
          matplotlib
          # nglview is missing
          plotly
          ### dev ###
          coverage
          furo
          mypy
          pytest
          pytest-cov
          sphinx
          sphinxcontrib-bibtex
        ]);

    in {
      devShells."${system}".default = with pkgs;
        mkShell {
          buildInputs = [
            pyEnv
            ### dispersion ###
            # simple-dftd3 does not work in the Python environment
            ### dev ###
            ruff
          ];

          shellHook = ''
            pip install -e . --prefix "$TMPDIR"
            export PYTHONPATH="$(pwd):$PYTHONPATH"
            export OMP_NUM_THREADS="$(nproc)"
            export MKL_NUM_THREADS="$(nproc)"
            export MPLBACKEND="TKAgg"
          '';
        };
    };
}
