{
  description = "eminus - A plane wave density functional theory code.";

  inputs = { nixpkgs.url = "nixpkgs/nixpkgs-unstable"; };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

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
          ### viewer ###
          jupyter
          # nglview is missing
          plotly
          ### dev ###
          coverage
          furo
          pytest
          pytest-cov
          ruff
          sphinx
        ]);

    in {
      devShells."${system}".default = with pkgs;
        mkShell {
          buildInputs = [ pyEnv ];

          shellHook = ''
            pip install -e . --prefix $TMPDIR
            export PYTHONPATH="$(pwd):$PYTHONPATH"
            export MKL_NUM_THREADS="$(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}')"
            export OMP_NUM_THREADS="$(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}')"
          '';
        };
    };

}
