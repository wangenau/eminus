# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
{
  description = "eminus - Pythonic electronic structure theory.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = pkgs.python314;
        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        version = pyproject.project.version;

        eminus = python.pkgs.buildPythonPackage {
          pname = "eminus";
          inherit version;
          src = ./.;
          format = "pyproject";

          nativeBuildInputs = [ python.pkgs.hatchling ];

          propagatedBuildInputs = with python.pkgs; [
            ### backends ###
            # array-api-compat
            # cupy
            # torch-bin
            ### basic ###
            numpy
            scipy
            ### d3 ###
            simple-dftd3
            ### gui ###
            nglview
            plotly
            ### hdf5 ###
            h5py
            ### fods and libxc ###
            libxc
            pyscf
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
          ];

          pythonImportsCheck = [ "eminus" ];

          meta = with pkgs.lib; {
            description = "Pythonic electronic structure theory.";
            homepage = "https://wangenau.gitlab.io/eminus";
            license = licenses.asl20;
            maintainers = with maintainers; [ wangenau ];
          };
        };

      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ eminus ];
          shellHook = "export JUPYTER_PLATFORM_DIRS=1";
        };
        devShells.cupy = pkgs.mkShell {
          packages = [
            eminus
            python.pkgs.array-api-compat
            python.pkgs.cupy
          ];
          shellHook = ''
            export JUPYTER_PLATFORM_DIRS=1
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath [ pkgs.cudaPackages.cuda_nvrtc ]}
          '';
        };
        devShells.torch = pkgs.mkShell {
          packages = [
            eminus
            python.pkgs.array-api-compat
            python.pkgs.torch-bin
          ];
          shellHook = "export JUPYTER_PLATFORM_DIRS=1";
        };
        packages.default = python.withPackages (p: [ eminus ]);
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
