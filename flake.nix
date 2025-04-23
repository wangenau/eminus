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
          # config.allowUnfree = true;
        };
        python = pkgs.python313;
        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        version = pyproject.project.version;

        eminus = python.pkgs.buildPythonPackage {
          pname = "eminus";
          inherit version;
          src = ./.;
          format = "pyproject";

          nativeBuildInputs = [ python.pkgs.hatchling ];

          propagatedBuildInputs = with python.pkgs; [
            ### basic ###
            numpy
            scipy
            ### dispersion ###
            simple-dftd3
            ### hdf5 ###
            h5py
            ### fods and libxc ###
            pyscf
            ### jax ###
            # jax
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
        devShells.default = pkgs.mkShell { buildInputs = [ eminus ]; };
        packages.default = python.withPackages (p: [ eminus ]);
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
