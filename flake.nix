{
  description = "ICSSI 2026 Hackathon development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        allowedUnfree = [ "graphite-cli" ];
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfreePredicate = pkg:
            builtins.elem (nixpkgs.lib.getName pkg) allowedUnfree;
        };
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          numpy
          scipy
          pandas
          matplotlib
          plotly
          jupyterlab
          ipykernel
          ipywidgets
          seaborn
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            pythonEnv
            nodejs_22
            pnpm
            gnumake
            graphite-cli
          ];
          env = {
            UV_PYTHON_DOWNLOADS = "never";
            UV_PYTHON = "${pythonEnv}/bin/python3";
          };
          shellHook = ''
            uv sync
            export PATH="$PWD/.venv/bin:$PATH"
            pnpm install
          '';
        };
      });
}
