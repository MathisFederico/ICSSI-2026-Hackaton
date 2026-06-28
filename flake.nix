{
  description = "ICSSI 2026 Hackathon development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        allowedUnfree = [ "graphite-cli" "google-cloud-sdk" ];
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
            google-cloud-sdk
          ];
          env = {
            UV_PYTHON_DOWNLOADS = "never";
            UV_PYTHON = "${pythonEnv}/bin/python3";
          };
          shellHook = ''
            # Wheel-installed binary extensions (e.g. pyzmq) dlopen libstdc++
            # at runtime. The nix-provided python doesn't put it on the loader
            # path, so without this the Jupyter kernel dies on `import zmq`
            # and notebook cells hang waiting for it.
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:$LD_LIBRARY_PATH"
            uv sync
            export PATH="$PWD/.venv/bin:$PATH"
            pnpm install
          '';
        };
      });
}
