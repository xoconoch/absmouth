{
  description = "A flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ self, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem =
        {
          config,
          lib,
          pkgs,
          system,
          ...
        }:
        {
          _module.args.pkgs = import self.inputs.nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          packages = { };
          devShells.default = pkgs.mkShell {
            LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath (
              with pkgs;
              [
                stdenv.cc.cc.lib
                zlib
                cudaPackages_13.cuda_cudart
                cudaPackages_13.libcublas
                cudaPackages_13.cudnn
                cudaPackages_13.cuda_nvrtc
                cudaPackages_13.libcufft
                cudaPackages_13.libcurand
                cudaPackages_13.cudatoolkit
                linuxPackages.nvidiaPackages.stable
              ]
            )}";

            packages = with pkgs; [
              ruff
              stdenv.cc.cc.lib
              zlib
              linuxPackages.nvidiaPackages.stable
              cudaPackages_13.cuda_cudart
              cudaPackages_13.libcublas
              cudaPackages_13.cudnn
              cudaPackages_13.cuda_nvrtc
              cudaPackages_13.libcufft
              cudaPackages_13.libcurand
              cudaPackages_13.cudatoolkit
            ];
          };
        };
    };
}
