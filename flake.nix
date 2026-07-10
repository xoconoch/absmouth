{
  description = "A flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    master.url = "github:NixOS/nixpkgs/master";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, ... }:
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
          pkgs,
          system,
          ...
        }:
        let
          # Instantiate master for this specific system
          masterPkgs = import inputs.master { inherit system; };
        in
        {
          packages = { };
          devShells.default = pkgs.mkShell {
            LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]}";

            packages = with pkgs; [
              (python3.withPackages (
                ps: with ps; [
                  librosa
                  numpy
                  scipy
                  einops
                  transformers
                  openvino
                  onnx
                  # Pull directly from master's python packages, then override it
                  (masterPkgs.python3Packages.onnxruntime.override {
                    onnxruntime = masterPkgs.onnxruntime.override { openvinoSupport = true; };
                  })
                  onnxruntime
                  onnxscript
                  requests
                  mutagen
                ]
              ))
              ruff
              stdenv.cc.cc.lib
              zlib
            ];
          };
        };
    };
}
