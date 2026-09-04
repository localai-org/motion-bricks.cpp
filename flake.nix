{
  description = "motion-bricks.cpp — GGML CPU/Vulkan motion planning";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/4bd9165a9165d7b5e33ae57f3eecbcb28fb231c9";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      eachSystem = nixpkgs.lib.genAttrs systems;
    in {
      devShells = eachSystem (system:
        let pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cmake
              ninja
              clang
              pkg-config
              git
              git-lfs
              ffmpeg-headless
              go
              python3
              python3Packages.huggingface-hub
              python3Packages.numpy
              python3Packages.safetensors
              shaderc
              spirv-tools
              vulkan-headers
              vulkan-loader
              vulkan-tools
            ];
            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            '';
          };
        });
    };
}
