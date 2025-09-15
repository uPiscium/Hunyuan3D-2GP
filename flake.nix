{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          uv
          openssl
          llvm_14
          stdenv
          cudatoolkit
          # cudaPackages.cudatoolkit-legacy-runfile
          nvidia-docker
        ];
        buildInputs = with pkgs; [
          openssl
          llvm_14
          stdenv
        ];
        # LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.openssl}/lib:${pkgs.llvm_14}/lib";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc.lib
          openssl
          llvm_14
          cudatoolkit
          nvidia-docker
          libGL
          glib
        ]);
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        shellHook = ''
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH
        '';
      };
    };
}

