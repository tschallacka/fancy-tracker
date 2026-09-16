{
  description = "fancy-tracker - per-monitor cursor memory driven by webcam head tracking (macOS)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3;

        # YuNet face detector, pinned by the sha256 git-lfs records for the file
        # itself. raw.githubusercontent.com serves a 131-byte LFS pointer for this
        # path, so the media. host is the one that yields the actual ONNX.
        #
        # The .onnx name is load-bearing: cv2.dnn picks its importer from the
        # extension, and the store path is what the program is handed.
        yunetModel = pkgs.fetchurl {
          name = "face_detection_yunet_2023mar.onnx";
          url =
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx";
          hash = "sha256-jyOD5N08+7RVPqhxgQf8BCMhDclk+fQoBgSATtJVL6Q=";
        };

        # A face for the self-test, so it needs neither a camera nor the network.
        testFace = pkgs.fetchurl {
          name = "messi5.jpg";
          url =
            "https://raw.githubusercontent.com/opencv/opencv/913c266b4aa102c3c9d0507c794137de76f42bf6/samples/data/messi5.jpg";
          hash = "sha256-HVcOSWVOhMepQ5GFN72eXh74KSAVLhR8g0AG4jW+l8k=";
        };

        pyDeps = ps: [
          ps.numpy
          ps.opencv4
          ps.pyobjc-framework-Quartz
          ps.pyobjc-framework-Cocoa
        ];

        fancy-tracker = python.pkgs.buildPythonApplication {
          pname = "fancy-tracker";
          version = "0.1.0";
          pyproject = true;
          src = ./.;

          build-system = [ python.pkgs.setuptools ];
          dependencies = pyDeps python.pkgs;
          nativeBuildInputs = [ pkgs.makeWrapper ];

          # The model is content-addressed in the store; bake its path in so the
          # program never needs the network at runtime.
          postFixup = ''
            wrapProgram $out/bin/fancy-tracker \
              --set-default FANCY_TRACKER_MODEL ${yunetModel}
          '';

          # nixpkgs ships OpenCV and PyObjC under different distribution names than
          # PyPI does, so the wheel's own dependency list cannot be satisfied by
          # name here. The dependencies above are the real ones.
          dontCheckRuntimeDeps = true;
          doCheck = false;
          meta.platforms = pkgs.lib.platforms.darwin;
        };
      in
      {
        packages = {
          default = fancy-tracker;
          inherit fancy-tracker;
          yunet-model = yunetModel;
        };

        apps.default = {
          type = "app";
          program = "${fancy-tracker}/bin/fancy-tracker";
        };

        devShells.default = pkgs.mkShell {
          packages = [ (python.withPackages pyDeps) pkgs.ruff ];
          FANCY_TRACKER_MODEL = "${yunetModel}";
          FANCY_TRACKER_TEST_FACE = "${testFace}";
          shellHook = ''
            export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
            echo "fancy-tracker dev shell - run: python -m fancy_tracker --help"
          '';
        };
      });
}
