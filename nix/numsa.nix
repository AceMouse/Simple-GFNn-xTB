{
  stdenv,
  fetchFromGitHub,
  meson,
  ninja,
  cmake,
  asciidoctor,
  pkg-config,
  gfortran,
  python3,
  mctc-lib,
}:
stdenv.mkDerivation rec {
  name = "numsa";
  version = "0.2.0";
  srcs = [
    (fetchFromGitHub {
      owner = "grimme-lab";
      repo = "numsa";
      rev = "v${version}";
      sha256 = "sha256-PAzxeYyg/9P/3YFxKzM4ZFm2xT0AGap6q8/ei8jD/3M=";
    })
    (fetchFromGitHub {
      owner = "grimme-lab";
      repo = "mstore";
      rev = "v0.3.0";
      sha256 = "sha256-zfrxdrZ1Um52qTRNGJoqZNQuHhK3xM/mKfk0aBLrcjw=";
    })
  ];

  sourceRoot = "numsa";

  doCheck = true;

  nativeBuildInputs = [
    pkg-config
    meson
    ninja
    cmake
    python3
    asciidoctor
    mctc-lib
  ];

  unpackPhase = ''
    runHook preUnpack

    read -ra srcs <<< "$srcs"
    cp -r ''${srcs[0]} numsa
    chmod -R +rw numsa
    cp -r ''${srcs[1]} numsa/subprojects/mstore
    chmod -R +rw numsa/subprojects

    runHook postUnpack
  '';

  patchPhase = ''
    runHook prePatch

    substituteInPlace config/install-mod.py \
      --replace '#!/usr/bin/env python' '#!${python3.interpreter}'

    runHook postPatch
  '';

  configurePhase = ''
    runHook preConfigure

    export FC=${gfortran}/bin/gfortran
    meson setup _build

    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild

    meson compile -C _build

    runHook postBuild
  '';

  checkPhase = ''
    runHook preCheck

    meson test -C _build --print-errorlogs

    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall

    meson configure _build --prefix=$out
    meson install -C _build

    runHook postInstall
  '';

  fixupPhase = ''
    if [ -d "$out/lib64" ]; then
      mv "$out/lib64" "$out/lib"
    fi
  '';
}
