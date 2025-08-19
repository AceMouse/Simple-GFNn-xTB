{
  stdenv,
  fetchFromGitHub,
  meson,
  ninja,
  lapack,
  cmake,
  asciidoctor,
  pkg-config,
  gfortran,
  python3,
  blas,
  tblite,
  go,
  toml-f,
  test-drive,
  simple-dftd3,
  mstore,
  multicharge,
  dftd4,
  json-fortran,
  mctc-lib,
  cpx,
  numsa,
}:
stdenv.mkDerivation {
  name = "xtb";
  version = "6.7.1";
  src = fetchFromGitHub {
    owner = "grimme-lab";
    repo = "xtb";
    rev = "8e4f8d25ba9de73aa199fefda1ecba8fe7dbfa3b";
    sha256 = "sha256-lla3oNtXYOKhTcNxfYHqZbcqirZyUBJH+OZL37b8Ux4=";
  };

  dontFixup = true;

  nativeBuildInputs = [
    meson
    ninja
    gfortran
    pkg-config
    cmake
    lapack
    python3
    blas
    asciidoctor
    tblite
    go
    toml-f
    test-drive
    simple-dftd3
    mstore
    multicharge
    dftd4
    json-fortran
    mctc-lib
    cpx
    numsa
  ];

  configurePhase = ''
    runHook preConfigure
    
    export FFLAGS=-fdump-ipa-cgraph
    export FC=${gfortran}/bin/gfortran
    meson setup build --buildtype=release -Dfortran_args=-fdump-ipa-cgraph
    meson configure build --prefix=$out -Dfortran_args=-fdump-ipa-cgraph

    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild

    ninja -C build

    runHook postBuild
  '';

  checkPhase = ''
    runHook preCheck

    ninja -C build test

    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall

    ninja -C build install

    runHook postInstall
  '';
}
