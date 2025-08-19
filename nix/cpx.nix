{
  stdenv,
  fetchFromGitHub,
  meson,
  ninja,
  lapack,
  cmake,
  pkg-config,
  gfortran,
  blas,
  toml-f,
  test-drive,
  mctc-lib,
  numsa,
}:
stdenv.mkDerivation rec {
  name = "cpx";
  version = "1.1.0";
  src = fetchFromGitHub {
    owner = "grimme-lab";
    repo = "CPCM-X";
    rev = "v${version}";
    sha256 = "sha256-FyPUECbcqUHoGq1LASvPF4qSUKQ5N/y1itq8e2wGliE=";
  };

  dontPatch = true;
  dontFixup = true;

  nativeBuildInputs = [
    pkg-config
    meson
    ninja
    cmake
    lapack
    blas
    toml-f
    test-drive
    mctc-lib
    numsa
  ];

  configurePhase = ''
    runHook preConfigure

    export FC=${gfortran}/bin/gfortran
    meson setup build --buildtype=release

    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild

    ninja -C build

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/{bin,lib,include}
    mv build/app/cpx $out/bin
    mv build/libcpx.a $out/lib
    mv build/libcpx.a.p $out/include/cpx

    mkdir $out/lib/pkgconfig
    cat <<'EOF' > $out/lib/pkgconfig/cpx.pc
    prefix=@out@
    includedir=''${prefix}/include
    libdir=''${prefix}/lib
    numsa_includedir=${numsa}/include
    numsa_libdir=${numsa}/lib

    Name: cpx
    Description: CPCX Library for various computations
    Version: 1.1.0
    Requires.private: numsa >= 0.2.0
    Cflags: -I''${includedir} -I''${includedir}/cpx -I''${numsa_includedir} -I''${numsa_includedir}/numsa/gcc-14.2.1
    Libs: -L''${libdir} -L''${numsa_libdir} -lcpx -lnumsa
    EOF

    substituteAllInPlace "$out/lib/pkgconfig/cpx.pc"

    runHook postInstall
  '';
}
