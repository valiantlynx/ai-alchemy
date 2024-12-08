{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  name = "python-environment";

  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.dash
    pkgs.python3Packages.plotly
    pkgs.python3Packages.pandas
    pkgs.python3Packages.numpy
    pkgs.python3Packages.scikit-learn
    pkgs.python3Packages.flask
    pkgs.python3Packages.matplotlib
    pkgs.jupyter
    pkgs.uv
  ];

  shellHook = ''
    echo "Welcome to your Python development environment!"
    python --version
  '';
}
