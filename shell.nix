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
    pkgs.python3Packages.fastapi
    pkgs.python3Packages.debugpy
    pkgs.python3Packages.ollama
    pkgs.python3Packages.uvicorn
    pkgs.python3Packages.matplotlib
    pkgs.jupyter
    pkgs.uv
  ];

  shellHook = ''
    echo "Welcome to your Python development environment!"
    python --version
  '';
}
