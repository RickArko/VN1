micromamba create -f env.yaml -y
micromamba run -n vn1 python -m ipykernel install --user --name vn1 --display-name "vn1"