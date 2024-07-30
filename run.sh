# python main.py --model vanillavae --dataset mpi3d
# python main.py --model dlqvae --dataset mpi3d
# python main.py --model factorvae --dataset mpi3d_pairs
# python main.py --model factorvae --dataset mpi3d_pairs --test 
python main.py --model factorvae --dataset mpi3d_pairs --latent_dim 10 --debug
python main.py --model factorvae --dataset mpi3d_pairs --latent_dim 10 --test --debug

