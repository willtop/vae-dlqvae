# python main.py --model vanillavae --dataset mpi3d --latent_dim 30
# python main.py --model dlqvae --dataset mpi3d
python main.py --model factorvae --dataset mpi3d_pairs --latent_dim 100 
python main.py --model factorvae --dataset mpi3d_pairs --latent_dim 100 --test 

