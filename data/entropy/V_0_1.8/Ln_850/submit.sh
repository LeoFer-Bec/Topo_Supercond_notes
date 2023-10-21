#!/bin/bash -l
#SBATCH -J V0_1.8_Ln850
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem=4000M
#SBATCH --time=2:35:00
#SBATCH -A g94-1727
#SBATCH -p topola
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=becerra@magtop.ifpan.edu.pl
#SBATCH -e errfile
    
source /lu/topola/home/victor/miniconda3/etc/profile.d/conda.sh
    
cd /lu/topola/home/victor/entropy/V_0_1.8/Ln_850
    
python3.7 entropy_cluster.py
    
exit
