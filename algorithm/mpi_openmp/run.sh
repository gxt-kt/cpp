#!/usr/bin/bash


# # 单线程
# mpirun -np 1 ./build/m_single -s 10240 -r 10240 -c 10240 -t 10 -R 1024 -C 1024 -v
#
# # 多线程
# OMP_NUM_THREADS=10 OMP_PROC_BIND=close ./build/m_multiple -s 10240 -r 10240 -c 10240 -t 10 -R 1024 -C 1024 -v
#
# # 多进程
# mpirun -np 10 ./build/m_single -s 10240 -r 10240 -c 10240 -t 10 -R 1024 -C 1024 -v
#
# # OMP_NUM_THREADS=10 OMP_PROC_BIND=close mpirun -np 2 ./build/m_multiple -r 10240 -c 10240 -t 10 -b 1024 -R 1024 -C 1024 -v

current_dir=$(dirname "$0")

# 单线程
mpirun -np 1 ${current_dir}/bin/m_single -s 10240 -r 10240 -c 10240 -t 10 -R 1024 -C 1024 -v

echo "======================"
echo "======================"

# 多线程
OMP_NUM_THREADS=10 OMP_PROC_BIND=close ${current_dir}/bin/m_multiple -s 10240 -r 10240 -c 10240 -t 10 -R 1024 -C 1024 -v

echo "======================"
echo "======================"

# 多进程
mpirun -np 10 ${current_dir}/bin/m_single -s 10240 -r 10240 -c 10240 -t 10 -R 1024 -C 1024 -v

# OMP_NUM_THREADS=10 OMP_PROC_BIND=close mpirun -np 2 ./build/m_multiple -r 10240 -c 10240 -t 10 -b 1024 -R 1024 -C 1024 -v

