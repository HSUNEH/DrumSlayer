import multiprocessing

# 시스템의 CPU 코어 수 확인
num_cpus = multiprocessing.cpu_count()
print("Number of CPU cores:", num_cpus)
