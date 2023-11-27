import numpy as np

# npy 파일 경로
file_path = "/Users/hwang/DrumSlayer/pracitce.npy"

# npy 파일 읽기
data = np.load(file_path)

# 0이 아닌 위치와 값을 저장할 리스트
nonzero_positions = []
nonzero_values = []

# npy 배열 순회
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i, j] != 0:
            nonzero_positions.append((i, j))
            nonzero_values.append(data[i, j])

# 결과 출력
for pos, value in zip(nonzero_positions, nonzero_values):
    print(f"Position: {pos}, Value: {value}")

