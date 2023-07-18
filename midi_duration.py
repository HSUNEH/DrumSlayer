def get_midi_duration(file_path):
    try:
        ticks_per_beat = None
        total_ticks = 0

        with open(file_path, 'rb') as file:
            file.seek(10)  # 헤더 길이 위치로 이동
            header_length = int.from_bytes(file.read(4), 'big')  # 헤더 길이 읽기
            file.seek(14)  # 트랙 수 위치로 이동
            num_tracks = int.from_bytes(file.read(2), 'big')  # 트랙 수 읽기

            for _ in range(num_tracks):
                file.seek(file.tell() + 4)  # 트랙 헤더 건너뛰기
                track_length = int.from_bytes(file.read(4), 'big')  # 트랙 길이 읽기

                # 트랙 메시지 읽기
                while track_length > 0:
                    delta_time = 0
                    variable_length = int.from_bytes(file.read(1), 'big')

                    while variable_length & 0x80:
                        delta_time = (delta_time << 7) + (variable_length & 0x7F)
                        variable_length = int.from_bytes(file.read(1), 'big')

                    delta_time = (delta_time << 7) + variable_length
                    total_ticks += delta_time
                    track_length -= variable_length.bit_length() // 7 + 1

                    event_type = int.from_bytes(file.read(1), 'big')

                    if event_type == 0xFF:  # 메타 이벤트
                        meta_type = int.from_bytes(file.read(1), 'big')

                        if meta_type == 0x51:  # 템포 이벤트
                            file.read(3)  # 템포 데이터 건너뛰기
                        elif meta_type == 0x2F:  # 트랙 종료 이벤트
                            break

                if ticks_per_beat is None:
                    file.seek(header_length + 8)  # 트랙 길이 위치로 이동
                    ticks_per_beat = int.from_bytes(file.read(2), 'big')  # 틱당 박자 읽기

        beats = total_ticks / ticks_per_beat
        duration = beats * (60 / 120)  # 기본적인 BPM 값 사용
        return duration

    except Exception as e:
        print(f"Error occurred: {e}")


# MIDI 파일 경로 설정
midi_file_path = "./generated_midi/practice.midi"

# duration 가져오기
duration = get_midi_duration(midi_file_path)
print(f"MIDI duration: {duration} seconds")
