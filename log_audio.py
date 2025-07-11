import os
import wandb
import numpy as np
import soundfile as sf
from tqdm import tqdm
import argparse
import tempfile
from pydub import AudioSegment
import shutil

def convert_to_mp3(wav_path, bitrate="192k"):
    """WAV 파일을 MP3로 변환하여 임시 파일로 저장하고 경로 반환"""
    # 임시 파일 생성
    temp_dir = tempfile.gettempdir()
    mp3_filename = os.path.basename(wav_path).replace('.wav', '.mp3')
    mp3_path = os.path.join(temp_dir, mp3_filename)
    
    # WAV를 MP3로 변환
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate=bitrate)
    
    return mp3_path

def log_audio_comparison(original_dir, evaluation_results_dir, project_name="music-source-separation", use_mp3=True, mp3_bitrate="192k"):
    """
    원본 오디오와 여러 체크포인트의 분리된 오디오를 Wandb에 로깅하는 함수
    
    Args:
        original_dir: 원본 오디오 파일이 있는 디렉토리
        evaluation_results_dir: 평가 결과가 저장된 디렉토리 (checkpoint 폴더들이 있는 상위 디렉토리)
        project_name: Wandb 프로젝트 이름
        use_mp3: MP3로 변환하여 업로드할지 여부
        mp3_bitrate: MP3 변환 시 비트레이트
    """
    # 임시 파일 저장을 위한 디렉토리
    temp_files = []
    
    try:
        # Wandb 초기화
        run = wandb.init(project=project_name)
        
        # 체크포인트 디렉토리 찾기
        checkpoint_dirs = {}
        for checkpoint_name in os.listdir(evaluation_results_dir):
            checkpoint_path = os.path.join(evaluation_results_dir, checkpoint_name)
            if os.path.isdir(checkpoint_path) and not checkpoint_name.startswith('.'):
                # evaluation_results.txt와 config.yaml 파일이 있는지 확인하여 유효한 체크포인트 폴더인지 확인
                if os.path.exists(os.path.join(checkpoint_path, "evaluation_results.txt")):
                    checkpoint_dirs[checkpoint_name] = checkpoint_path
        
        print(f"Found {len(checkpoint_dirs)} checkpoint directories: {list(checkpoint_dirs.keys())}")
        
        # 모든 체크포인트 결과 디렉토리에서 공통된 곡 목록 찾기
        all_songs = set()
        for checkpoint_name, result_dir in checkpoint_dirs.items():
            songs = [d for d in os.listdir(result_dir) 
                    if os.path.isdir(os.path.join(result_dir, d)) 
                    and not d.startswith('.') 
                    and not d in ['evaluation_results', 'config']]
            all_songs.update(songs)
        
        all_songs = sorted(list(all_songs))
        
        print(f"Found {len(all_songs)} songs across all model directories")
        
        # 각 곡별로 처리
        for song_dir in tqdm(all_songs):
            print(f"Processing {song_dir}...")
            
            # 원본 mixture 파일 찾기
            original_mixture = None
            # 먼저 test 폴더에서 찾기
            test_dir = os.path.join(original_dir, "test")
            if os.path.exists(test_dir):
                # 폴더 내에 mixture.wav 파일이 있는지 확인
                potential_dir = os.path.join(test_dir, song_dir)
                if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
                    mixture_path = os.path.join(potential_dir, "mixture.wav")
                    if os.path.exists(mixture_path):
                        original_mixture = mixture_path
                
                # 또는 직접 wav 파일이 있는지 확인
                if original_mixture is None:
                    potential_file = os.path.join(test_dir, f"{song_dir}.wav")
                    if os.path.exists(potential_file):
                        original_mixture = potential_file
            
            # 원본 파일을 찾지 못했다면 train 폴더에서도 찾기
            if original_mixture is None:
                train_dir = os.path.join(original_dir, "train")
                if os.path.exists(train_dir):
                    potential_dir = os.path.join(train_dir, song_dir)
                    if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
                        mixture_path = os.path.join(potential_dir, "mixture.wav")
                        if os.path.exists(mixture_path):
                            original_mixture = mixture_path
            
            # 원본 파일을 찾지 못했다면 체크포인트 디렉토리에서 mixture.wav 찾기
            if original_mixture is None:
                for checkpoint_name, result_dir in checkpoint_dirs.items():
                    song_path = os.path.join(result_dir, song_dir)
                    if os.path.exists(song_path) and os.path.isdir(song_path):
                        mixture_path = os.path.join(song_path, "mixture.wav")
                        if os.path.exists(mixture_path):
                            original_mixture = mixture_path
                            break
            
            # 원본 파일을 찾지 못했다면 건너뛰기
            if original_mixture is None:
                print(f"Could not find original mixture for {song_dir}, skipping...")
                continue
                
            # 원본 stems 파일 찾기 (있는 경우)
            original_stems = {}
            stems_dir = None
            
            # test 폴더에서 stems 찾기
            if os.path.exists(test_dir):
                potential_dir = os.path.join(test_dir, song_dir)
                if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
                    stems_dir = potential_dir
            
            # train 폴더에서 stems 찾기
            if stems_dir is None and os.path.exists(os.path.join(original_dir, "train")):
                train_dir = os.path.join(original_dir, "train")
                potential_dir = os.path.join(train_dir, song_dir)
                if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
                    stems_dir = potential_dir
            
            # stems 파일 로드 (있는 경우)
            if stems_dir is not None:
                for stem_name in ['bass', 'drums', 'other', 'vocals']:
                    stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
                    if os.path.exists(stem_path):
                        original_stems[stem_name] = stem_path
                        
            # 체크포인트 디렉토리에서 reference 파일 찾기 (원본이 없는 경우)
            if not original_stems:
                for checkpoint_name, result_dir in checkpoint_dirs.items():
                    song_path = os.path.join(result_dir, song_dir)
                    if os.path.exists(song_path) and os.path.isdir(song_path):
                        for stem_name in ['bass', 'drums', 'other', 'vocals']:
                            ref_path = os.path.join(song_path, f"{stem_name}_reference.wav")
                            if os.path.exists(ref_path) and stem_name not in original_stems:
                                original_stems[stem_name] = ref_path
            
            # 각 체크포인트별 분리된 stems 파일 찾기
            checkpoint_stems = {}
            for checkpoint_name, result_dir in checkpoint_dirs.items():
                song_path = os.path.join(result_dir, song_dir)
                if not os.path.exists(song_path) or not os.path.isdir(song_path):
                    continue
                    
                checkpoint_stems[checkpoint_name] = {}
                for stem_name in ['bass', 'drums', 'other', 'vocals']:
                    # 먼저 estimate 파일 찾기
                    est_path = os.path.join(song_path, f"{stem_name}_estimate.wav")
                    if os.path.exists(est_path):
                        checkpoint_stems[checkpoint_name][stem_name] = est_path
                    else:
                        # estimate 파일이 없으면 일반 stem 파일 찾기
                        stem_path = os.path.join(song_path, f"{stem_name}.wav")
                        if os.path.exists(stem_path):
                            checkpoint_stems[checkpoint_name][stem_name] = stem_path
            
            # 각 stem 유형별로 비교 테이블 생성
            for stem_name in ['bass', 'drums', 'other', 'vocals']:
                # 해당 stem에 대한 오디오 파일 모음
                stem_audios = []
                
                # 원본 stem 추가 (있는 경우)
                if stem_name in original_stems:
                    audio_path = original_stems[stem_name]
                    # MP3로 변환
                    if use_mp3:
                        mp3_path = convert_to_mp3(audio_path, mp3_bitrate)
                        temp_files.append(mp3_path)
                        audio_path = mp3_path
                    
                    sample_rate = sf.info(original_stems[stem_name]).samplerate
                    stem_audios.append(wandb.Audio(
                        audio_path,
                        caption=f"Ground Truth {stem_name.capitalize()}",
                        sample_rate=sample_rate
                    ))
                else:
                    stem_audios.append(None)  # 원본이 없는 경우 None 추가
                
                # 각 체크포인트의 분리 결과 추가
                for checkpoint_name in checkpoint_dirs.keys():
                    if checkpoint_name in checkpoint_stems and stem_name in checkpoint_stems[checkpoint_name]:
                        audio_path = checkpoint_stems[checkpoint_name][stem_name]
                        # MP3로 변환
                        if use_mp3:
                            mp3_path = convert_to_mp3(audio_path, mp3_bitrate)
                            temp_files.append(mp3_path)
                            audio_path = mp3_path
                        
                        sample_rate = sf.info(checkpoint_stems[checkpoint_name][stem_name]).samplerate
                        stem_audios.append(wandb.Audio(
                            audio_path,
                            caption=f"{checkpoint_name} - {stem_name.capitalize()}",
                            sample_rate=sample_rate
                        ))
                    else:
                        stem_audios.append(None)  # 해당 체크포인트의 결과가 없는 경우 None 추가
                
                # Wandb 테이블에 로깅
                columns = ["Ground Truth"]
                columns.extend(list(checkpoint_dirs.keys()))
                
                # None 값을 제외한 실제 데이터만 포함
                valid_audios = [a for a in stem_audios if a is not None]
                if valid_audios:
                    # 테이블 생성
                    table = wandb.Table(columns=columns)
                    table.add_data(*stem_audios)
                    
                    # 로깅
                    wandb.log({
                        f"{song_dir}/{stem_name}_comparison": table
                    })
                
                # 원본 mixture도 별도로 로깅 (참고용)
                if original_mixture:
                    audio_path = original_mixture
                    # MP3로 변환
                    if use_mp3:
                        mp3_path = convert_to_mp3(audio_path, mp3_bitrate)
                        temp_files.append(mp3_path)
                        audio_path = mp3_path
                    
                    sample_rate = sf.info(original_mixture).samplerate
                    wandb.log({
                        f"{song_dir}/original_mixture": wandb.Audio(
                            audio_path,
                            caption=f"Original Mixture",
                            sample_rate=sample_rate
                        )
                    })
        
        # Wandb 종료
        wandb.finish()
    
    finally:
        # 임시 파일 정리
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {temp_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log audio comparison to Wandb")
    parser.add_argument('--original_dir', type=str, default="/home/kof008/custom_separation_dataset",
                        help='Directory containing original audio files')
    parser.add_argument('--evaluation_dir', type=str, default="./evaluation_results",
                        help='Directory containing evaluation results (with checkpoint folders)')
    parser.add_argument('--project', type=str, default="music-source-separation",
                        help='Wandb project name')
    parser.add_argument('--use_mp3', action='store_true', default=True,
                        help='Convert WAV files to MP3 before uploading (reduces size)')
    parser.add_argument('--mp3_bitrate', type=str, default="192k",
                        help='Bitrate for MP3 conversion (e.g., 128k, 192k, 256k)')
    args = parser.parse_args()
    
    # Wandb 로그인 (처음 실행 시 API 키 입력 필요)
    wandb.login()
    
    log_audio_comparison(
        args.original_dir, 
        args.evaluation_dir, 
        args.project,
        use_mp3=args.use_mp3,
        mp3_bitrate=args.mp3_bitrate
    )