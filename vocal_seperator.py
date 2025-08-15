import argparse
import os
import shutil
import subprocess
from audio_separator.separator import Separator




def separator_audio(audio_file_path, audio_separator_model_file,cache_dir,audio_file_prefix):
    #cache_dir = os.path.join(os.path.dirname(audio_file_path), "vocals")
    os.makedirs(cache_dir, exist_ok=True)
    audio_separator = Separator(
        output_dir=cache_dir,
        output_single_stem="vocals",
        model_file_dir=os.path.dirname(audio_separator_model_file),
    )
    audio_separator.load_model(os.path.basename(audio_separator_model_file))
    assert audio_separator.model_instance is not None, "Fail to load audio separate model."
    outputs = audio_separator.separate(audio_file_path)
    subfolder_path = os.path.dirname(audio_file_path)
    vocal_audio_file = os.path.join(audio_separator.output_dir, outputs[0])
    destination_file = os.path.join(subfolder_path, f"audio_{audio_file_prefix}_vocal.wav")
    shutil.copy(vocal_audio_file, destination_file)
    os.remove(vocal_audio_file)
    print(f"Separated vocals saved to {destination_file}")
    return destination_file


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--audio_file_path", type=str)
#     parser.add_argument("--saved_vocal_path", type=str)
#     parser.add_argument("--audio_separator_model_file", type=str)
#     args = parser.parse_args()
#     audio_file_path = args.audio_file_path
#     audio_separator_model_file = args.audio_separator_model_file
#     saved_vocal_path = args.saved_vocal_path
#     cache_dir = os.path.join(os.path.dirname(audio_file_path), "vocals")
#     os.makedirs(cache_dir, exist_ok=True)
#     audio_separator = Separator(
#         output_dir=cache_dir,
#         output_single_stem="vocals",
#         model_file_dir=os.path.dirname(audio_separator_model_file),
#     )
#     audio_separator.load_model(os.path.basename(audio_separator_model_file))
#     assert audio_separator.model_instance is not None, "Fail to load audio separate model."
#     outputs = audio_separator.separate(audio_file_path)
#     subfolder_path = os.path.dirname(audio_file_path)
#     vocal_audio_file = os.path.join(audio_separator.output_dir, outputs[0])
#     destination_file = os.path.join(subfolder_path, "vocal.wav")
#     shutil.copy(vocal_audio_file, destination_file)
#     os.remove(vocal_audio_file)
