import whisperx
import gc 

device = "cuda" 
audio_file = "transcripts/clip_11_temp_audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("D:/vegam-whisper-medium-ml", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code="ml", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment


#=======================================================================================================================


# from faster_whisper import WhisperModel

# model_path = "D:/vegam-whisper-medium-ml"

# # Run on GPU with FP16
# model = WhisperModel(model_path, device="cuda", compute_type="float16")

# # or run on GPU with INT8
# # model = WhisperModel(model_path, device="cuda", compute_type="int8_float16")
# # or run on CPU with INT8
# # model = WhisperModel(model_path, device="cpu", compute_type="int8")

# segments, info = model.transcribe("transcripts/clip_2_temp_audio.mp3", beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
