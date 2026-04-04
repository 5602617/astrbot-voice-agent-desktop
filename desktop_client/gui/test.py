from pathlib import Path

root_dir_text = r"D:\app_dasktop\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604"
root = Path(root_dir_text)

print("root =", root)
print("root exists =", root.exists())
print("root is_dir =", root.is_dir())

gpt_candidates = [
    root / "GPT_weights",
    root / "GPT_weights_v2",
    root / "GPT_weights_v2ProPlus",
    root / "GPT_SoVITS" / "pretrained_models",
]
sovits_candidates = [
    root / "SoVITS_weights",
    root / "SoVITS_weights_v2",
    root / "SoVITS_weights_v2ProPlus",
    root / "GPT_SoVITS" / "pretrained_models",
]

print("\nGPT candidates:")
for p in gpt_candidates:
    print(p, "exists=", p.exists(), "is_dir=", p.is_dir() if p.exists() else False)

print("\nSoVITS candidates:")
for p in sovits_candidates:
    print(p, "exists=", p.exists(), "is_dir=", p.is_dir() if p.exists() else False)

gpt_dir = next((p for p in gpt_candidates if p.exists() and p.is_dir()), None)
sovits_dir = next((p for p in sovits_candidates if p.exists() and p.is_dir()), None)

print("\nselected gpt_dir =", gpt_dir)
print("selected sovits_dir =", sovits_dir)

gpt_files = sorted(str(p) for p in gpt_dir.glob("*.ckpt")) if gpt_dir else []
sovits_files = sorted(str(p) for p in sovits_dir.glob("*.pth")) if sovits_dir else []

print("\ngpt_files =", gpt_files)
print("sovits_files =", sovits_files)