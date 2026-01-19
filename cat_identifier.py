# python
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

CAT_KEYWORDS = ("cat", "tabby", "siamese", "persian", "egyptian", "kitty", "kitten")

def preprocess_pil(img, target_size=(224, 224)):
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, 0)
    return preprocess_input(arr)

def cat_score_from_decoded(decoded):
    score = 0.0
    for _, name, prob in decoded:
        name_norm = name.replace("_", " ").lower()
        if "cat" in name_norm or any(k in name_norm for k in CAT_KEYWORDS):
            score += float(prob)
    return score

def detect_cat(image_path, top=5):
    model = MobileNetV2(weights="imagenet")

    # Full-image prediction (original behavior)
    x_full = preprocess_pil(Image.open(image_path))
    preds_full = model.predict(x_full)
    decoded_full = decode_predictions(preds_full, top=top)[0]

    # Left / Right scoring: decode more top classes to capture cat variants
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    left_img = img.crop((0, 0, w // 3, h))
    center_img = img.crop((w // 3, 0, w // 3 * 2, h))
    right_img = img.crop((w // 3 * 2, 0, w, h))

    x_left = preprocess_pil(left_img)
    x_center = preprocess_pil(center_img)
    x_right = preprocess_pil(right_img)

    preds_left = model.predict(x_left)
    preds_center = model.predict(x_center)
    preds_right = model.predict(x_right)

    decoded_left = decode_predictions(preds_left, top=50)[0]
    decoded_center = decode_predictions(preds_center, top=50)[0]
    decoded_right = decode_predictions(preds_right, top=50)[0]

    left_score = cat_score_from_decoded(decoded_left)
    center_score = cat_score_from_decoded(decoded_center)
    right_score = cat_score_from_decoded(decoded_right)

    if left_score == 0 and right_score == 0 and center_score == 0:
        side = "unknown"
    elif center_score > right_score and center_score > left_score:
        side = "center"
    elif left_score > right_score and left_score > center_score:
        side = "left"
    elif right_score > left_score and right_score > center_score:
        side = "right"
    else:
        side = "both"

    # Also return matches for full-image decoding
    cat_matches = []
    for _, name, prob in decoded_full:
        name_norm = name.replace("_", " ").lower()
        if "cat" in name_norm or any(k in name_norm for k in CAT_KEYWORDS):
            cat_matches.append((name, float(prob)))

    return decoded_full, cat_matches, side

def main():
    p = argparse.ArgumentParser(description="Detect whether an image contains a cat and which side it's on.")
    p.add_argument("--image", required=True, help="Path to the image file")
    args = p.parse_args()

    decoded, cats, side = detect_cat(args.image, top=5)
    print("Top predictions:")
    for cls, name, prob in decoded:
        print(f"  {name}: {prob:.3f}")

    if cats:
        best = max(cats, key=lambda x: x[1])
        print(f"Cat detected: yes (best: {best[0]} @ {best[1]:.3f})")
        print(f"Cat likely on: {side}")
    else:
        print("Cat detected: no")
        print(f"Cat likely on: {side}")

if __name__ == "__main__":
    main()