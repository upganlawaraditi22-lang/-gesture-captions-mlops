from gesture_captions.captions import CaptionBuilder
from gesture_captions.language import gloss_text, translate
from gesture_captions.predictor import Prediction


def feed(builder, label, frames, confidence=0.9):
    for _ in range(frames):
        builder.update(Prediction(label, confidence))


def test_word_needs_steady_frames():
    builder = CaptionBuilder(stable_frames=3)
    feed(builder, "hello", 2)
    assert builder.text == ""
    feed(builder, "hello", 1)
    assert builder.text == "hello"


def test_held_gesture_adds_one_word():
    builder = CaptionBuilder(stable_frames=3)
    feed(builder, "hello", 50)
    assert builder.text == "hello"


def test_low_confidence_is_ignored_and_does_not_repeat_word():
    builder = CaptionBuilder(confidence_threshold=0.7, stable_frames=2)
    feed(builder, "hello", 2)
    feed(builder, "hello", 1, confidence=0.2)
    feed(builder, "hello", 5)
    assert builder.text == "hello"


def test_word_repeats_after_no_gesture():
    builder = CaptionBuilder(stable_frames=2)
    feed(builder, "hello", 2)
    feed(builder, "NOTHING", 2)
    feed(builder, "hello", 2)
    assert builder.text == "hello hello"


def test_brief_hand_loss_does_not_repeat_word():
    builder = CaptionBuilder(stable_frames=3)
    feed(builder, "hello", 3)
    feed(builder, "NOTHING", 1)
    feed(builder, "hello", 5)
    assert builder.text == "hello"


def test_different_gestures_build_a_sentence():
    builder = CaptionBuilder(stable_frames=2)
    feed(builder, "thank", 2)
    feed(builder, "you", 2)
    assert builder.text == "thank you"


def test_caption_length_is_capped():
    builder = CaptionBuilder(stable_frames=1, max_words=2)
    for label in ("a", "b", "c"):
        feed(builder, label, 1)
    assert builder.text == "b c"


def test_clear_resets_state():
    builder = CaptionBuilder(stable_frames=1)
    feed(builder, "hello", 1)
    builder.clear()
    assert builder.text == ""
    feed(builder, "hello", 1)
    assert builder.text == "hello"


def test_signs_become_sentences_in_each_language():
    words = ["I", "WANT", "WATER", "PLEASE"]
    assert translate(words, "en") == "I would like some water, please."
    assert "पानी" in translate(words, "hi")
    assert "पाणी" in translate(words, "mr")
    assert gloss_text(["LETTER_R", "LETTER_A", "LETTER_M"]) == "RAM"
