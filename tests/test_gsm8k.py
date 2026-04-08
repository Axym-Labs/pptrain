from pptrain.eval.tasks.gsm8k import extract_final_number


def test_extract_final_number_uses_last_numeric_answer() -> None:
    text = "Reasoning...\nAnswer: 1,234"
    assert extract_final_number(text) == "1234"

