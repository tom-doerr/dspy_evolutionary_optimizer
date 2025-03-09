"""Simple test script to verify that the language model is working."""

import time
import pytest
import dspy


def main() -> None:
    """Run tests to verify language model functionality."""
    print("Testing direct model access...")

    # Initialize the language model
    lm = dspy.LM("openrouter/google/gemini-2.0-flash-001")

    # Explicitly disable caching
    dspy.settings.configure(lm=lm, cache=False)

    # Make a direct call to the model
    print("Making direct call to the model...")
    start_time = time.time()
    try:
        response = lm("Hello, can you hear me? This is a test message.")
        elapsed = time.time() - start_time
        print(f"Call successful! Took {elapsed:.2f} seconds")
        print(f"Response: {response}")
    except (ConnectionError, RuntimeError) as e:
        print(f"Error calling model: {e}")

    # Test with DSPy Predict
    print("\nTesting with DSPy Predict...")

    # Define a simple signature
    signature = dspy.Signature("message -> response")
    signature.__doc__ = "Given a message, generate a response"

    # Create a predictor
    predictor = dspy.Predict(signature)

    # Make a prediction
    print("Making prediction with DSPy Predict...")
    start_time = time.time()
    try:
        result = predictor(message="Hello, this is a test message from DSPy Predict.")
        elapsed = time.time() - start_time
        print(f"Prediction successful! Took {elapsed:.2f} seconds")
        print(f"Result: {result.response}")
    except (ConnectionError, RuntimeError) as e:
        print(f"Error making prediction: {e}")


def test_model_error_handling():
    """Test error handling in model interactions."""
    # Test invalid model name
    with pytest.raises(ValueError, match="Invalid model name"):
        dspy.LM("invalid/model/name")

    # Test connection timeout
    lm = dspy.LM("openrouter/google/gemini-2.0-flash-001")
    dspy.settings.configure(lm=lm, cache=False, timeout=0.001)  # Set very low timeout
    with pytest.raises((TimeoutError, ConnectionError)):
        lm("Test message")

    # Test invalid input type
    with pytest.raises(TypeError):
        lm(12345)

    # Test empty input
    with pytest.raises(ValueError):
        lm("")
    # Test connection timeout
    lm = dspy.LM("openrouter/google/gemini-2.0-flash-001")
    dspy.settings.configure(lm=lm, cache=False, timeout=0.001)  # Set very low timeout
    with pytest.raises(Exception):
        lm("Test message")
    # Test connection timeout
    lm = dspy.LM("openrouter/google/gemini-2.0-flash-001")
    dspy.settings.configure(lm=lm, cache=False, timeout=0.001)  # Set very low timeout
    with pytest.raises(Exception):
        lm("Test message")
    # Test connection timeout
    lm = dspy.LM("openrouter/google/gemini-2.0-flash-001")
    dspy.settings.configure(lm=lm, cache=False, timeout=0.001)  # Set very low timeout
    with pytest.raises(Exception):
        lm("Test message")
    # Test connection timeout
    lm = dspy.LM("openrouter/google/gemini-2.0-flash-001")
    dspy.settings.configure(lm=lm, cache=False, timeout=0.001)  # Set very low timeout
    with pytest.raises(Exception):
        lm("Test message")

    # Test connection timeout
    lm = dspy.LM("openrouter/google/gemini-2.0-flash-001")
    dspy.settings.configure(lm=lm, cache=False, timeout=0.001)  # Set very low timeout
    with pytest.raises(Exception):
        lm("Test message")


def test_predictor_error_handling():
    """Test error handling in DSPy Predict."""
    # Test invalid signature
    with pytest.raises(TypeError, match="signature must be a dspy.Signature"):
        dspy.Predict(None)

    # Test invalid input
    signature = dspy.Signature("text -> response", "Given text, generate a response")
    signature.__doc__ = "Given text, generate a response"
    predictor = dspy.Predict(signature)
    with pytest.raises(TypeError, match="Input must be a dictionary"):
        predictor(None)

    # Test missing required input field
    with pytest.raises(ValueError, match="Missing required input field 'text'"):
        predictor({})
    # Test invalid input
    signature = dspy.Signature("text -> response")
    signature.__doc__ = "Given text, generate a response"
    predictor = dspy.Predict(signature)
    with pytest.raises(TypeError):
        predictor(None)
    # Test invalid input
    signature = dspy.Signature("text -> response")
    signature.__doc__ = "Given text, generate a response"
    predictor = dspy.Predict(signature)
    with pytest.raises(TypeError):
        predictor(None)
    # Test invalid input
    signature = dspy.Signature("text -> response")
    signature.__doc__ = "Given text, generate a response"
    predictor = dspy.Predict(signature)
    with pytest.raises(TypeError):
        predictor(None)
    # Test invalid input
    signature = dspy.Signature("text -> response")
    signature.__doc__ = "Given text, generate a response"
    predictor = dspy.Predict(signature)
    with pytest.raises(TypeError):
        predictor(None)
    # Test invalid input
    signature = dspy.Signature("text -> response")
    signature.__doc__ = "Given text, generate a response"
    predictor = dspy.Predict(signature)
    with pytest.raises(TypeError):
        predictor(None)

    # Test invalid input
    signature = dspy.Signature("text -> response")
    signature.__doc__ = "Given text, generate a response"
    predictor = dspy.Predict(signature)
    with pytest.raises(TypeError):
        predictor(None)


if __name__ == "__main__":
    main()
