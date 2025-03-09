"""Simple test script to verify that the language model is working."""

import time

import dspy


def main() -> None:
    """Run tests to verify language model functionality."""
    print("Testing direct model access...")
    
    # Initialize the language model
    lm = dspy.LM('openrouter/google/gemini-2.0-flash-001')
    
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
    except Exception as e:
        print(f"Error calling model: {e}")
    
    # Test with DSPy Predict
    print("\nTesting with DSPy Predict...")
    
    # Define a simple signature
    signature = dspy.Signature("message -> response")
    
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
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
