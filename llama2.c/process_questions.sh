#!/bin/zsh

# Define input and output files
INPUT_CSV="questions.csv"
OUTPUT_CSV="responses.csv"
MODEL_BIN="./run"
TEMP_FILE="temp_output.txt"

# Ensure output file is cleared or created
echo "Response from Transformer Model" > "$OUTPUT_CSV"

# Function to process a single question
process_question() {
    local question="$1"
    echo "Processing question: $question"

    # Run the LLM binary
    $MODEL_BIN out/model.bin -i "Q: $question" > "$TEMP_FILE"

    # Check if temp file has content
    if [[ ! -s "$TEMP_FILE" ]]; then
        echo "Warning: No response generated for: $question"
        echo "\"\"" >> "$OUTPUT_CSV"
        return
    fi

    # Extract response using sed
    local response
    response=$(sed -n 's/^A: //p' "$TEMP_FILE")

    # Log the response
    if [[ -z "$response" ]]; then
        echo "Warning: No 'A:' response found for: $question"
        echo "\"\"" >> "$OUTPUT_CSV"
    else
        echo "\"$response\"" >> "$OUTPUT_CSV"
        echo "Finished processing: $question"
    fi
}

# Loop through each question in the input CSV
tail -n +2 "$INPUT_CSV" | while IFS=, read -r question; do
    # Skip empty lines
    if [[ -z "$question" ]]; then
        continue
    fi

    # Process the question
    process_question "$question"
done

# Clean up
rm -f "$TEMP_FILE"

echo "Processing completed. Results saved to $OUTPUT_CSV."
