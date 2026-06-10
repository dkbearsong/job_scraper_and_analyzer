# main.py
- 912-914, 989-990, 1063-1064, 1120-1121: should use the function error_logger_continue
- 854-865, 952-955, 1019-1020, 1079-1080: We appear to be using a separate instance directly to the database when one is already established at line 314. Will need to have that adjusted.
- 867-888, 957-969, 974-984, 1023-1037, 1043-1058, 1083-1094, 1098-1115: Should exist in pull_data.py



# vector_engine.py
- 8: KEYWORD_ADJUSTMENT needs to be modified to use .env vars
- 16: METADATA_ADJUSTMENT needs to be modified to use .env vars
- 

# llm_classifier.py
- Overall the system prompts may need revising. Will have to test and see how they do.
- 118-131, 234-251: _init_client need to make sure there's alot of different options available. Right now has openai, gemini, and lm studio and no other options
- 291-335: Can we condense this into smaller if/elses? Reuse some code as a function?
- 375-382: This should be condensable so that each of the keys are a checked and changed in a single line
- 394-401: Convert this into .env vars so users can customize them. Keep the current values as fallbacks
- 409-419: This appears as though it could be rewritten to be simpler and shorter, maybe easier to read. Reusable functions perhaps?
- Need to review pylance errors in this file. Several errors found regarding clients and properties within.


# Overall (no specific file)
- Should make sure there are also options for Ollama