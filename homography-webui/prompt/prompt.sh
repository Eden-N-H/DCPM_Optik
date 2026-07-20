#!/bin/bash
OF="$(pwd)/output.txt"
> "$OF"

# Source Code
cd ..
log_directory.sh static ".css" ".js" >> "$OF"
log_directory.sh templates ".html" >> "$OF"
log_files.sh requirements.txt >> "$OF"
log_directory.sh src ".py" >> "$OF"

# Prompt
cd prompt
cat task.md >> "$OF"

cat >> "$OF" << 'EOF'

FILE OUTPUT FORMAT

When providing complete file contents, follow this exact structure for every file:

1. On its own line, write the file path as inline code using single backticks,
   e.g. `src/cv_bev.py`
2. Nothing else may appear on that line — no "File:", no colon, no description.
3. Leave one blank line between the path line and the code fence.
4. Immediately follow with a triple-backtick fenced code block containing the
   full file contents, tagged with the appropriate language
   (e.g. ```python, ```javascript, ```json).
5. Exactly one file path precedes exactly one code block — never combine
   multiple files under one path, and never leave a code block without a
   preceding path line.
6. Use relative paths with forward slashes (e.g. src/utils/helpers.py),
   matching the project's actual directory structure.
7. Do not put the path inside the code block itself (e.g. as a comment) —
   the plain-text backticked path is what gets parsed.
EOF