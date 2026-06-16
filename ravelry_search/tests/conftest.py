"""
Ensure ravelry_search/ is on sys.path so test imports work when running
`pytest tests/` from ravelry_search/.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Silence the langfuse logger so the "Unexpected error occurred" message
# (logged via logging.error in langfuse/utils.py) does not appear in test
# output when the SDK can't reach the fake host.
logging.getLogger("langfuse").setLevel(logging.CRITICAL)
