---
name: reader-mindset
description: Pre-finalize pass for comments and code. The reader has only the diff, not your authoring path.
---

You overfit code and comments to the mental path YOU took to get there. The reader has only the artifact.

Before shipping, switch roles. Read as someone who has never seen the work-in-progress:

- Justifying a choice (author voice) vs answering a WHY the code can't (reader voice). Delete justifications.
- References to branches, callers, or history not visible from where they sit. Delete or relocate.
- Restating what the code does. Delete.
- "Could a careful reader infer this without the comment?" Yes: delete.

If you cannot speak the critique in the reader's voice, it's still in the author's voice. Pass again.
