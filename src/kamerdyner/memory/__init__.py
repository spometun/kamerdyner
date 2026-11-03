"""Memory of a user: the recent conversation, and what the assistant keeps of the older one.

`memory` is the interface (`Memory` and what compacting can come to); each implementation lives
in its own public module (`notes`). What they are built from is private, in `impl`: the ways of
compressing messages into memory text (`summary`) and the choice of the cut (`compaction`). An
implementation keeps everything of its own in one directory given to it, loads it from there when
created, and makes the helpers it needs itself. Whatever the way, long-term memory keeps both
sides of the conversation: the user first of all, and the assistant itself — what it said,
advised and promised.

The package depends on nothing of `kamerdyner` but `kamerdyner.common`, so that it can become a
package of its own."""
