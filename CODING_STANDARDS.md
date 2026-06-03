# Coding Standards

Guidance for humans and AI coding agents working in this repository.
Project-specific tooling (Ruff, mypy strict, pytest layout) lives in
`CLAUDE.md` and `pyproject.toml`; this file covers the timeless principles
and the agent-specific guardrails that sit on top of them.

## Foundational principles

### DRY (Don't Repeat Yourself)
Every piece of knowledge should have a single, authoritative representation
in the system. Duplication is cheap to write and expensive to evolve: when
the rule changes, every copy must change in lock-step or the system goes
inconsistent.

The test for "should this be shared?" is not "do these blocks look
alike?" but **"when this logic changes, would every call site want to
change with it?"** If yes, factor it out; the call sites share a concept.
If no, the resemblance is incidental and merging them couples things
that should be free to evolve independently. Reuse only across genuinely
related contexts.

Two more caveats:

- Wait for the third occurrence before extracting. The first two might
  diverge in ways you can't see yet.
- If avoiding a repetition would require a giant refactor disproportionate
  to the benefit, repeat. A small, local duplication is cheaper to live
  with than a large, invasive abstraction.

### YAGNI (You Aren't Gonna Need It)
Do not build for hypothetical future needs. Implement what the current
task requires; add the next thing when the next thing is actually asked
for. Speculative generality (config knobs nobody sets, plugin points
nobody plugs into, abstract base classes with one subclass) costs
maintenance forever and rarely fits the future requirement when it
arrives.

### KISS (Keep It Simple)
Prefer the simplest design that satisfies the requirements. Complexity
should be justified by a concrete problem it solves, not by elegance,
symmetry, or "good engineering" in the abstract. A flat function beats a
class hierarchy when no polymorphism is needed.

### SOLID (when designing OO interfaces)
1. **Single Responsibility**; a module changes for one reason only.
2. **Open/Closed**; open to extension, closed to modification (achieve via
   composition, not deep inheritance).
3. **Liskov Substitution**; subtypes must honor the contract of their
   supertype.
4. **Interface Segregation**; many small, focused interfaces beat one fat
   one.
5. **Dependency Inversion**; depend on abstractions, not concretions, at
   module boundaries.

Apply SOLID at module boundaries; do not impose it on a 20-line script.

### Boy Scout Rule
Leave the code cleaner than you found it. Small, in-scope improvements
(rename a confusing variable, delete a dead branch, tighten a type) are
welcome on the way past. Do not turn this into "refactor the whole file
while fixing a typo"; keep the diff focused.

### Fail fast, validate at boundaries
Validate untrusted input (CLI args, file contents, network payloads) at
the system edge. Inside the trusted core, assume invariants hold and let
exceptions propagate; defensive `if x is None` checks scattered through
internal code hide bugs rather than prevent them.

### Separation of concerns
Keep I/O, computation, and presentation in distinct functions or modules.
Pure functions are easy to test; functions that mix file reads, math, and
logging are not.

## Style and craft

- **Names carry the docs.** A clear name removes the need for a comment.
  If you need a comment to explain *what* a function does, the name is
  wrong.
- **Comments explain *why*, not *what*.** The *what* is in the code
  itself; restating it is noise. Default to no inline comment. Write one
  only when the reason is non-obvious: a hidden constraint, a workaround
  for a specific bug, surprising behavior, a load-bearing invariant.
- **Docstrings: Google style, what *and* why, length proportional to the
  function.** Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
  (`Args:`, `Returns:`, `Raises:`). Unlike inline comments, docstrings
  cover both *what* the function does (so callers don't have to read the
  body) and *why* it exists (the context, the constraint, the role it
  plays). Scale the docstring to the function: a one-line helper gets a
  one-line summary; a 200-line algorithm gets a paragraph plus full
  `Args`/`Returns`/`Raises` blocks. Public API: always documented.
  Private one-liners: a summary line is enough.
- **Type everything public.** This package ships `py.typed`; downstream
  consumers get our annotations. Public functions, classes, and module
  attributes are annotated. `mypy --strict` must pass.
- **Tests are part of the API.** A function without a test is a function
  whose contract is undefined. Write the test alongside the change, not
  "later".
- **No dead code.** Delete unused branches, unused parameters, unused
  imports, unused files. Git remembers; the working tree should not.

## Pragmatism and dependencies

- **Use the language.** Python is multi-paradigm; not every function
  needs a class or an object. Free functions, modules, and `@dataclass`
  cover most of what a class hierarchy would, with less ceremony. Reach
  for OO when you genuinely need state plus behavior or runtime
  polymorphism, not by default.
- **Reuse before reimplementing.** Check the standard library first
  (`itertools`, `functools`, `collections`, `pathlib`, `statistics`,
  `dataclasses`, etc.) and then existing project dependencies before
  writing your own version. Hand-rolling `defaultdict`, a half-correct
  `sorted` key, or your own `Path.glob` is wasted effort and a bug farm.
- **Weigh each dependency against what you use from it.** A new
  dependency is forever: install time, attack surface, version pins,
  someone else's release cadence, and a transitive blast radius you
  inherit. Do not pull in a 100 MB library for one 100-line function;
  copy the function into the codebase (with attribution and a
  license-compatibility check) or reimplement it. The flip side: do not
  hand-roll something a well-maintained library already solves
  correctly (parsers, numerical kernels, crypto, HTTP). Match the
  dependency to the surface area you actually use.

## Guidance for AI coding agents

Agents operate without the surrounding human context (PR description,
Slack thread, in-flight conversation), so these guardrails matter more
for agents than for humans.

1. **Match scope.** Do exactly what was asked. A bug fix is not an
   invitation to refactor the file; a typo correction is not the time to
   modernize imports. If you see other issues, mention them in the
   summary; do not silently expand the diff.
2. **Read before you write.** Before editing, read the file and the
   nearest tests. Before adding a helper, grep for an existing one.
   Re-implementing what already exists is the most common agent failure
   mode.
3. **Prefer editing to creating.** New files fragment the codebase. Add
   to an existing module unless the new thing is genuinely a new concern.
4. **No speculative abstractions.** Concrete first. Three call sites
   justify a helper; one does not. If the abstraction is wrong, it is
   harder to undo than to never have written.
5. **No backwards-compat shims for code that has never shipped.** While
   the codebase is pre-1.0 and unreleased, rename freely, delete freely,
   change signatures freely. Do not leave `_old_name = new_name` aliases
   "just in case".
6. **State assumptions explicitly.** When a task is ambiguous, pick the
   reasonable default, do the work, and call out the assumption in the
   summary so the human can redirect. Do not silently choose.
7. **Verify with the tools, not by inspection.** `pytest`, `ruff check`,
   `mypy` are the source of truth. "It looks right" is not a pass.
8. **Keep commits and PRs scoped.** One logical change per commit. The
   commit message says *why*; the diff shows *what*.
9. **Don't narrate the obvious in code.** Comments like
   `# increment counter` above `counter += 1` are noise. Save commentary
   for non-obvious *why*.
10. **When in doubt, do less.** A smaller, correct change is always
    better than a larger, plausible one.

## Anti-patterns to avoid

- Catching `Exception` to "be safe". Catch the specific exception you
  expect; let the rest surface.
- Wrapping a one-line call in a helper "for readability". The helper
  *is* a layer; the one-liner was readable.
- Adding a config flag to choose between two behaviors when only one is
  ever used.
- Comments that restate the next line in English.
- `TODO` comments without an owner or a condition for resolution; they
  become permanent furniture.
- Tests that mock the system under test (you end up asserting on the
  mock, not the code).
- **Deep nesting** of `try` / `if` / `with` / `for`. Three-plus levels of
  indentation is usually a signal to refactor, not a structure to live
  with. Common fixes: early returns and guard clauses to flatten `if`
  pyramids, extract the inner block into a named function, combine
  `with` statements onto one line (`with open(a) as fa, open(b) as fb:`),
  narrow `try` blocks to the single call that can actually raise, and
  replace nested loops with a comprehension or `itertools.product`. If
  the nesting genuinely reflects the problem (a state machine, a parser),
  that is fine; if it just accumulated, flatten it.

## Further reading

- [KISS, DRY, SOLID, YAGNI primer (HlfDev, Medium)](https://medium.com/@hlfdev/kiss-dry-solid-yagni-a-simple-guide-to-some-principles-of-software-engineering-and-clean-code-05e60233c79f)
- [Software engineering principles cheatsheet (Tuanhadev)](https://blog.tuanhadev.tech/software-engineering-principles-cheatsheet)
- [AGENTS.md (open format for agent guidance)](https://agents.md/)
- [Writing a good CLAUDE.md (HumanLayer)](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Coding Guidelines for Your AI Agents (JetBrains)](https://blog.jetbrains.com/idea/2025/05/coding-guidelines-for-your-ai-agents/)
- [Building shared coding guidelines for AI and people (Stack Overflow blog)](https://stackoverflow.blog/2026/03/26/coding-guidelines-for-ai-agents-and-people-too/)
