# Async Python and APIs

This document covers `async`/`await`, the event loop, tasks, blocking vs
non-blocking work, and the fundamentals of working with HTTP APIs.

## Why async exists

A normal ("synchronous") function that waits on a network response blocks
the entire program while it waits — nothing else can run. For a program
that spends most of its time waiting on I/O (network requests, database
queries, file operations) rather than computing, that's wasted time: the
CPU sits idle while one request is in flight instead of starting the
next one. `asyncio` lets a single thread juggle many such waits
concurrently, switching to other work whenever the current one is
blocked on I/O.

This is concurrency, not parallelism: `asyncio` code still runs on one
thread. It helps enormously with I/O-bound waiting; it does nothing for
CPU-bound work (a tight numeric loop still blocks everything else,
exactly as it would without `asyncio` — see "Blocking vs non-blocking
work" below).

## Coroutines, `async`/`await`

A function defined with `async def` is a coroutine function; calling it
doesn't run its body immediately — it returns a coroutine object that
must be `await`ed (or scheduled as a Task) to actually execute:

```python
import asyncio

async def fetch_greeting(name):
    await asyncio.sleep(1)   # simulates an I/O wait, not real work
    return f"Hello, {name}!"

async def main():
    result = await fetch_greeting("Ada")
    print(result)

asyncio.run(main())
```

`await` means "suspend this coroutine here, let the event loop run
something else, and resume me once this awaited thing completes." You
can only `await` inside an `async def` function.

## The event loop

The event loop is the scheduler that drives every coroutine: it tracks
which coroutines are waiting on what, and resumes each one the instant
whatever it's waiting for (a timer, a socket becoming readable, another
coroutine finishing) is ready. `asyncio.run(main())` creates a fresh event
loop, runs `main()` to completion, then closes the loop — this is the
standard, recommended top-level entry point for an asyncio program.

## Tasks

`await`ing a coroutine directly runs it and waits for it before moving
on — strictly sequential. To run multiple coroutines *concurrently*, wrap
each in a `Task` (which schedules it on the event loop immediately) or
use `asyncio.gather`:

```python
async def fetch(n):
    await asyncio.sleep(1)
    return n * n

async def sequential():
    a = await fetch(1)   # waits 1s
    b = await fetch(2)   # then waits another 1s — total ~2s
    return a, b

async def concurrent():
    task_a = asyncio.create_task(fetch(1))
    task_b = asyncio.create_task(fetch(2))
    a = await task_a
    b = await task_b      # both slept concurrently — total ~1s
    return a, b

async def concurrent_with_gather():
    a, b = await asyncio.gather(fetch(1), fetch(2))   # same idea, less code
    return a, b
```

A `Task` starts running as soon as it's created (well, at the next
opportunity the event loop gets), independent of whether/when you
`await` it. `asyncio.gather()` is a convenient way to schedule several
awaitables and collect all their results together.

## Cancellation basics

Any `Task` can be cancelled with `task.cancel()`, which schedules a
`CancelledError` to be raised inside the coroutine at its next suspension
point (its next `await`) — not instantly, and not while the coroutine is
in the middle of a synchronous (non-`await`) stretch of code:

```python
async def worker():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("cleaning up before exiting")
        raise   # convention: re-raise after cleanup, don't swallow it

task = asyncio.create_task(worker())
await asyncio.sleep(0.1)
task.cancel()
```

Two details that matter in real programs:

- Cancelling a `Task` does **not** stop code that's already running on a
  separate OS thread (e.g. inside `asyncio.to_thread(...)`) — Python
  cannot forcibly interrupt a running thread. The `Task` wrapper can flip
  to "cancelled" while the underlying thread keeps running to completion
  regardless.
- A coroutine can be cancelled more than once in a row before it finishes
  reconciling the first cancellation. Code with real cleanup
  responsibilities (closing a file, releasing a lock, rolling back a
  partial write) needs to be written with that in mind, not just handle a
  single `CancelledError` naively.

## Blocking vs non-blocking work

`asyncio` cooperative multitasking only works if coroutines actually
yield control back to the event loop at `await` points. A call that
blocks synchronously — a CPU-heavy loop, a synchronous file read, a
synchronous HTTP library call — freezes the *entire* event loop for its
duration, starving every other coroutine, even ones that were otherwise
ready to run:

```python
async def bad():
    time.sleep(5)          # BLOCKS the whole event loop for 5 seconds

async def good():
    await asyncio.sleep(5)  # yields control; other coroutines keep running
```

For genuinely blocking work you can't avoid (a CPU-bound computation, a
library with no async API), move it off the event loop entirely:

```python
result = await asyncio.to_thread(blocking_function, arg1, arg2)
```

`asyncio.to_thread()` runs `blocking_function` in a separate worker
thread and lets the event loop keep servicing everything else while it
runs. For genuinely CPU-bound work (not I/O-bound), a thread doesn't
bypass Python's Global Interpreter Lock (GIL) — a `ProcessPoolExecutor`
is the right tool when you need true parallel CPU work instead.

## HTTP/API fundamentals

Most external services are exposed as HTTP APIs. A request has a method
(`GET` to read, `POST` to create, `PUT`/`PATCH` to update, `DELETE` to
remove), a URL, headers (metadata like authentication tokens and content
type), and optionally a body (usually JSON). The response has a status
code:

- `2xx` — success (`200 OK`, `201 Created`, `204 No Content`)
- `3xx` — redirection
- `4xx` — the client's fault (`400 Bad Request`, `401 Unauthorized`,
  `404 Not Found`, `429 Too Many Requests`)
- `5xx` — the server's fault (`500 Internal Server Error`,
  `503 Service Unavailable`)

Never assume a request succeeded just because it didn't raise a network
exception — check the status code (or let your HTTP client raise on
non-2xx, e.g. `response.raise_for_status()`).

## Timeouts

Every network call needs a timeout. Without one, a single unresponsive
server can hang your program indefinitely — worse than an outright
failure, because it gives no signal that anything is wrong:

```python
import asyncio
import aiohttp

async def fetch_json(url):
    timeout = aiohttp.ClientTimeout(total=10)  # seconds
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
```

`asyncio.wait_for(coro, timeout=...)` applies a timeout to any awaitable,
cancelling it and raising `asyncio.TimeoutError` if it doesn't finish in
time — useful even when the underlying call has no timeout option of its
own.

## Concurrency with rate limits

Calling `asyncio.gather()` on hundreds of requests at once can overwhelm
either your own network stack or the remote server's rate limits. An
`asyncio.Semaphore` caps how many run concurrently:

```python
async def fetch_all(urls, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_fetch(url):
        async with semaphore:
            return await fetch_json(url)

    return await asyncio.gather(*(bounded_fetch(url) for url in urls))
```

## Common mistakes

- **Calling a coroutine function without `await`ing it.** `fetch_greeting("Ada")`
  alone (no `await`, not wrapped in a `Task`) just creates a coroutine
  object and does nothing — Python usually warns
  "coroutine was never awaited," which is easy to miss in noisy output.
- **Blocking the event loop** with a synchronous call inside an `async
  def` function (`time.sleep`, a synchronous `requests.get`, heavy
  synchronous computation) — see "Blocking vs non-blocking work" above.
- **No timeout on network calls**, leaving the program hung on an
  unresponsive peer with no way to recover automatically.
- **Assuming cancellation is instantaneous or single-shot.** A `Task` only
  observes cancellation at its next `await`, can require handling being
  cancelled more than once before cleanup completes, and cannot forcibly
  stop code already running on a separate thread.
- **Mixing `asyncio.run()` calls inside already-running async code.**
  `asyncio.run()` creates its own event loop and fails if one is already
  running in the current thread — inside an existing coroutine, `await`
  the awaitable directly instead.
