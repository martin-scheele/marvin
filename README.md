# Marvin

Marvin is a hypothetical computer with sixteen 32-bit registers and 64 KB of main memory (RAM). In addition to
the sixteen registers, Marvin has a program counter `pc` and an instruction register `ir`. A Marvin program
(ie, a `.marv` file) is assembled and loaded into memory starting at location 0. Marvin supports 72
instructions, each of which accepts between 0 and 3 arguments (aka inputs). See the
[Marvin Machine Specification](https://www.cs.umb.edu/~siyer/teaching/marvin_machine.pdf) for the full
description of the machine, including its instruction encoding.

The program `marvin.py` is an emulator for the Marvin machine. Here is the usage string for the emulator:

```
$ python3 marvin.py -h

usage: marvin.py [-h] [-d] [-c] [-a ARGS [ARGS ...]] filename

This program serves as an emulator for a register-based machine called Marvin
(named after the paranoid android character, Marvin, from The Hitchhiker's
Guide to the Galaxy by Douglas Adams). The program accepts a .marv file as
input, assembles and simulates the instructions within, and prints any output
to stdout. Any input to the .marv program is via stdin.

positional arguments:
  filename              input .marv file

options:
  -h, --help            show this help message and exit
  -d, --debug           enable debug mode
  -c, --count           count instruction calls
  -a ARGS [ARGS ...], --args ARGS [ARGS ...]
                        pass cli arguments
```

Here is a sample Marvin program called `HelloWorld.marv` (in `tests/`) that declares a string in the
`.data` section and prints it with `writes`.

```
# Declares a string in .data and prints it with writes.
# Exercises: .data strings, lda, writes.

.data
.string msg = "Hello, World!\n"

.text
main:
    lda r1 msg       # r1 = address of msg
    writes r1
    halt
```

Here is the output from running `HelloWorld.marv` using `marvin.py`:

```
$ python3 marvin.py tests/HelloWorld.marv
Hello, World!
```

## The Machine

Marvin is a small 32-bit machine: **16 registers**, **64 KB of memory**, and **72 instructions**. A word is
4 bytes; shorts are 2 bytes; characters are UTF-16 code units. The stack grows **downward** from the top of
memory.

| Region        | Addresses             | Holds                                          |
| ------------- | --------------------- | ---------------------------------------------- |
| Text          | 0x0000–0x1FFF         | assembled instructions (pc moves through here) |
| Heap / data   | 0x2000–↑              | .data variables, then arrays you allocate (gp) |
| Stack         | ↓–0xFFFC              | pushw/pushs/pushb frames (sp)                  |

## Registers

| Name  | Alias | Conventional use                                                  |
| ----- | ----- | ----------------------------------------------------------------- |
| r0–r2 | a0–a2 | arguments / scratch                                               |
| r11   | ra    | return address (`jsr ra sub`)                                     |
| r12   | rv    | return value; also read status: 0 = ok, 1 = EOF                   |
| r13   | fp    | frame pointer                                                     |
| r14   | sp    | stack pointer (starts at 0xFFFC, grows down)                      |
| r15   | gp    | heap pointer (top of allocated data; bump to allocate)            |

These aliases are conventions, not hardware — nothing stops you from using r12 as a loop counter, but
following them keeps programs readable. The remaining registers r3–r10 have no aliases.

## Writing Programs

A program has a `.data` section (variables) and a `.text` section (instructions). Labels end with `:` on
their own line (an inline comment after the colon is fine); `#` starts a comment anywhere. Data lines look
like `.int n = 42` — the directives are:

| Directive             | Meaning                                                           |
| --------------------- | ----------------------------------------------------------------- |
| .byte / .short / .int | integer variable (stored as a full word)                          |
| .float                | floating-point variable (32-bit float bits)                       |
| .char                 | single character (stored as a short)                              |
| .string               | UTF-16 string, printed with `writes`; use `\n` for newline        |

Instruction operands are registers (`r0`–`r15` or aliases) except where an immediate is allowed — `seti r1 42`,
`inci r1 -1`, `j loop`, `lda r1 msg`. There is no `add`: integer arithmetic is `addi`, `subi`, `muli`, `divi`,
`modi`, all taking three **registers**.

## Instruction Reference

### Input
| Instruction | Description |
| ----------- | ----------- |
| `readi rX`  | read an integer from stdin into rX; rv = 0 on success, rv = 1 at EOF (rX unchanged) |
| `readf rX`  | read a float from stdin into rX; rv = 0 on success, rv = 1 at EOF (rX unchanged)    |
| `readc rX`  | read a character from stdin into rX; rv = 0 on success, rv = 1 at EOF (rX unchanged) |

### Output
| Instruction | Description |
| ----------- | ----------- |
| `writei rX` | print rX as an integer, followed by a newline |
| `writef rX` | print rX as a float, followed by a newline    |
| `writec rX` | print the low 16 bits of rX as one character (no newline) |
| `writes rX` | print the UTF-16 string whose length-prefixed descriptor is at address rX |

### Integer Arithmetic
| Instruction | Description |
| ----------- | ----------- |
| `addi rX rY rZ` | rX ← rY + rZ (integers) |
| `subi rX rY rZ` | rX ← rY − rZ (integers) |
| `muli rX rY rZ` | rX ← rY × rZ (integers) |
| `divi rX rY rZ` | rX ← rY ÷ rZ (integer division, truncated) |
| `modi rX rY rZ` | rX ← rY mod rZ (remainder) |
| `negi rX rY`    | rX ← −rY |

### Float Arithmetic
| Instruction | Description |
| ----------- | ----------- |
| `addf rX rY rZ` | rX ← rY + rZ (floats) |
| `subf rX rY rZ` | rX ← rY − rZ (floats) |
| `mulf rX rY rZ` | rX ← rY × rZ (floats) |
| `divf rX rY rZ` | rX ← rY ÷ rZ (floats) |
| `negf rX rY`    | rX ← −rY (float) |

### Logic & Shifts
| Instruction | Description |
| ----------- | ----------- |
| `and rX rY rZ`  | rX ← rY bitwise-AND rZ |
| `or rX rY rZ`   | rX ← rY bitwise-OR rZ  |
| `xor rX rY rZ`  | rX ← rY bitwise-XOR rZ |
| `not rX rY`     | rX ← bitwise complement of rY |
| `lshl rX rY rZ` | rX ← rY shifted left by rZ bits, zeros shifted in |
| `lshr rX rY rZ` | rX ← rY shifted right by rZ bits, zeros shifted in |
| `ashl rX rY rZ` | rX ← rY shifted left by rZ bits, sign preserved |
| `ashr rX rY rZ` | rX ← rY shifted right by rZ bits, sign preserved |

### Register Moves
| Instruction | Description |
| ----------- | ----------- |
| `seti rX n`  | rX ← immediate n (16-bit) |
| `inci rX n`  | rX ← rX + n (16-bit immediate) |
| `copy rX rY` | rX ← rY |

### Control Flow
| Instruction      | Description |
| ---------------- | ----------- |
| `j label`        | jump to label |
| `jr rX`          | jump to the address in rX |
| `jeqz rX label`  | jump to label if rX = 0  |
| `jnez rX label`  | jump to label if rX ≠ 0  |
| `jeq rX rY label` | jump to label if rX = rY  |
| `jne rX rY label` | jump to label if rX ≠ rY  |
| `jgt rX rY label` | jump to label if rX > rY   |
| `jge rX rY label` | jump to label if rX ≥ rY   |
| `jlt rX rY label` | jump to label if rX < rY   |
| `jle rX rY label` | jump to label if rX ≤ rY   |
| `jsr rX label`   | call: rX ← return address (pc + 4), then jump to label |

### Stack
| Instruction | Description |
| ----------- | ----------- |
| `pushb rX rY` | store low byte of rX at address rY, then rY ← rY − 1 (stack grows down) |
| `pushs rX rY` | store low short of rX at address rY, then rY ← rY − 2 (stack grows down) |
| `pushw rX rY` | store word rX at address rY, then rY ← rY − 4 (stack grows down) |
| `popb rX rY`  | rY ← rY + 1, then rX ← byte loaded from address rY |
| `pops rX rY`  | rY ← rY + 2, then rX ← short loaded from address rY |
| `popw rX rY`  | rY ← rY + 4, then rX ← word loaded from address rY |

### Memory
| Instruction | Description |
| ----------- | ----------- |
| `lda rX addr` | rX ← address of the named variable |
| `ldb rX rY n` | rX ← byte loaded from address rY + n |
| `lds rX rY n` | rX ← short loaded from address rY + n |
| `ldw rX rY n` | rX ← word loaded from address rY + n |
| `stb rX rY n` | store low byte of rX at address rY + n |
| `sts rX rY n` | store low short of rX at address rY + n |
| `stw rX rY n` | store word rX at address rY + n |

### Arrays
| Instruction | Description |
| ----------- | ----------- |
| `anewb rX rY` | initialize a byte array at rX: store length rY as a short header, then rY zeroed bytes |
| `anews rX rY` | initialize a short array at rX: store length rY as a short header, then rY zeroed shorts |
| `aneww rX rY` | initialize a word array at rX: store length rY as a short header, then rY zeroed words |
| `aldb rX rY rZ` | rX ← element rZ of the byte array at rY (bounds-checked) |
| `alds rX rY rZ` | rX ← element rZ of the short array at rY (bounds-checked) |
| `aldw rX rY rZ` | rX ← element rZ of the word array at rY (bounds-checked) |
| `astb rX rY rZ` | store low byte of rX into element rZ of the byte array at rY (bounds-checked) |
| `asts rX rY rZ` | store low short of rX into element rZ of the short array at rY (bounds-checked) |
| `astw rX rY rZ` | store word rX into element rZ of the word array at rY (bounds-checked) |
| `alen rX rY`    | rX ← length (short header) of the array at rY |

### Conversions
| Instruction | Description |
| ----------- | ----------- |
| `i2f rX rY` | rX ← float value of integer rY |
| `i2c rX rY` | rX ← character code of integer rY (identity conversion) |
| `f2i rX rY` | rX ← integer value of float rY (truncated) |

### System
| Instruction    | Description |
| -------------- | ----------- |
| `time rX`      | rX ← current time in milliseconds since midnight |
| `date rX`      | rX ← today's date packed as (year << 13) \| (month << 9) \| day |
| `seed rX`      | seed the random number generator with rX |
| `rand rX rY rZ` | rX ← random integer in [rY, rZ] |
| `nop`          | do nothing |
| `halt`         | stop the machine |

## Command-Line Arguments

Pass arguments to the program with `-a` (parsed at assembly time). Inside the program they appear as `argc`
and `argv`:

```
    lda r1 argc
    lds r2 r1 0        # r2 = argc
    lda r3 argv        # r3 = address of the argv array (shorts)
    seti r4 0
    alds r5 r3 r4      # r5 = address of argv[0] (a string -> writes prints it)
```

Arguments are **strings**; the SumInts example shows how to parse one into an integer.

## Standard Input and EOF

Programs read from stdin with `readi` / `readf` / `readc`: type a line and press Enter. Illegal input
re-prompts inside the same read. Press **Ctrl+D** (or close the pipe) to signal EOF — the pending and all
future reads then complete with **rv = 1** and leave their register unchanged (a successful read sets
rv = 0):

```
loop:
    readf r3           # rv = 1 once input is exhausted
    jnez rv done
    addf r2 r2 r3      # sum += x
    j loop
done:
```

See `Average.marv` for a complete program.

## Arrays and the Heap

An array is a **short length header** followed by the elements. `aneww r1 r2` initializes a word array at
address r1 with r2 elements (zero-filled); `aldw`/`astw` (and the byte/short variants) access elements
**with bounds checking**; `alen` reads the length.

Allocate by bumping `gp`: save the old gp as the array's base, then advance gp past the array (header +
elements). For a word array of n elements that is `2 + 4n` bytes.

Multi-dimensional arrays are flattened: a rows×cols matrix is a 1-D array of length rows×cols in row-major
order, and element [i][j] sits at flat index `i*cols + j` (two instructions: `muli` + `addi`).

## Subroutines

`jsr ra sub` stores the return address (pc + 4) in ra and jumps; `jr ra` returns. Arguments and results move
through registers by convention — commonly r1–r4 for arguments, rv (r12) for results. Helpers that clobber
registers should save them with push/pop. There is no hardware stack frame; fp/sp discipline is up to you.

## Example Programs

The `tests/` directory contains a suite of example programs:

| Example | What it demonstrates |
| ------- | -------------------- |
| Average.marv | reads floats until EOF and prints their average: `readf`, `addf`, `divf` loop |
| BitOps.marv | `and` / `or` / `xor` / `not`, the four shifts, `modi`, `negi`, `nop` |
| Cipher.marv | Caesar cipher built in a short array: `anews` / `alds` / `asts` / `alen` |
| Combinations.marv | C(n, k) via a factorial subroutine called three times: `jsr` / `jr`, `divi`, `subi` |
| Copy.marv | scalar loads and stores at all widths: `ldw`/`lds`/`ldb`, `stw`/`sts`/`stb` |
| DateTime.marv | prints the current date and time: `date` / `time` packing, `writec` digit output |
| EchoArgs.marv | echoes the command-line arguments: `argc` / `argv` (run with `-a ...`) |
| Factorial.marv | recursive n!: `fp`/`ra` stack frames |
| FlipCase.marv | flips letter case until EOF: `readc` / `writec`, byte arrays, `xor`, `i2c` |
| Grades.marv | letter grades for scores read until EOF: comparison jumps `jge` / `jgt` / `jle` / `jne` |
| Greet.marv | greets you by name: `argv` strings, string concatenation (run with `-a <name>`) |
| HelloWorld.marv | the classic: `.data` strings, `writes` |
| MatMul.marv | 2×3 by 3×2 matrix product: 2-D array as a flat row-major word array |
| MatMulSub.marv | same product, computed by `jsr` helper subroutines |
| Matrix.marv | 3×4 matrix fill and row sums: 2-D indexing as `i*cols + j` |
| PI.marv | π via the Nilakantha series: float ops, `i2f` / `f2i` |
| StackMix.marv | mixed-size push and pop: `pushb`/`pushs`/`popb`/`pops` |
| SumInts.marv | sum 1..n where n is a command-line arg: `argc`/`argv`, atoi from a string |
| Temperature.marv | Celsius/Fahrenheit conversions: `subf` / `negf`, float comparisons |
| TwentyQuestions.marv | number-guessing game: `seed` / `rand`, interactive stdin, EOF quits |

## The Browser Simulator

A browser-based simulator (plain HTML/CSS/JavaScript) implements the same machine with an IDE-style UI:
editor with syntax highlighting, assemble/run/step/reset, and live views of RAM, the CPU registers, and a
stdin/stdout console. It assembles and executes programs exactly like `marvin.py`, including the error
messages.

## Software Dependencies

* [Python >=3.12](https://www.python.org/)
