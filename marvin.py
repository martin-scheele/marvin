#!/usr/bin/env python3
import argparse
import builtins
import datetime
import os
import random
import struct
import sys

description = """
This program serves as an emulator for a register-based machine called Marvin (named after
the paranoid android character, Marvin, from The Hitchhiker's Guide to the Galaxy by 
Douglas Adams). The design of the machine was inspired by that of the Harvey Mudd 
Miniature Machine (HMMM) developed at Harvey Mudd College. The program accepts a .marv file 
as input, assembles and simulates the instructions within, and prints any output to stdout. 
Any input to the .marv program is via stdin. If the optional -v argument is specified, 
the emulator prints the assembled instructions to stdout before simulating them.
"""

# Maps opcodes to their binary 8-bit codes.
opcode_to_bin = {
    # system instructions
    "halt":   0b00000000, "readi":  0b00000001, "readf":  0b00000010, "readc":  0b00000011,
    "writei": 0b00000100, "writef": 0b00000101, "writec": 0b00000110, "seed":   0b00000111,
    "rand":   0b00001000, "time":   0b00001001, "date":   0b00001010, "nop":    0b00001111,
    # arithmetic instructions
    "add":    0b00010000, "sub":    0b00010001, "mul":    0b00010010, "div":    0b00010011,
    "mod":    0b00010100, "neg":    0b00010101, "fadd":   0b00010110, "fsub":   0b00010111,
    "fmul":   0b00011000, "fdiv":   0b00011001, "fneg":   0b00011010,
    # bitwise instructions
    "and":    0b00100000, "or":     0b00100001, "xor":    0b00100010, "not":    0b00100011,
    "lshl":   0b00100100, "lshr":   0b00100101, "ashl":   0b00100110, "ashr":   0b00100111,
    # jump instructions
    "jumpn":  0b00110000, "jumpr":  0b00110001, "jeqzn":  0b00110010, "jnezn":  0b00110011,
    "jgen":   0b00110100, "jlen":   0b00110101, "jeqn":   0b00110110, "jnen":   0b00110111,
    "jgtn":   0b00111000, "jltn":   0b00111001, "calln":  0b00111010,
    # register instructions
    "setn":   0b01000000, "addn":   0b01000001, "setf":   0b01000010, "addf":   0b01000011,
    "copy":   0b01000100,
    # memory instructions
    "pushr":  0b01010000, "popr":   0b01010001, "loadn":  0b01010010, "storen": 0b01010011,
    "loadr":  0b01010100, "storer": 0b01010101,
}

# Maps 8-bit binary codes to the opcodes they represent.
bin_to_opcode = {opcode_to_bin[opcode]: opcode for opcode in opcode_to_bin.keys()}

# Maps opcodes to their symbolic argument masks.
opcode_to_argmask = {
    # system instructions
    "halt":   "",    "readi":  "r",   "readf":  "r",   "readc":  "r",
    "writei": "r",   "writef": "r",   "writec": "r",   "seed":   "r",
    "rand":   "rrr", "time":   "r",   "date":   "r",   "nop":    "",
    # arithmetic instructions
    "add":    "rrr", "sub":    "rrr", "mul":    "rrr", "div":    "rrr",
    "mod":    "rrr", "neg":    "rr",  "fadd":   "rrr", "fsub":   "rrr",
    "fmul":   "rrr", "fdiv":   "rrr", "fneg":   "rr",
    # bitwise instructions
    "and":    "rrr", "or":     "rrr", "xor":    "rrr", "not":    "rrr",
    "lshl":   "rr",  "lshr":   "rr",  "ashl":   "rr",  "ashr":   "rr",
    # jump instructions
    "jumpn":  "n",   "jumpr":  "r",   "jeqzn":  "rn",  "jnezn":  "rn",
    "jgen":   "rrn", "jlen":   "rrn", "jeqn":   "rrn", "jnen":   "rrn",
    "jgtn":   "rrn", "jltn":   "rrn", "calln":  "rn",
    # register instructions
    "setn":   "rn",  "addn":   "rn",  "setf":   "rf",  "addf":   "rf",
    "copy":   "rr",
    # memory instructions
    "pushr":  "rr",  "popr":   "rr",  "loadn":  "rrn", "storen": "rrn",
    "loadr":  "rr",  "storer": "rr",
}

# Maps opcodes to their costs.
opcode_to_cost = {opcode: 2 if opcode_to_bin[opcode] >> 4 == 0b0101 else 1 for opcode in opcode_to_bin.keys()}

# Maps opcodes to their call counts.
opcode_to_calls: dict[str, int] = {opcode: 0 for opcode in opcode_to_bin.keys()}

# Maps register names to their binary 4-bit codes.
reg_to_bin = {
    "r0":  0b0000, "r1":  0b0001, "r2":  0b0010, "r3":  0b0011,
    "a0":  0b0000, "a1":  0b0001, "a2":  0b0010,

    "r4":  0b0100, "r5":  0b0101, "r6":  0b0110, "r7":  0b0111,

    "r8":  0b1000, "r9":  0b1001, "r10": 0b1010, "r11": 0b1011,
                                                 "ra":  0b1011,

    "r12": 0b1100, "r13": 0b1101, "r14": 0b1110, "r15": 0b1111,
    "rv":  0b1100, "fp":  0b1101, "sp":  0b1110, "gp":  0b1111
}

# Maps registers to the current type of their contents.
reg_curr_type: dict[int, builtins.type] = {reg_to_bin[reg]: builtins.int for reg in reg_to_bin.keys()}

# Global argument variables.
verbose = False
verbose_output: list[str] = []
debug = False
count_calls = False

def main():
    # Process command-line inputs and exit if they are not as expected.
    parser = argparse.ArgumentParser(description=description)
    _ = parser.add_argument("filename", help="input .marv file")
    _ = parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    _ = parser.add_argument("-d", "--debug", action="store_true", help="enable debug mode")
    _ = parser.add_argument("-c", "--count", action="store_true", help="count instruction calls")
    args = parser.parse_args()

    global debug, verbose, count_calls
    inFile = args.filename
    verbose = args.verbose
    debug = args.debug
    count_calls = args.count

    if not inFile.endswith(".marv") or not os.path.exists(inFile):
        sys.exit(f"Error: invalid file '{inFile}'")

    # Validate .marv and extract tokens.
    tuples = validate_marv(inFile)

    # Assemble the instructions into machine codes.
    machineCodes = assemble(tuples)

    # Print verbose output
    if verbose:
        print_verbose_output()

    # Simulate the machine codes.
    if len(machineCodes) > 0:
        simulate(machineCodes)

def validate_marv(inFile: str) -> list[tuple[int, int, str, *tuple[str, ...]]]:
    with open(inFile, "r") as fh:
        lines = fh.readlines()

    expectedID = 0
    tuples: list[tuple[int, int, str, *tuple[str, ...]]] = []

    for i, line in enumerate(lines):
        lineno = i + 1
        line = line.strip().lower()
        # Ignore if line is empty or is a comment.
        if line == "" or line.startswith("#"):
            continue

        # Remove inlined comment if any.
        if "#" in line:
            line = line[:line.find("#")].strip()

        # Exit with error if number of tokens in line is less than 2, or if the instruction ID is
        # invalid, or if the instruction is invalid.
        toks = line.split()
        if len(toks) < 2:
            sys.exit(f"Error {inFile}@{lineno}: not enough tokens")
        if not is_int(toks[0]) or int(toks[0]) != expectedID:
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction ID '{toks[0]}'")
        if toks[1] not in opcode_to_bin:
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction '{toks[1]}'")
        expectedID += 1

        # Validate the instruction arguments.
        id, opcode, args = int(toks[0]), toks[1], toks[2:]
        if len(args) != len(opcode_to_argmask[opcode]):
            # TODO: more verbose error: e.g. def argmask_to_str for argument types
            sys.exit(f"Error {inFile}@{lineno}: opcode expects {len(opcode_to_argmask[opcode])} arguments")
        for i, c in enumerate(opcode_to_argmask[opcode]):
            if c == "r":
                if not valid_reg(args[i]):
                    sys.exit(f"Error {inFile}@{lineno}: invalid register '{args[i]}'")
            elif c == "n":
                if not is_int(args[i]):
                    sys.exit(f"Error {inFile}@{lineno}: invalid number '{args[i]}'")
            elif c == "f":
                if not is_float(args[i]):
                    sys.exit(f"Error {inFile}@{lineno}: invalid number '{args[i]}'")

        # Append valid tuple.
        tuples.append((lineno, id, opcode, *toks[2:]))

    # Additional validation of instruction arguments.
    # TODO: setf, addf?
    for t in tuples:
        lineno, id, opcode, args = t[0], t[1], t[2], t[3:]
        if opcode in {"jumpn"} and int(args[0]) >= len(tuples):
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction address '{args[0]}'")
        elif opcode in {"addn", "setn"} and not valid_int(int(args[1])):
            sys.exit(f"Error {inFile}@{lineno}: invalid number '{args[1]}'")
        elif opcode in {"loadn", "storen"} and not valid_int(int(args[2])):
            sys.exit(f"Error {inFile}@{lineno}: invalid address '{args[2]}'")
        elif opcode in {"calln", "jeqzn", "jnezn"} and int(args[1]) >= len(tuples):
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction address '{args[1]}'")
        elif opcode in {"jeqn", "jgen", "jgtn", "jlen", "jltn", "jnen"} and int(args[2]) >= len(tuples):
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction address '{args[2]}'")

    return tuples


# Assembles the instructions in tuples and returns a list containing the corresponding machine
# codes. Prints the assembled instructions to stdout if verbose is True.
def assemble(tuples: list[tuple[int, int, str, *tuple[str, ...]]]) -> list[int]:
    machine_code: list[int] = []

    for t in tuples:
        id, opcode, args = t[1], t[2], t[3:]

        # TODO: do something that is not this...
        byte_list: list[int] = [opcode_to_bin[opcode], 0, 0, 0]
        curr_byte = 3
        nibble = 0
        i = len(args) - 1
        for c in reversed(opcode_to_argmask[opcode]):
            if c == "r":
                byte_list[curr_byte] |= reg_to_bin[args[i]] << nibble * 4
                nibble = (nibble + 1) % 2
                if nibble == 0:
                    curr_byte -= 1
            elif c == "n":
                val = tc_int_to_b16(int(args[i]))
                byte_list[curr_byte] = val & 0xff
                byte_list[curr_byte - 1] = val >> 8
                curr_byte -= 2
            elif c == "f":
                val = fp_float_to_f16(float(args[i]))
                byte_list[curr_byte] = val & 0xff
                byte_list[curr_byte - 1] = val >> 8
                curr_byte -= 2
            i -= 1

        code = byte_list[0] << 24 | byte_list[1] << 16 | byte_list[2] << 8 | byte_list[3]
        machine_code.append(code)

        if verbose or debug:
            binCode = f"{id: >5}: {" ".join([format(byte, "08b") for byte in byte_list])}"
            asmCode = f"{id: >5}: {opcode: <6} {" ".join(args)}"
            verbose_output.append(f"{binCode: <50} {asmCode}")

    return machine_code


# Global CPU variables.
reg = [0] * 16      # registers
mem = [0] * 65536   # main memory
pc = 0              # program counter
ir = 0              # instruction register

# Simulate the assembled instructions in machineCodes.
def simulate(machine_code: list[int]):
    global reg, mem, ir

    # Initialize the frame, stack, and global pointers
    reg[reg_to_bin["fp"]] = reg[reg_to_bin["sp"]] = 65535
    reg[reg_to_bin["gp"]] = 8192

    # Load the machine code into memory starting at location 0.
    for i, v in enumerate(machine_code):
        mem[i] = v

    while True:
        # Fetch the next instruction to simulate.
        try:
            ir = machine_code[pc]
        except IndexError:
            sys.exit(f"Error: attempted to execute mem['{pc}']; halting the machine")

        # Conditionally execute the debugger.
        if debug: 
            debug_exec()

        # Extract the opcode from the instruction register.
        opcode = bin_to_opcode[ir >> 24]

        # Extract arguments from the instruction register.
        args = extract_args(ir, opcode_to_argmask[opcode])

        # Track register type
        # track_register_type(opcode, args)

        # Simulate the instruction for the opcode.
        instructions[opcode](args)

        # Count instruction calls
        if count_calls:
            opcode_to_calls[opcode] += 1
            # TODO: actually delay and visualize

def debug_exec():
    global reg, mem, pc, ir
    opcode = bin_to_opcode[ir >> 24]
    print(f"pc: {pc} opcode: {opcode}")
    print_regs()
    for i in range(65535, tc_b32_to_int(reg[reg_to_bin["sp"]]) - 1, -1):
        if i == reg[reg_to_bin["fp"]]:
            print("*", end="")
        if i == reg[reg_to_bin["sp"]]:
            print("^", end="")
        print(mem[i])
    _ = input()

def print_regs():
    # TODO: figure out floating point spacing
    print(f"r0:  {format_reg(reg[0]):  10d} r1:  {format_reg(reg[1]):  10d} r2:  {format_reg(reg[2]):  10d} r3:  {format_reg(reg[3]):  10d}")
    print(f"r4:  {format_reg(reg[4]):  10d} r5:  {format_reg(reg[5]):  10d} r6:  {format_reg(reg[6]):  10d} r7:  {format_reg(reg[7]):  10d}")
    print(f"r8:  {format_reg(reg[8]):  10d} r9:  {format_reg(reg[9]):  10d} r10: {format_reg(reg[10]): 10d} r11: {format_reg(reg[11]): 10d}")
    print(f"r12: {format_reg(reg[12]): 10d} r13: {format_reg(reg[13]): 10d} r14: {format_reg(reg[14]): 10d} r15: {format_reg(reg[15]): 10d}")

def format_reg(regbin: int) -> str:
    match reg_curr_type[regbin]:
        case builtins.int:
            return f"{tc_b32_to_int(reg[regbin])}"
        case builtins.float:
            return f"{fp_f32_to_float(reg[regbin])}"
        case builtins.str:
            return f"{chr(reg[regbin])}"
        case _:
            return ""

# System Instructions

def op_halt(_):
    if count_calls:
        print_opcode_cost()
    sys.exit()

def op_readi(args: list[int]):
    global reg, pc
    while True:
        try:
            x = int(input())
            if (valid_int(x)):
                break
            raise ValueError
        except ValueError:
            print("Illegal input: input must be a number must be in [-32768, 32767]")
    reg[args[0]] = tc_int_to_b32(x)
    pc += 1


def op_readf(args: list[int]):
    global reg, pc
    while True:
        try:
            x = float(input())
            break
        except ValueError:
            # TODO: better error message
            print("Illegal input: input must be a number")
    reg[args[0]] = fp_float_to_f32(x)
    pc += 1

def op_readc(args: list[int]):
    global reg, pc
    while True:
        try:
            # TODO: OS compliant getch()
            x = sys.stdin.read(1)
            _ = sys.stdin.readline()
            break
        except Exception:
            pass
    reg[args[0]] = ord(x)
    pc += 1

def op_writei(args: list[int]):
    global pc
    print(tc_b32_to_int(reg[args[0]]))
    pc += 1

def op_writef(args: list[int]):
    global pc
    print(fp_f32_to_float(reg[args[0]]))
    pc += 1

def op_writec(args: list[int]):
    global pc
    # TODO: how to handle value outside of range? clamp?
    print(chr(reg[args[0]]))
    pc += 1

def op_seed(args: list[int]):
    global pc
    random.seed((reg[args[0]]))
    pc += 1

def op_rand(args: list[int]):
    global reg, pc
    lo = tc_b32_to_int(reg[args[0]])
    hi = tc_b32_to_int(reg[args[1]])
    reg[args[2]] = tc_int_to_b32(random.randint(lo, hi))
    pc += 1

def op_time(args: list[int]):
    global reg, pc
    now = datetime.datetime.now()
    reg[args[0]] = (now.hour * 3600 + now.minute * 60 + now.second) * 1000 + now.microsecond // 1000
    pc += 1


def op_date(args: list[int]):
    global reg, pc
    today = datetime.date.today()
    reg[args[0]] = today.year << 13 | today.month << 9 | today.day
    pc += 1

def op_nop(_):
    global pc
    pc += 1

# Arithmetic instructions

def op_neg(args: list[int]):
    global reg, pc
    reg[args[0]] = tc_neg(reg[args[1]])
    pc += 1

def op_add(args: list[int]):
    global reg, pc
    reg[args[0]] = tc_add(reg[args[1]], reg[args[2]])
    pc += 1

def op_sub(args: list[int]):
    global reg, pc
    reg[args[0]] = tc_sub(reg[args[1]], reg[args[2]])
    pc += 1

def op_mul(args: list[int]):
    global reg, pc
    reg[args[0]] = tc_mul(reg[args[1]], reg[args[2]])
    pc += 1

def op_div(args: list[int]):
    global reg, pc
    reg[args[0]] = tc_div(reg[args[1]], reg[args[2]])
    pc += 1

def op_mod(args: list[int]):
    global reg, pc
    reg[args[0]] = tc_mod(reg[args[1]],  reg[args[2]])
    pc += 1

def op_fneg(args: list[int]):
    global reg, pc
    reg[args[0]] = reg[args[1]] ^ (1 << 31)
    pc += 1

def op_fadd(args: list[int]):
    global reg, pc
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[1]]) + fp_f32_to_float(reg[args[2]]))
    pc += 1

def op_fsub(args: list[int]):
    global reg, pc
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[1]]) - fp_f32_to_float(reg[args[2]]))
    pc += 1

def op_fmul(args: list[int]):
    global reg, pc
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[1]]) * fp_f32_to_float(reg[args[2]]))
    pc += 1

def op_fdiv(args: list[int]):
    global reg, pc
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[1]]) / fp_f32_to_float(reg[args[2]]))
    pc += 1

# Bitwise instructions

def op_and(args: list[int]):
    global reg, pc
    reg[args[0]] = reg[args[1]] & reg[args[2]]
    pc += 1

def op_or(args: list[int]):
    global reg, pc
    reg[args[0]] = reg[args[1]] | reg[args[2]]
    pc += 1

def op_xor(args: list[int]):
    global reg, pc
    reg[args[0]] = reg[args[1]] ^ reg[args[2]]
    pc += 1

def op_not(args: list[int]):
    global reg, pc
    reg[args[0]] = ~reg[args[1]]
    pc += 1

def op_lshl(args: list[int]):
    global reg, pc
    reg[args[0]] = (reg[args[1]] << reg[args[2]]) & 0xffffffff
    pc += 1

def op_lshr(args: list[int]):
    global reg, pc
    reg[args[0]] = (reg[args[1]] >> reg[args[2]]) & 0xffffffff
    pc += 1

def op_ashl(args: list[int]):
    global reg, pc
    sign = reg[args[1]] & (1 << 31)
    temp = ((reg[args[1]] ^ sign) << reg[args[2]]) & 0xffffffff
    reg[args[0]] = 0 if temp == 0 else temp | sign
    pc += 1

def op_ashr(args: list[int]):
    global reg, pc
    sign = reg[args[1]] & (1 << 31)
    temp = ((reg[args[1]] ^ sign) >> reg[args[2]]) & 0xffffffff
    reg[args[0]] = 0 if temp == 0 else temp | sign
    pc += 1

# Jump Instructions

def op_jumpn(args: list[int]):
    global pc
    pc = args[0]

def op_jumpr(args: list[int]):
    global pc
    pc = reg[args[0]]

def op_jeqzn(args: list[int]):
    global pc
    pc = args[1] if reg[args[0]] == 0 else pc + 1

def op_jnezn(args: list[int]):
    global pc
    pc = args[1] if reg[args[0]] != 0 else pc + 1

def op_jgen(args: list[int]):
    global pc
    pc = args[2] if reg[args[0]] >= reg[args[1]] else pc + 1

def op_jlen(args: list[int]):
    global pc
    pc = args[2] if reg[args[0]] <= reg[args[1]] else pc + 1

def op_jeqn(args: list[int]):
    global pc
    pc = args[2] if reg[args[0]] == reg[args[1]] else pc + 1

def op_jnen(args: list[int]):
    global pc
    pc = args[2] if reg[args[0]] != reg[args[1]] else pc + 1

def op_jgtn(args: list[int]):
    global pc
    pc = args[2] if reg[args[0]] > reg[args[1]] else pc + 1

def op_jltn(args: list[int]):
    global pc
    pc = args[2] if reg[args[0]] < reg[args[1]] else pc + 1

def op_calln(args: list[int]):
    global reg, pc
    # I think this is guaranteed to never be over integer limit
    reg[args[0]] = pc + 1
    pc = args[1]

# Register instructions

def op_setn(args: list[int]):
    global reg, pc
    reg[args[0]] = tc_b16_to_b32(args[1])
    pc += 1

def op_addn(args: list[int]):
    global reg, pc
    reg[args[0]] = tc_add(reg[args[0]], tc_b16_to_b32(args[1]))
    pc += 1

def op_setf(args: list[int]):
    global reg, pc
    reg[args[0]] = fp_f16_to_f32(args[1])
    pc += 1

def op_addf(args: list[int]):
    global reg, pc
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[0]]) + fp_f32_to_float(reg[args[1]]))
    pc += 1

def op_copy(args: list[int]):
    global reg, pc
    reg[args[0]] = reg[args[1]]
    pc += 1

# Memory instructions

def op_loadr(args: list[int]):
    global reg, pc
    reg[args[0]] = mem[reg[args[1]]]
    pc += 1

def op_pushr(args: list[int]):
    global reg, mem, pc
    # check if sp == gp
    if reg[reg_to_bin["sp"]] == reg[reg_to_bin["gp"]]:
        sys.exit(f"Error: stack overflow attempting to execute mem['{pc}']; halting the machine")
    mem[reg[args[1]]] = reg[args[0]]
    reg[args[1]] = tc_sub(reg[args[1]], 1)
    pc += 1

def op_popr(args: list[int]):
    global reg, pc
    reg[args[1]] = tc_add(reg[args[1]], 1)
    reg[args[0]] = mem[reg[args[1]]]
    pc += 1

def op_loadn(args: list[int]):
    global reg, pc
    reg[args[0]] = mem[tc_add(reg[args[1]], tc_b16_to_b32(args[2]))]
    pc += 1

def op_storen(args: list[int]):
    global mem, pc
    mem[tc_add(reg[args[1]], tc_b16_to_b32(args[2]))] = reg[args[0]]
    pc += 1

def op_storer(args: list[int]):
    global mem, pc
    mem[reg[args[1]]] = reg[args[0]]
    pc += 1

def op_unimp(_):
    sys.exit(f"Error: operation {bin_to_opcode[ir >> 24]} unimplemented")

# TODO: where to put this? awkward
instructions = {}
for op in opcode_to_bin.keys():
    func = "op_" + op
    instructions[op] = globals()[func] if func in globals() else op_unimp

# Integer constants

INT16_MIN  = -2 ** 15
INT16_MAX  =  2 ** 15 - 1
UINT16_MAX =  2 ** 16
INT32_MIN  = -2 ** 31
INT32_MAX  =  2 ** 31 - 1
UINT32_MAX =  2 ** 32

# Two's complement helper functions

# TODO: combine into one function tc_int_to_bin(val: int, width: int) -> int
# TODO: report under/overflow
# TODO: completely redo tc over/underflow - look at existing implementations
def tc_int_to_b16(val: int) -> int:
    while (val > INT16_MAX):
        val -= UINT16_MAX
    while (val < INT16_MIN):
        val += UINT16_MAX
    if val < 0:
        val = -val
        val = (val ^ 0xffff) + 1
    return val

# TODO: report under/overflow
def tc_int_to_b32(val: int) -> int:
    while (val > INT32_MAX):
        val -= UINT32_MAX
    while (val < INT32_MIN):
        val += UINT32_MAX
    if val < 0:
        val = -val
        val = (val ^ 0xffffffff) + 1
    return val

def tc_b16_to_b32(val: int) -> int:
    if val & 1 << 15:
        val = val | (0xffff << 16)
    return val

def tc_b32_to_int(val: int) -> int:
    if val & 1 << 31:
        val = (val - 1) ^ 0xffffffff
        val = -val
    return val

def tc_neg(val: int) -> int:
    if val > 0:
        val = (val ^ 0xffffffff) + 1
    elif val < 0:
        val = (val - 1) ^ 0xffffffff
    else:
        val = 0
    return val

# TODO: report under/overflow
def tc_add(val1: int, val2: int) -> int:
    return (val1 + val2) & 0xffffffff

# TODO: report under/overflow
def tc_sub(val1: int, val2: int) -> int:
    return (val1 - val2) & 0xffffffff

# TODO: this isn't wrapping - verify under/overflow
def tc_mul(val1: int, val2: int) -> int:
    if (val1 & (1 << 31)):
        val1 = val1 | (0xffffffff << 32)
    if (val2 & (1 << 31)):
        val2 = val2 | (0xffffffff << 32) 
    return (val1 * val2) & 0xffffffff

# TODO: verify accuracy
def tc_div(val1: int, val2: int) -> int:
    sign = (val1 & (1 << 31)) ^ (val2 & (1 << 31))
    return sign | ((val1 & 0x7fffffff) // (val2 & 0x7fffffff))

# TODO: didn't actually test this at all. most definitely wrong
def tc_mod(val1: int, val2: int) -> int:
    sign = (val1 & (1 << 31)) ^ (val2 & (1 << 31))
    return sign | ((val1 & 0x7fffffff) % (val2 & 0x7fffffff))

# Floating point helper functions

# TODO: optimize
def fp_float_to_f16(val: float) -> int:
    if val == 0:
        return 0
    # find sign
    if val < 0:
        sign = 1
        val = -val
    else:
        sign = 0

    # separate integer and fractional parts
    int_bin = int(val)
    frac_part = val - int_bin

    # determine bit lengths for exponent + normalization
    len_int_bin = int_bin.bit_length()
    len_frac_part = 11 - len_int_bin

    # convert fractional part to binary representation
    # TODO: how to round?
    frac_bin = 0
    for i in range(len_frac_part - 1, -1, -1):
        frac_part *= 2
        if frac_part >= 1:
            frac_part -= 1
            frac_bin |= 1 << i

    # calculate exponent + adjust bias
    exp = len_int_bin - 1 + 15

    # normalize + truncate significand
    significand = ((int_bin << len_frac_part) | frac_bin) & 0xffff

    return (sign << 15) | (exp << 10) | significand

def fp_float_to_f32(val: float) -> int:
    # TODO: test speed between struct and manual conversion
    bstr = "".join(format(byte, "08b") for byte in struct.pack("!f", val))
    return int(bstr, 2)

def fp_f16_to_f32(val: int) -> int:
    sign = val >> 15
    exp = ((val & 0x1f << 10) >> 10) - 15 + 127
    significand = (val & 0x3ff) << (23 - 10)
    f32 = sign << 31 | exp << 23 | significand
    return f32

def fp_f32_to_float(val: int) -> float:
    # TODO: test speed between struct and manual conversion
    return struct.unpack("!f", val.to_bytes(4))[0]

def extract_args(ir: int, mask: str) -> list[int]:
    ret: list[int] = []
    for c in reversed(mask):
        if c == "r":
            ret.insert(0, ir & 0xf)
            ir >>= 4
        elif c == "n" or c == "f":
            ret.insert(0, ir & 0xffff)
            ir >>= 16
    return ret

# Misc. helper functions

# Returns True if s encodes an integer, and False otherwise.
def is_int(s: str) -> bool:
    try:
        _ = int(s)
        # TODO: check to disclude floats
        return True
    except ValueError:
        return False

def is_float(s: str) -> bool:
    try:
        _ = float(s)
        return True
    except ValueError:
        return False

# Return True if n is a valid signed 16-bit integer, and False otherwise.
def valid_int(n: int) -> bool:
    return -2 ** 15 <= n <= 2 ** 15 - 1

# Return True if s is "r" followed by a number from the interval [0, 15], and False otherwise.
def valid_reg(s: str) -> bool:
    return s in reg_to_bin.keys()

def print_verbose_output():
    for s in verbose_output:
        print(s)
    print()

def print_opcode_cost():
    for k, v in opcode_to_calls.items():
        if v > 0:
            print(f"{k: <6}: {v * opcode_to_cost[k]}")

if __name__ == "__main__":
    main()
