#!/usr/bin/env python3
import argparse
import os
import struct
import sys
import random
import time

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
    "halt":  0b00000000, "read":   0b00000001, "write": 0b00000010, "nop":    0b00000011,
    "set0":  0b00000100, "set1":   0b00000101, "setn":  0b00000110, "addn":   0b00000111,
    "copy":  0b00001000, "neg":    0b00001001, "add":   0b00001010, "sub":    0b00001011,
    "mul":   0b00001100, "div":    0b00001101, "mod":   0b00001110, "jumpn":  0b00001111,
    "jumpr": 0b00010000, "jeqzn":  0b00010001, "jnezn": 0b00010010, "jgen":   0b00010011,
    "jlen":  0b00010110, "jeqn":   0b00010100, "jnen":  0b00010101, "jgtn":   0b00010111,
    "jltn":  0b00011000, "calln":  0b00011001, "pushr": 0b00011010, "popr":   0b00011011,
    "loadn": 0b00011100, "storen": 0b00011101, "loadr": 0b00011110, "storer": 0b00011111
}

# Maps 8-bit binary codes to the opcodes they represent.
bin_to_opcode = {opcode_to_bin[opcode]: opcode for opcode in opcode_to_bin.keys()}

# Maps register names to their binary 4-bit codes.
reg_to_bin = {
    "r0":  0b0000, "r1":  0b0001, "r2":  0b0010, "r3":  0b0011, 
    "r4":  0b0100, "r5":  0b0101, "r6":  0b0110, "r7":  0b0111, 
    "r8":  0b1000, "r9":  0b1001, "r10": 0b1010, "r11": 0b1011,
    "r12": 0b1100, "r13": 0b1101, "r14": 0b1110, "r15": 0b1111,
    "ra":  0b1100, "rv":  0b1101, "fp":  0b1110, "sp":  0b1111
}

def main():
    # Process command-line inputs and exit if they are not as expected.
    parser = argparse.ArgumentParser(description=description)
    _ = parser.add_argument("filename", help="input .marv file")
    _ = parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    _ = parser.add_argument("-d", "--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()

    inFile = args.filename
    verbose = args.verbose
    debug = args.debug

    if not inFile.endswith(".marv") or not os.path.exists(inFile):
        sys.exit(f"Error: invalid file '{inFile}'")

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
        if not isNum(toks[0]) or int(toks[0]) != expectedID:
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction ID '{toks[0]}'")
        if toks[1] not in opcode_to_bin:
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction '{toks[1]}'")
        expectedID += 1

        # Validate the instruction arguments.
        id, opcode, args = int(toks[0]), toks[1], toks[2:]
        if opcode in {"halt", "nop"}:
            if len(args) != 0:
                sys.exit(f"Error {inFile}@{lineno}: '{opcode}' expects 0 arguments")
            tuples.append((lineno, id, opcode))
        elif opcode in {"jumpr", "read", "set0", "set1", "write"}:
            if len(args) != 1:
                sys.exit(f"Error {inFile}@{lineno}: '{opcode}' expects 1 argument, rX")
            if not validReg(toks[2]):
                sys.exit(f"Error {inFile}@{lineno}: invalid register '{toks[2]}'")
            tuples.append((lineno, id, opcode, toks[2]))
        elif opcode in {"jumpn"}:
            if len(args) != 1:
                sys.exit(f"Error {inFile}@{lineno}: '{opcode}' expects 1 argument, N")
            if not isNum(toks[2]):
                sys.exit(f"Error {inFile}@{lineno}: invalid number '{toks[2]}'")
            tuples.append((lineno, id, opcode, toks[2]))
        elif opcode in {"copy", "loadr", "neg", "popr", "pushr", "storer"}:
            if len(args) != 2:
                sys.exit(f"Error {inFile}@{lineno}: '{toks[1]}' expects 2 arguments, rX rY")
            if not validReg(toks[2]):
                sys.exit(f"Error {inFile}@{lineno}: invalid register '{toks[2]}'")
            if not validReg(toks[3]):
                sys.exit(f"Error {inFile}@{lineno}: invalid register '{toks[3]}'")
            tuples.append((lineno, id, opcode, toks[2], toks[3]))
        elif opcode in {"addn", "calln", "jeqzn", "jnezn", "setn"}:
            if len(args) != 2:
                sys.exit(f"Error {inFile}@{lineno}: '{toks[1]}' expects 2 arguments, rX N")
            if not validReg(toks[2]):
                sys.exit(f"Error {inFile}@{lineno}: invalid register '{toks[2]}'")
            if not isNum(toks[3]):
                sys.exit(f"Error {inFile}@{lineno}: invalid number '{toks[3]}'")
            tuples.append((lineno, id, opcode, toks[2], toks[3]))
        elif opcode in {"add", "div", "mod", "mul", "sub"}:
            if len(args) != 3:
                sys.exit(f"Error {inFile}@{lineno}: '{toks[1]}' expects 3 arguments, rX rY rZ")
            if not validReg(toks[2]):
                sys.exit(f"Error {inFile}@{lineno}: invalid register '{toks[2]}'")
            if not validReg(toks[3]):
                sys.exit(f"Error {inFile}@{lineno}: invalid register '{toks[3]}'")
            if not validReg(toks[4]):
                sys.exit(f"Error {inFile}@{lineno}: invalid register '{toks[4]}'")
            tuples.append((lineno, id, opcode, toks[2], toks[3], toks[4]))
        elif opcode in {"jeqn", "jgen", "jgtn", "jlen", "jltn", "jnen", "loadn", "storen"}:
            if len(args) != 3:
                sys.exit(f"Error {inFile}@{lineno}: '{toks[1]}' expects 3 arguments, rX rY N")
            if not validReg(toks[2]):
                sys.exit(f"Error {inFile}@{lineno}: invalid register '{toks[2]}'")
            if not validReg(toks[3]):
                sys.exit(f"Error {inFile}@{lineno}: invalid register '{toks[3]}'")
            if not isNum(toks[4]):
                sys.exit(f"Error {inFile}@{lineno}: invalid number '{toks[4]}'")
            tuples.append((lineno, id, opcode, toks[2], toks[3], toks[4]))

    # Additional validation of instruction arguments.
    for t in tuples:
        lineno, id, opcode, args = t[0], t[1], t[2], t[3:]
        if opcode in {"jumpn"} and int(args[0]) >= len(tuples):
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction address '{args[0]}'")
        elif opcode in {"addn", "setn"} and not validNum(int(args[1])):
            sys.exit(f"Error {inFile}@{lineno}: invalid number '{args[1]}'")
        elif opcode in {"loadn", "storen"} and not validNum(int(args[2])):
            sys.exit(f"Error {inFile}@{lineno}: invalid address '{args[2]}'")
        elif opcode in {"calln", "jeqzn", "jnezn"} and int(args[1]) >= len(tuples):
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction address '{args[1]}'")
        elif opcode in {"jeqn", "jgen", "jgtn", "jlen", "jltn", "jnen"} and int(args[2]) >= len(tuples):
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction address '{args[2]}'")

    # Assemble the instructions into machine codes.
    machineCodes = assemble(tuples, verbose)

    # Simulate the machine codes.
    if len(machineCodes) > 0:
        simulate(machineCodes, debug)

# Assembles the instructions in tuples and returns a list containing the corresponding machine
# codes. Prints the assembled instructions to stdout if verbose is True.
def assemble(tuples: list[tuple[int, int, str, *tuple[str, ...]]], verbose: bool) -> list[int]:
    machineCodes: list[int] = []
    verboseOutput: list[str] = []

    for t in tuples:
        id, opcode, args = t[1], t[2], t[3:]
        bArg1, bArg2, bArg3 = 0, 0, 0

        if opcode in {"halt", "nop"}:
            bArg1, bArg2, bArg3 = 0, 0, 0
        elif opcode in {"jumpr", "read", "set0", "set1", "write"}:
            bArg1, bArg2, bArg3 = 0, 0, reg_to_bin[args[0]]
        elif opcode in {"jumpn"}:
            val = int(args[0])
            bArg1, bArg2, bArg3 = 0, val >> 8, val & 0xff
        elif opcode in {"copy", "loadr", "neg", "popr", "pushr", "storer"}:
            bArg1, bArg2, bArg3 = 0, 0, reg_to_bin[args[0]] << 4 | reg_to_bin[args[1]]
        elif opcode in {"addn", "setn"}:
            bArg1 = reg_to_bin[args[0]]
            val = tc_int_to_b16(int(args[1]))
            bArg2, bArg3 = val >> 8, val & 0xff
        elif opcode in {"calln", "jeqzn", "jnezn"}:
            bArg1 = reg_to_bin[args[0]]
            val = int(args[1])
            bArg2, bArg3 =  val >> 8, val & 0xff
        elif opcode in {"add", "div", "mod", "mul", "sub"}:
            bArg1, bArg2, bArg3 = 0, reg_to_bin[args[0]], reg_to_bin[args[1]] << 4 | reg_to_bin[args[2]]
        elif opcode in {"jeqn", "jgen", "jgtn", "jlen", "jltn", "jnen"}:
            bArg1 = reg_to_bin[args[0]] << 4 | reg_to_bin[args[1]]
            val = int(args[2])
            bArg2, bArg3 = val >> 8, val & 0xff
        elif opcode in {"loadn", "storen"}:
            bArg1 = reg_to_bin[args[0]] << 4 | reg_to_bin[args[1]]
            val = tc_int_to_b16(int(args[2]))
            bArg2, bArg3 = val >> 8, val & 0xff

        op = opcode_to_bin[opcode]
        code = op << 24 | bArg1 << 16 | bArg2 << 8 | bArg3
        machineCodes.append(code)

        if verbose:
            bArg1 = format(bArg1, '08b')
            bArg2 = format(bArg2, '08b')
            bArg3 = format(bArg3, '08b')

            binCode = f"{id: >5}: {format(opcode_to_bin[opcode], '08b')} {bArg1} {bArg2} {bArg3}"
            asmCode = f"{id: >5}: {opcode: <6} {" ".join(args)}"
            verboseOutput.append(f"{binCode: <50} {asmCode}")

    if verbose:
        for s in verboseOutput:
            print(s)
        print()

    return machineCodes

reg: list[int]
mem: list[int]
pc: int
stack_end: int

# Simulate the assembled instructions in machineCodes.
def simulate(machineCodes: list[int], debug: bool):
    global reg, mem, pc, stack_end
    reg = [0] * 16     # registers
    mem = [0] * 65536  # main memory
    pc = 0             # program counter
    ir = 0             # instruction register

    # Initialize the frame and stack pointers to 65535 (the base address of the stack).
    reg[14], reg[15] = 65535, 65535

    # Load the machine codes into memory starting at location 0.
    for i, v in enumerate(machineCodes):
        mem[i] = v

    # Initialize end of stack; 8K of text
    stack_end = 8192

    while True:
        # Fetch the next instruction to simulate.
        try:
            ir = machineCodes[pc]
        except IndexError:
            sys.exit(f"Error: attempted to execute mem['{pc}']; halting the machine")

        # Extract the opcode.
        code = ir
        op = code >> 24
        opcode = bin_to_opcode[op]

        # Debug stub
        if debug:
            debug_exec(code)

        # Simulation the instruction given by opcode.
        instructions[opcode](code)

def debug_exec(code: int):
    global reg, mem, pc
    opcode = bin_to_opcode[code >> 24]
    # TODO: how to represent different types?
    print(f"pc: {pc} opcode: {opcode}")
    print(f"r0:  {tc_b32_to_int(reg[0]): 10d} r1:  {tc_b32_to_int(reg[1]): 10d} r2:  {tc_b32_to_int(reg[2]): 10d} r3:  {tc_b32_to_int(reg[3]): 10d}")
    print(f"r4:  {tc_b32_to_int(reg[4]): 10d} r5:  {tc_b32_to_int(reg[5]): 10d} r6:  {tc_b32_to_int(reg[6]): 10d} r7:  {tc_b32_to_int(reg[7]): 10d}")
    print(f"r8:  {tc_b32_to_int(reg[8]): 10d} r9:  {tc_b32_to_int(reg[9]): 10d} r10: {tc_b32_to_int(reg[10]): 10d} r11: {tc_b32_to_int(reg[11]): 10d}")
    print(f"r12: {tc_b32_to_int(reg[12]): 10d} r13: {tc_b32_to_int(reg[13]): 10d} r14: {tc_b32_to_int(reg[14]): 10d} r15: {tc_b32_to_int(reg[15]): 10d}")
    for i in range(65535, tc_b32_to_int(reg[15]) - 1, -1):
        if i == reg[14]:
            print("*", end="")
        if i == reg[15]:
            print("^", end="")
        print(mem[i])
    _ = input()

def op_halt(code: int):
    sys.exit()

def op_read(code: int):
    global reg, pc
    arg1 = code & 0xf
    while True:
        try:
            x = int(input())
            if (validNum(x)):
                break
            raise ValueError
        except ValueError:
            print("Illegal input: number must be in [-32768, 32767]")
    reg[arg1] = tc_int_to_b32(x)
    pc += 1

def op_write(code: int):
    global reg, pc
    arg1 = code & 0xf
    print(tc_b32_to_int(reg[arg1]))
    pc += 1

def op_nop(code: int):
    global pc
    pc += 1

def op_set0(code: int):
    global reg, pc
    arg1 = code & 0xf
    reg[arg1] = 0
    pc += 1

def op_set1(code: int):
    global reg, pc
    arg1 = code & 0xf
    reg[arg1] = 1
    pc += 1

def op_jumpr(code: int):
    global reg, pc
    arg1 = code & 0xf
    pc = reg[arg1]

def op_jumpn(code: int):
    global pc
    arg1 = code & 0xffff
    pc = arg1

def op_copy(code: int):
    global reg, pc
    arg1 = (code & 0xf << 4) >> 4
    arg2 = code & 0xf
    reg[arg1] = reg[arg2]
    pc += 1

def op_loadr(code: int):
    global reg, mem, pc
    arg1 = (code & 0xf << 4) >> 4
    arg2 = code & 0xf
    reg[arg1] = mem[reg[arg2]]
    pc += 1

def op_neg(code: int):
    global reg, pc
    arg1 = (code & 0xf << 4) >> 4
    arg2 = code & 0xf
    reg[arg1] = tc_neg(reg[arg2])
    pc += 1

def op_popr(code: int):
    global reg, mem, pc
    arg1 = (code & 0xf << 4) >> 4
    arg2 = code & 0xf
    reg[arg2] = tc_add(reg[arg2], 1)
    reg[arg1] = mem[reg[arg2]]
    pc += 1

def op_pushr(code: int):
    global reg, mem, pc, stack_end
    if reg[15] == stack_end:
        sys.exit(f"Error: stack overflow attempting to execute mem['{pc}']; halting the machine")
    arg1 = (code & 0xf << 4) >> 4
    arg2 = code & 0xf
    mem[reg[arg2]] = reg[arg1]
    reg[arg2] = tc_sub(reg[arg2], 1)
    pc += 1

def op_storer(code: int):
    global reg, mem, pc
    arg1 = (code & 0xf << 4) >> 4
    arg2 = code & 0xf
    mem[reg[arg2]] = reg[arg1]
    pc += 1

def op_addn(code: int):
    global reg, pc
    arg1 = (code & 0xf << 16) >> 16
    arg2 = code & 0xffff
    reg[arg1] = tc_add(reg[arg1], tc_b16_to_b32(arg2))
    pc += 1

def op_calln(code: int):
    global reg, pc
    arg1 = (code & 0xf << 16) >> 16
    arg2 = code & 0xffff
    # I think this is guaranteed to never be over integer limit
    reg[arg1] = pc + 1
    pc = arg2

def op_jeqzn(code: int):
    global reg, pc
    arg1 = (code & 0xf << 16) >> 16
    arg2 = code & 0xffff
    pc = arg2 if reg[arg1] == 0 else pc + 1

def op_jnezn(code: int):
    global reg, pc
    arg1 = (code & 0xf << 16) >> 16
    arg2 = code & 0xffff
    pc = arg2 if reg[arg1] != 0 else pc + 1

def op_loadn(code: int):
    global reg, mem, pc
    arg1 = (code & 0xf << 20) >> 20
    arg2 = (code & 0xf << 16) >> 16
    arg3 = code & 0xffff
    reg[arg1] = mem[tc_add(reg[arg2], tc_b16_to_b32(arg3))]
    pc += 1

def op_setn(code: int):
    global reg, pc
    arg1 = (code & 0xf << 16) >> 16 
    arg2 = code & 0xffff
    reg[arg1] = tc_b16_to_b32(arg2)
    pc += 1

def op_storen(code: int):
    global reg, mem, pc
    arg1 = (code & 0xf << 20) >> 20
    arg2 = (code & 0xf << 16) >> 16
    arg3 = code & 0xffff
    mem[tc_add(reg[arg2], tc_b16_to_b32(arg3))] = reg[arg1]
    pc += 1

def op_add(code: int):
    global reg, pc
    arg1 = (code & 0xf << 8) >> 8
    arg2 = (code & 0xf << 4) >> 4 
    arg3 = code & 0xf
    reg[arg1] = tc_add(reg[arg2], reg[arg3])
    pc += 1

def op_div(code: int):
    global reg, pc
    arg1 = (code & 0xf << 8) >> 8
    arg2 = (code & 0xf << 4) >> 4
    arg3 = code & 0xf
    reg[arg1] = tc_div(reg[arg2], reg[arg3])
    pc += 1

def op_mod(code: int):
    global reg, pc
    arg1 = (code & 0xf << 8) >> 8
    arg2 = (code & 0xf << 4) >> 4
    arg3 = code & 0xf
    reg[arg1] = reg[arg2] % reg[arg3]
    pc += 1

def op_mul(code: int):
    global reg, pc
    arg1 = (code & 0xf << 8) >> 8
    arg2 = (code & 0xf << 4) >> 4
    arg3 = code & 0xf
    reg[arg1] = tc_mul(reg[arg2], reg[arg3])
    pc += 1

def op_sub(code: int):
    global reg, pc
    arg1 = (code & 0xf << 8) >> 8
    arg2 = (code & 0xf << 4) >> 4
    arg3 = code & 0xf
    reg[arg1] = tc_sub(reg[arg2], reg[arg3])
    pc += 1

def op_jeqn(code: int):
    global reg, pc
    arg1 = (code & 0xf << 20) >> 20
    arg2 = (code & 0xf << 16) >> 16
    arg3 = code & 0xffff
    pc = arg3 if reg[arg1] == reg[arg2] else pc + 1

def op_jgen(code: int):
    global reg, pc
    arg1 = (code & 0xf << 20) >> 20
    arg2 = (code & 0xf << 16) >> 16 
    arg3 = code & 0xffff
    pc = arg3 if reg[arg1] >= reg[arg2] else pc + 1

def op_jgtn(code: int):
    global reg, pc
    arg1 = (code & 0xf << 20) >> 20
    arg2 = (code & 0xf << 16) >> 16
    arg3 = code & 0xffff
    pc = arg3 if reg[arg1] > reg[arg2] else pc + 1

def op_jlen(code: int):
    global reg, pc
    arg1 = (code & 0xf << 20) >> 20
    arg2 = (code & 0xf << 16) >> 16
    arg3 = code & 0xffff
    pc = arg3 if reg[arg1] <= reg[arg2] else pc + 1

def op_jltn(code: int):
    global reg, pc
    arg1 = (code & 0xf << 20) >> 20
    arg2 = (code & 0xf << 16) >> 16
    arg3 = code & 0xffff
    pc = arg3 if reg[arg1] < reg[arg2] else pc + 1

def op_jnen(code: int):
    global reg, pc
    arg1 = (code & 0xf << 20) >> 20
    arg2 = (code & 0xf << 16) >> 16
    arg3 = code & 0xffff
    pc = arg3 if reg[arg1] != reg[arg2] else pc + 1

def op_seed(code: int):
    global reg, pc
    arg1 = code & 0xf
    random.seed((reg[arg1]))
    pc += 1

def op_rand(code: int):
    global reg, pc
    arg1 = (code & 0xf << 8) >> 8
    arg2 = (code & 0xf << 4) >> 4
    arg3 = code & 0xf
    lo = tc_b32_to_int(reg[arg1])
    hi = tc_b32_to_int(reg[arg2])
    reg[arg3] = tc_int_to_b32(random.randint(lo, hi))

def op_time(code: int):
    raise NotImplementedError

def op_unimp(code: int):
    sys.exit(f"Error: operation {bin_to_opcode[code >> 24]} unimplemented")

# TODO: where to put this? awkward
instructions = {}
for op in opcode_to_bin.keys():
    func = "op_" + op
    instructions[op] = globals()[func] if func in globals() else op_unimp

def tc_int_to_b16(val: int) -> int:
    if val < 0:
        val = -val
        val = (val ^ 0xffff) + 1
    return val

def tc_int_to_b32(val: int) -> int:
    if val < 0:
        val = -val
        val = (val ^ 0xffffffff) + 1
    return val

def tc_b16_to_b32(val: int) -> int:
    if (val & 1 << 15) >> 15:
        val = val | (0xffff << 16)
    return val

def tc_b32_to_int(val: int) -> int:
    if (val & 1 << 31) >> 31:
        val = (val - 1) ^ 0xffffffff
        val = -val
    return val

def tc_neg(val: int) -> int:
    return val ^ (1 << 31)

# TODO: make sure this works 
def tc_add(val1: int, val2: int) -> int:
    return (val1 + val2) & 0xffffffff

# TODO: make sure this works 
def tc_sub(val1: int, val2: int) -> int:
    return (val1 - val2) & 0xffffffff

def tc_mul(val1: int, val2: int) -> int:
    if (val1 & (1 << 31)) >> 31:
        val1 = val1 | (0xffffffff << 8)
    if (val2 & (1 << 31)) >> 31:
        val2 = val2 | (0xffffffff << 8)
    return (val1 * val2) & 0xffffffff

def tc_div(val1: int, val2: int) -> int:
    sign = ((val1 & (1 << 31)) >> 31) ^ ((val2 & (1 << 31)) >> 31)
    return ((val1 & 0x7fffffff) // (val2 & 0x7fffffff)) | (sign << 31)

def tc_mod(val1: int, val2: int) -> int:
    raise NotImplementedError

# TODO: optimize
def fp_float_to_f16(val: float) -> int:
    # find sign
    if val < 0:
        sign = 1
        val = -val
    else:
        sign = 0

    # separate integer and decimal parts
    int_bin = int(val)
    frac_part = val - int_bin

    # determine bit lengths for exponent + noramlization
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

# Returns True if s encodes an integer, and False otherwise.
def isNum(s: str) -> bool:
    ans = True
    try:
        _ = int(s)
    except ValueError:
        ans = False
    return ans

# Return True if n is a valid signed 16-bit integer, and False otherwise.
def validNum(n: int) -> bool:
    return -2 ** 15 <= n <= 2 ** 15 - 1

# Return True if s is "r" followed by a number from the interval [0, 15], and False otherwise.
def validReg(s: str) -> bool:
    return s in reg_to_bin.keys()

if __name__ == "__main__":
    main()
