#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Any

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
opcode2bin = {
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
bin2opcode = {opcode2bin[opcode]: opcode for opcode in opcode2bin.keys()}

# Maps register names to their binary 4-bit codes.
reg2bin = {
    "r0":  0b0000, "r1":  0b0001, "r2":  0b0010, "r3":  0b0011, 
    "r4":  0b0100, "r5":  0b0101, "r6":  0b0110, "r7":  0b0111, 
    "r8":  0b1000, "r9":  0b1001, "r10": 0b1010, "r11": 0b1011,
    "r12": 0b1100, "r13": 0b1101, "r14": 0b1110, "r15": 0b1111
}

def main(argv: list[str]):
    verbose: bool = False
    debug: bool = False
    inFile: str

    # Process command-line inputs and exit if they are not as expected.
    parser = argparse.ArgumentParser(prog="marvin", description=description)
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
    # TODO: fix typing?
    tuples: list[tuple[Any, ...]] = []

    for i, line in enumerate(lines):
        lineno = i + 1
        line = line.strip()
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
        if toks[1] not in opcode2bin:
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction '{toks[1]}'")
        expectedID += 1

        # Validate the instruction arguments.
        # TODO: is there a cleaner way to do this validation?
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
            tuples.append((lineno, id, opcode, int(toks[2])))
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
            tuples.append((lineno, id, opcode, toks[2], int(toks[3])))
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
            tuples.append((lineno, id, opcode, toks[2], toks[3], int(toks[4])))

    # Additional validation of instruction arguments.
    for t in tuples:
        lineno, id, opcode, args = t[0], t[1], t[2], t[3:]
        if opcode in {"jumpn"} and args[0] >= len(tuples):
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction address '{args[0]}'")
        elif opcode in {"addn", "setn"} and not validNum(args[1]):
            sys.exit(f"Error {inFile}@{lineno}: invalid number '{args[1]}'")
        elif opcode in {"loadn", "storen"} and not validNum(args[2]):
            sys.exit(f"Error {inFile}@{lineno}: invalid address '{args[2]}'")
        elif opcode in {"calln", "jeqzn", "jnezn"} and args[1] >= len(tuples):
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction address '{args[1]}'")
        elif opcode in {"jeqn", "jgen", "jgtn", "jlen", "jltn", "jnen"} and args[2] >= len(tuples):
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction address '{args[2]}'")

    # Assemble the instructions into machine codes.
    machineCodes = assemble(tuples, verbose)

    # Simulate the machine codes.
    if len(machineCodes) > 0:
        simulate(machineCodes, debug)

# Assembles the instructions in tuples and returns a list containing the corresponding machine
# codes. Prints the assembled instructions to stdout if verbose is True.
def assemble(tuples: list[tuple[Any, ...]], verbose: bool) -> list[int]:
    machineCodes: list[int] = []
    verboseOutput: list[str] =  []

    for t in tuples:
        id, opcode, args = t[1], t[2], t[3:]
        bArg1, bArg2, bArg3 = 0, 0, 0

        # TODO: is there a cleaner way to do this?
        if opcode in {"halt", "nop"}:
            bArg1, bArg2, bArg3 = 0, 0, 0
        elif opcode in {"jumpr", "read", "set0", "set1", "write"}:
            bArg1, bArg2, bArg3 = 0, 0, reg2bin[args[0]]
        elif opcode in {"jumpn"}:
            val = int(args[0])
            bArg1, bArg2, bArg3 = 0, val >> 8, val & ~(0xFF << 8),
        elif opcode in {"copy", "loadr", "neg", "popr", "pushr", "storer"}:
            bArg1, bArg2, bArg3 = 0, 0, reg2bin[args[0]] << 4 | reg2bin[args[1]]
        elif opcode in {"addn", "setn"}:
            bArg1 = reg2bin[args[0]]
            val = int(args[1])
            if (val < 0):
                val = -val
                val = val | 1 << 15
            bArg2, bArg3 = val >> 8, val & ~(0xFF << 8),
        elif opcode in {"calln", "jeqzn", "jnezn"}:
            bArg1 = reg2bin[args[0]]
            val = int(args[1])
            bArg2, bArg3 =  val >> 8, val & ~(0xFF << 8),
        elif opcode in {"add", "div", "mod", "mul", "sub"}:
            bArg1, bArg2, bArg3 = 0, reg2bin[args[0]], reg2bin[args[1]] << 4 | reg2bin[args[2]]
        elif opcode in {"jeqn", "jgen", "jgtn", "jlen", "jltn", "jnen"}:
            bArg1 = reg2bin[args[0]] << 4 | reg2bin[args[1]]
            val = int(args[2])
            bArg2, bArg3 = val >> 8, val & ~(0xFF << 8),
        elif opcode in {"loadn", "storen"}:
            bArg1 = reg2bin[args[0]] << 4 | reg2bin[args[1]]
            val = int(args[2])
            if (val < 0):
                val = -val
                val = val | 1 << 15
            bArg2, bArg3 = val >> 8, val & ~(0xFF << 8),

        op = opcode2bin[opcode]
        code = op << 24 | bArg1 << 16 | bArg2 << 8 | bArg3
        machineCodes.append(code)

        if verbose:
            # TODO: ...again, is there a cleaner way to do this?
            aArg1, aArg2, aArg3 = "", "", ""
            if opcode in {"jumpn", "jumpr", "read", "set0", "set1", "write"}:
                aArg1 = args[0]
            elif opcode in {"addn", "setn", "calln", "copy", "jeqzn", "jnezn", "loadr", "neg", "popr", "pushr", "storer"}:
                aArg1, aArg2 = args[0], args[1]
            elif opcode in {"add", "div", "jeqn", "jgen", "jgtn", "jlen", "jltn", "jnen", "loadn", "mod", "mul", "storen", "sub"}:
                aArg1, aArg2, aArg3 = args[0], args[1], args[2]

            bArg1 = format(bArg1, '08b')
            bArg2 = format(bArg2, '08b')
            bArg3 = format(bArg3, '08b')

            binCode = f"{id: >5}: {format(opcode2bin[opcode], '08b')} {bArg1} {bArg2} {bArg3}"
            asmCode = f"{id: >5}: {opcode: <6} {aArg1} {aArg2} {aArg3}"
            verboseOutput.append(f"{binCode: <50}        {asmCode}")

    if verbose:
        for s in verboseOutput:
            print(s)
        print()

    return machineCodes

# Simulate the assembled instructions in machineCodes.
def simulate(machineCodes: list[int], debug: bool):
    reg = [0] * 16     # registers
    mem = [0] * 65536  # main memory
    pc = 0             # program counter
    ir = 0             # instruction register

    # Initialize the frame and stack pointers to 65535 (the base address of the stack).
    reg[14], reg[15] = 65535, 65535

    # Load the machine codes into memory starting at location 0.
    for i, v in enumerate(machineCodes):
        mem[i] = v

    # Find end of stack
    stack_end = len(machineCodes)

    while True:
        # Fetch the next instruction to simulate.
        try:
            ir = machineCodes[pc]
        except IndexError:
            sys.exit(f"Error: attempted to execute mem['{pc}']; halting the machine")


        # Extract the opcode.
        code = ir
        op = code >> 24
        opcode = bin2opcode[op]

        # Debug stub
        if debug:
            # TODO: actually have a function here
            # - args: mem, reg, code, pc
            print(f"pc: {pc} opcode: {opcode}")
            print(f"r0:  {reg[0]: 10d} r1:  {reg[1]: 10d} r2:  {reg[2]: 10d} r3:  {reg[3]: 10d}")
            print(f"r4:  {reg[4]: 10d} r5:  {reg[5]: 10d} r6:  {reg[6]: 10d} r7:  {reg[7]: 10d}")
            print(f"r8:  {reg[8]: 10d} r9:  {reg[9]: 10d} r10: {reg[10]: 10d} r11: {reg[11]: 10d}")
            print(f"r12: {reg[12]: 10d} r13: {reg[13]: 10d} r14: {reg[14]: 10d} r15: {reg[15]: 10d}")
            for i in range(65535, reg[15] - 1, -1):
                if i == reg[14]:
                    print("*", end="")
                if i == reg[15]:
                    print("^", end="")
                print(mem[i])
            _ = input()

        # Simulation the instruction given by opcode.

        # TODO: dict of funcs like HMMM
        # - args: mem, reg, pc, code (make global?)

        # halt
        if opcode == "halt":
            break
        # read
        elif opcode == "read":
            arg1 = code & 0xF
            while True:
                try:
                    x = int(input())
                    if (validNum(x)):
                        break
                    raise ValueError
                except ValueError:
                    print("Illegal input: number must be in [-32768, 32767]")
            reg[arg1] = x
            pc += 1
        # write
        elif opcode == "write":
            arg1 = code & 0xF
            print(reg[arg1])
            pc += 1
        # nop
        elif opcode == "nop":
            pc += 1
            continue
        # set0
        elif opcode == "set0":
            arg1 = code & 0xF
            reg[arg1] = 0
            pc += 1
        # set1
        elif opcode == "set1":
            arg1 = code & 0xF
            reg[arg1] = 1
            pc += 1
        # jumpr
        elif opcode == "jumpr":
            arg1 = code & 0xF
            pc = reg[arg1]
        # jumpn
        elif opcode == "jumpn":
            arg1 = code & 0xFFFF
            pc = arg1
        # copy
        elif opcode == "copy":
            arg1 = (code & 0xF << 4) >> 4
            arg2 = code & 0xF
            reg[arg1] = reg[arg2]
            pc += 1
        # loadr
        elif opcode == "loadr":
            arg1 = (code & 0xF << 4) >> 4
            arg2 = code & 0xF
            reg[arg1] = mem[reg[arg2]]
            pc += 1
        # neg
        elif opcode == "neg":
            arg1 = (code & 0xF << 4) >> 4
            arg2 = code & 0xF
            reg[arg1] = -reg[arg2]
            pc += 1
        # popr
        elif opcode == "popr":
            arg1 = (code & 0xF << 4) >> 4
            arg2 = code & 0xF
            reg[arg2] += 1
            reg[arg1] = mem[reg[arg2]]
            pc += 1
        # pushr
        elif opcode == "pushr":
            if reg[15] == stack_end:
                sys.exit(f"Error: stack overflow attempting to execute mem['{pc}']; halting the machine")
            arg1 = (code & 0xF << 4) >> 4
            arg2 = code & 0xF
            mem[reg[arg2]] = reg[arg1]
            reg[arg2] -= 1
            pc += 1
        # storer
        elif opcode == "storer":
            arg1 = (code & 0xF << 4) >> 4
            arg2 = code & 0xF
            mem[reg[arg2]] = reg[arg1]
            pc += 1
        # addn
        elif opcode == "addn":
            arg1 = (code & 0xF << 16) >> 16
            arg2 = code & 0xFFFF
            if ((arg2 & 0b1 << 15) >> 15):
                arg2 = arg2 & ~(0b1 << 15)
                arg2 = -arg2
            reg[arg1] += arg2
            pc += 1
        # calln
        elif opcode == "calln":
            arg1 = (code & 0xF << 16) >> 16
            arg2 = code & 0xFFFF
            reg[arg1] = pc + 1
            pc = arg2
        # jeqzn
        elif opcode == "jeqzn":
            arg1 = (code & 0xF << 16) >> 16
            arg2 = code & 0xFFFF
            pc = arg2 if reg[arg1] == 0 else pc + 1
        # jnezn
        elif opcode == "jnezn":
            arg1 = (code & 0xF << 16) >> 16
            arg2 = code & 0xFFFF
            pc = arg2 if reg[arg1] != 0 else pc + 1
        # loadn
        elif opcode == "loadn":
            arg1 = (code & 0xF << 20) >> 20
            arg2 = (code & 0xF << 16) >> 16
            arg3 = code & 0xFFFF
            if ((arg3 & 0b1 << 15) >> 15):
                arg3 = arg3 & ~(0b1 << 15)
                arg3 = -arg3
            reg[arg1] = mem[reg[arg2] + arg3]
            pc += 1
        # setn
        elif opcode == "setn":
            arg1 = (code & 0xF << 16) >> 16 
            arg2 = code & 0xFFFF
            if ((arg2 & 0b1 << 15) >> 15):
                arg2 = arg2 & ~(0b1 << 15)
                arg2 = -arg2
            reg[arg1] = arg2
            pc += 1
        # storen
        elif opcode == "storen":
            arg1 = (code & 0xF << 20) >> 20
            arg2 = (code & 0xF << 16) >> 16
            arg3 = code & 0xFFFF
            if ((arg3 & 0b1 << 15) >> 15):
                arg3 = arg3 & ~(0b1 << 15)
                arg3 = -arg3
            mem[reg[arg2] + arg3] = reg[arg1]
            pc += 1
        # add
        elif opcode == "add":
            arg1 = (code & 0xF << 8) >> 8
            arg2 = (code & 0xF << 4) >> 4 
            arg3 = code & 0xF
            reg[arg1] = reg[arg2] + reg[arg3]
            pc += 1
        # div
        elif opcode == "div":
            arg1 = (code & 0xF << 8) >> 8
            arg2 = (code & 0xF << 4) >> 4
            arg3 = code & 0xF
            reg[arg1] = reg[arg2] // reg[arg3]
            pc += 1
        # mod
        elif opcode == "mod":
            arg1 = (code & 0xF << 8) >> 8
            arg2 = (code & 0xF << 4) >> 4
            arg3 = code & 0xF
            reg[arg1] = reg[arg2] % reg[arg3]
            pc += 1
        # mul
        elif opcode == "mul":
            arg1 = (code & 0xF << 8) >> 8
            arg2 = (code & 0xF << 4) >> 4
            arg3 = code & 0xF
            reg[arg1] = reg[arg2] * reg[arg3]
            pc += 1
        # sub
        elif opcode == "sub":
            arg1 = (code & 0xF << 8) >> 8
            arg2 = (code & 0xF << 4) >> 4
            arg3 = code & 0xF
            reg[arg1] = reg[arg2] - reg[arg3]
            pc += 1
        # jeqn
        elif opcode == "jeqn":
            arg1 = (code & 0xF << 20) >> 20
            arg2 = (code & 0xF << 16) >> 16
            arg3 = code & 0xFFFF
            pc = arg3 if reg[arg1] == reg[arg2] else pc + 1
        # jgen
        elif opcode == "jgen":
            arg1 = (code & 0xF << 20) >> 20
            arg2 = (code & 0xF << 16) >> 16 
            arg3 = code & 0xFFFF
            pc = arg3 if reg[arg1] >= reg[arg2] else pc + 1
        # jgtn
        elif opcode == "jgtn":
            arg1 = (code & 0xF << 20) >> 20
            arg2 = (code & 0xF << 16) >> 16
            arg3 = code & 0xFFFF
            pc = arg3 if reg[arg1] > reg[arg2] else pc + 1
        # jlen
        elif opcode == "jlen":
            arg1 = (code & 0xF << 20) >> 20
            arg2 = (code & 0xF << 16) >> 16
            arg3 = code & 0xFFFF
            pc = arg3 if reg[arg1] <= reg[arg2] else pc + 1
        # jltn
        elif opcode == "jltn":
            arg1 = (code & 0xF << 20) >> 20
            arg2 = (code & 0xF << 16) >> 16
            arg3 = code & 0xFFFF
            pc = arg3 if reg[arg1] < reg[arg2] else pc + 1
        # jnen
        elif opcode == "jnen":
            arg1 = (code & 0xF << 20) >> 20
            arg2 = (code & 0xF << 16) >> 16
            arg3 = code & 0xFFFF
            pc = arg3 if reg[arg1] != reg[arg2] else pc + 1

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
    return s in {"r" + str(i) for i in range(16)}

if __name__ == "__main__":
    main(sys.argv)
