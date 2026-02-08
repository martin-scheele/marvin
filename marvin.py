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
    "halt":   0b00000000, "readi":   0b00000001, "readf":   0b00000010, "readc":   0b00000011,
    "writei": 0b00000100, "writef":  0b00000101, "writec":  0b00000110, "seed":    0b00000111,
    "rand":   0b00001000, "time":    0b00001001, "date":    0b00001010, "nop":     0b00001111,
    # arithmetic instructions
    "add":    0b00010000, "sub":     0b00010001, "mul":     0b00010010, "div":     0b00010011,
    "mod":    0b00010100, "neg":     0b00010101, "fadd":    0b00010110, "fsub":    0b00010111,
    "fmul":   0b00011000, "fdiv":    0b00011001, "fneg":    0b00011010,
    # bitwise instructions
    "and":    0b00100000, "or":      0b00100001, "xor":     0b00100010, "not":     0b00100011,
    "lshl":   0b00100100, "lshr":    0b00100101, "ashl":    0b00100110, "ashr":    0b00100111,
    # jump instructions
    "jumpn":  0b00110000, "jumpr":   0b00110001, "jeqzn":   0b00110010, "jnezn":   0b00110011,
    "jgen":   0b00110100, "jlen":    0b00110101, "jeqn":    0b00110110, "jnen":    0b00110111,
    "jgtn":   0b00111000, "jltn":    0b00111001, "calln":   0b00111010,
    # register instructions
    "seti":   0b01000000, "addi":    0b01000001, "setf":    0b01000010, "addf":    0b01000011,
    "copy":   0b01000100,
    # stack insructions
    "pushrb": 0b01010000, "poprb":   0b01010001, "pushrs":  0b01010010, "poprs":   0b01010011,
    "pushrw": 0b01010100, "poprw":   0b01010101,
    # load/store instructions
    "loadnb": 0b01100000, "storenb": 0b01100001, "loadrb":  0b01100010, "storerb": 0b01100011,
    "loadns": 0b01100100, "storens": 0b01100101, "loadrs":  0b01100110, "storers": 0b01100111,
    "loadnw": 0b01101000, "storenw": 0b01101001, "loadrw":  0b01101010, "storerw": 0b01101011,
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
    "jumpn":  "l",   "jumpr":  "r",   "jeqzn":  "rl",  "jnezn":  "rl",
    "jgen":   "rrl", "jlen":   "rrl", "jeqn":   "rrl", "jnen":   "rrl",
    "jgtn":   "rrl", "jltn":   "rrl", "calln":  "rl",
    # register instructions
    "seti":   "rn",  "addi":   "rn",  "setf":   "rf",  "addf":   "rf",
    "copy":   "rr",
    # stack insructions
    "pushrb": "rr",  "poprb":  "rr",  "pushrs": "rr", "poprs":   "rr",
    "pushrw": "rr",  "poprw":  "rr",
    # load/store instructions
    "loadnb": "rrn", "storenb": "rrn", "loadrb": "rr", "storerb": "rr",
    "loadns": "rrn", "storens": "rrn", "loadrs": "rr", "storers": "rr",
    "loadnw": "rrn", "storenw": "rrn", "loadrw": "rr", "storerw": "rr",
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

# Global argument variables.
verbose = False
verbose_output: list[str] = []
debug = False
count_calls = False

# determine getch function

def main():
    # Process command-line inputs and exit if they are not as expected.
    parser = argparse.ArgumentParser(description=description)
    _ = parser.add_argument("filename", help="input .marv file")
    _ = parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    _ = parser.add_argument("-d", "--debug",   action="store_true", help="enable debug mode")
    _ = parser.add_argument("-c", "--count",   action="store_true", help="count instruction calls")
    args = parser.parse_args()

    global debug, verbose, count_calls
    inFile = args.filename
    verbose = args.verbose
    debug = args.debug
    count_calls = args.count

    if not inFile.endswith(".marv") or not os.path.exists(inFile):
        sys.exit(f"Error: invalid file '{inFile}'")

    tuples = validate(inFile)
    machine_code = assemble(tuples)

    if verbose:
        print_verbose_output()

    # Simulate the machine codes.
    if len(machine_code) > 0:
        simulate(machine_code)

labels: dict[str, int] = {}

def validate(inFile: str) -> list[tuple[str, *tuple[str, ...]]]:
    with open(inFile, "r") as fh:
        lines = fh.readlines()

    tuples: list[tuple[str, *tuple[str, ...]]] = []

    instruction_number = 0

    # Scan for label validation
    for i, line in enumerate(lines):
        line = line.strip().lower()
        lineno = i + 1

        # Skip empty lines and comments.
        if not line or line.startswith("#"):
            continue

        # Remove inlined comment if any.
        if "#" in line:
            line = line[:line.find("#")].strip()

        # Find labels.
        if ":" in line:
            if not line.endswith(":"):
                sys.exit(f"Error {inFile}@{lineno}: invalid label {line}")
            label = line[:-1]
            if label in labels.keys():
                sys.exit(f"Error: {inFile}@{lineno}: duplicate label {label}")
            labels[label] = instruction_number
            continue

        instruction_number += 1

    for i, line in enumerate(lines):
        line = line.strip().lower()
        lineno = i + 1

        # Skip empty lines, comments, and labels.
        if not line or line.startswith("#") or line.endswith(":"):
            continue

        # Remove inlined comment if any.
        if "#" in line:
            line = line[:line.find("#")].strip()

        toks = line.split()

        # Exit with error if the instruction is invalid.
        if toks[0] not in opcode_to_bin:
            sys.exit(f"Error {inFile}@{lineno}: invalid instruction '{toks[0]}'")

        # Validate the instruction arguments.
        opcode, args = toks[0], toks[1:]
        if len(args) != len(opcode_to_argmask[opcode]):
            argmask = opcode_to_argmask[opcode]
            sys.exit(f"Error {inFile}@{lineno}: opcode expects {len(argmask)} arguments: '{argmask}'")
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
            elif c == "l":
                if not valid_label(args[i]):
                    sys.exit(f"Error {inFile}@{lineno}: invalid label '{args[i]}'")
                if labels[args[i]] >= instruction_number:
                    sys.exit(f"Error {inFile}@{lineno}: missing instruction after label '{args[i]}'")

        # Append valid tuple.
        tuples.append((opcode, *args))

    # TODO: confirm labels point to instruction

    return tuples

def valid_label(label: str) -> bool:
    return label in labels.keys()

# Assembles the instructions in tuples and returns a list containing the corresponding machine
# codes. Prints the assembled instructions to stdout if verbose is True.
def assemble(tuples: list[tuple[str, *tuple[str, ...]]]) -> list[tuple[int, int, int, int]]:
    machine_code: list[tuple[int, int, int, int]] = []

    for id, t in enumerate(tuples):
        opcode, args = t[0], t[1:]

        # TODO: do something that is not this...?
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
            elif c == "l":
                val = tc_int_to_b16(labels[args[i]])
                byte_list[curr_byte] = val & 0xff
                byte_list[curr_byte - 1] = val >> 8
                curr_byte -= 2

            i -= 1

        code = (byte_list[0], byte_list[1], byte_list[2], byte_list[3])
        machine_code.append(code)

        if verbose or debug:
            binCode = f"{id: >5}: {" ".join([format(byte, "08b") for byte in byte_list])}"
            asmCode = f"{id: >5}: {opcode: <6} {" ".join(args)}"
            verbose_output.append(f"{binCode: <50} {asmCode}")

    # TODO: this is awkward - find a way to do it in one place
    if verbose or debug:
        for label in reversed(labels.keys()):
            verbose_output.insert(labels[label], f"\t{label + ":": >46}")

    return machine_code

# Global CPU variables.
reg = [0] * 16      # registers
mem = [0] * 65536   # main memory
pc = 0              # program counter
ir = 0              # instruction register

WORD_SIZE  = 4
SHORT_SIZE = 2
BYTE_SIZE  = 1

STACK_MAX = 65535
HEAP_START = 8192

def step_pc():
    global pc
    pc += WORD_SIZE

# Simulate the assembled instructions in machineCodes.
def simulate(machine_code: list[tuple[int, int, int, int]]):
    global reg, mem, ir, pc

    # Initialize the frame, stack, and global pointers
    reg[reg_to_bin["fp"]] = reg[reg_to_bin["sp"]] = STACK_MAX
    reg[reg_to_bin["gp"]] = HEAP_START

    # Load the machine code into memory starting at location 0.
    for i, v in enumerate(machine_code):
        for b in range(4):
            mem[i * 4 + b] = v[b]

    while True:
        # Fetch the next instruction to simulate.
        try:
            ir = mem[pc] << 24 | mem[pc+1] << 16 | mem[pc+2] << 8 | mem[pc+3]
        except IndexError:
            sys.exit(f"Error: attempted to execute instruction {pc // WORD_SIZE}; halting the machine")

        # Conditionally execute the debugger.
        if debug: 
            debug_exec()

        # Extract the opcode from the instruction register.
        opcode = bin_to_opcode[ir >> 24]

        # Extract arguments from the instruction register.
        args = extract_args(ir, opcode_to_argmask[opcode])

        # Simulate the instruction for the opcode.
        instructions[opcode](args)

        # Count instruction calls
        if count_calls:
            opcode_to_calls[opcode] += 1
            # TODO: actually delay and visualize

# Debug globals.
breakpoints: set[int] = set()
continue_debug: bool = False

def debug_exec():
    global reg, mem, pc, ir
    global breakpoints, continue_debug

    # Check for breakpoints.
    if pc // 4 in breakpoints:
        pass
    elif continue_debug:
        return

    # Print current instruction; register and stack contents
    print(verbose_output[pc // 4], end="\n\n")

    # Get debug input.
    while cmd := input("> "):
        if not (tokens := cmd.split()):
            break
        cmd = tokens[0]
        args = [] if len(tokens) == 1 else tokens[1:]
        if cmd == "c" or cmd == "continue":
            continue_debug = True
            break
        elif cmd == "s" or cmd == "step":
            continue_debug = False
            break
        elif cmd == "b" or cmd == "break":
            if not args:
                breakpoints.add(pc // 4)
            try:
                breakpoints.add(int(args[0]))
            except ValueError:
                print(f"Invalid breakpoint: {args[0]}")
        elif cmd == "d" or cmd == "disable" or cmd == "delete":
            if not args:
                breakpoints = set()
            try:
                breakpoints.remove(int(args[0]))
            except ValueError:
                print(f"Invalid breakpoint: {args[0]}")
        elif cmd == "l" or cmd == "list":
            print(f"breakpoints: {", ".join([str(bp) for bp in breakpoints])}")
        elif cmd == "p" or cmd == "print":
            if not args:
                print_regs()
                print_stack(STACK_MAX, tc_b32_to_int(reg[reg_to_bin["sp"]]))
                continue
            if args[0] not in {"stack", "reg", "s", "r"}:
                print(f"Invalid print object: {args[0]}")
                continue
            if args[0] in {"stack", "s"}:
                # case 1: print entire stack
                if len(args) == 1:
                    print_stack(STACK_MAX, tc_b32_to_int(reg[reg_to_bin["sp"]]))
                    continue
                # case 2: print stack range
                # print stack <start> <stop>
                if len(args) != 3:
                    print(f"Invalid syntax: {"TODO"}")
                    continue
                if int(args[1]) > STACK_MAX or int(args[1]) < 0 \
                    or int(args[2]) > STACK_MAX or int(args[2]) < 0 \
                    or int(args[1]) > int(args[2]):
                    print(f"Invalid stack values: {args[1]} {args[2]}")
                    continue
                print_stack(int(args[2]), int(args[1]))

            elif args[0] in {"reg", "r"}:
                # case 1: print all regs
                # print reg
                if len(args) == 1:
                    print_regs()
                    continue
                # case 2: specific reg
                # print reg <reg>
                if args[1] not in reg_to_bin.keys():
                    print(f"Invalid register: {args[1]}")
                    continue
                if len(args) == 2:
                    print(tc_b32_to_int(reg[reg_to_bin[args[1]]]))
                    continue
                # case 3: print reg <reg> <type>
                if len(args) != 3:
                    print(f"Invalid syntax: {"TODO"}")
                    continue
                if args[2] not in {"int", "float", "char", "bin", "i", "f", "c", "b"}:
                    print(f"Invalid type: {args[2]}")
                    continue
                if args[2] in {"int", "i"}:
                    print(tc_b32_to_int(reg[reg_to_bin[args[1]]]))
                elif args[2] in {"float", "f"}:
                    print(fp_f32_to_float(reg[reg_to_bin[args[1]]]))
                elif args[2] in {"char", "c"}:
                    print(chr(reg[reg_to_bin[args[1]]]))
                elif args[2] in {"bin", "b"}:
                    print(format(reg[reg_to_bin[args[1]]], "032b"))

        elif cmd == "q" or cmd == "quit":
            global debug
            debug = False
            return
        elif cmd == "h" or cmd == "help":
            # TODO: de-indent this some other way
            help_str = """
List of commands:

    help, h -- print this command
        Usage: [h | help]

    quit, q -- exit the debugger
        Usage: [q | quit]

    continue, c -- continue to next breakpoint
        Usage: [c | continue]

    step, s -- step to next instruction
        Usage: [s | step]

    break, b -- set a breakpoint
        Usage: [b | break] <instruction>

    delete, d -- delete a breakpoint
        Usage: [d | delete] <instruction>

    list, l -- list breakpoints
        Usage: [l | list]

    print, p -- print register or stack information
        Usage: [p | print] [s | stack] [<start> <stop>]
               [p | print] [r | reg] [<reg>] [<type>]
            """
            print(help_str)
        else:
            print(f"Invalid command: {cmd}")
    print()


def print_regs():
    print(f"r0:  [{format_reg(0)  :>10}] r1:  [{format_reg(1)  :>10}] r2:  [{format_reg(2)  :>10}] r3:  [{format_reg(3)  :>10}]")
    print(f"r4:  [{format_reg(4)  :>10}] r5:  [{format_reg(5)  :>10}] r6:  [{format_reg(6)  :>10}] r7:  [{format_reg(7)  :>10}]")
    print(f"r8:  [{format_reg(8)  :>10}] r9:  [{format_reg(9)  :>10}] r10: [{format_reg(10) :>10}] ra:  [{format_reg(11) :>10}]")
    print(f"rv:  [{format_reg(12) :>10}] fp:  [{format_reg(13) :>10}] sp:  [{format_reg(14) :>10}] gp:  [{format_reg(15) :>10}]")
    print()

def print_stack(hi: int, lo: int):
    for i in range(hi, lo, -4):
        sym = ["*" if i == reg[reg_to_bin["fp"]] else " ", ">" if i == reg[reg_to_bin["sp"]] else " "]
        print(f"{"".join(sym)} {i:04x}: {" ".join(format(byte, "08b") for byte in mem[i - 3 : i + 1])}")
    print()

def format_reg(regbin: int) -> str:
    return f"{tc_b32_to_int(reg[regbin])}"

# System Instructions

def op_halt(_):
    if count_calls:
        print_opcode_cost()
    sys.exit()

def op_readi(args: list[int]):
    global reg
    while True:
        try:
            x = int(input())
            if (valid_int(x)):
                break
            raise ValueError
        except ValueError:
            print("Illegal input: input must be a number must be in [-32768, 32767]")
    reg[args[0]] = tc_int_to_b32(x)
    step_pc()


def op_readf(args: list[int]):
    global reg
    while True:
        try:
            x = float(input())
            break
        except ValueError:
            # TODO: better error message
            print("Illegal input: input must be a number")
    reg[args[0]] = fp_float_to_f32(x)
    step_pc()

def op_readc(args: list[int]):
    global reg
    reg[args[0]] = ord(getch())
    step_pc()

def op_writei(args: list[int]):
    print(tc_b32_to_int(reg[args[0]]))
    step_pc()

def op_writef(args: list[int]):
    print(fp_f32_to_float(reg[args[0]]))
    step_pc()

def op_writec(args: list[int]):
    # TODO: how to handle value outside of range? clamp?
    print(chr(reg[args[0]]))
    step_pc()

def op_seed(args: list[int]):
    random.seed((reg[args[0]]))
    step_pc()

def op_rand(args: list[int]):
    global reg
    lo = tc_b32_to_int(reg[args[0]])
    hi = tc_b32_to_int(reg[args[1]])
    reg[args[2]] = tc_int_to_b32(random.randint(lo, hi))
    step_pc()

def op_time(args: list[int]):
    global reg
    now = datetime.datetime.now()
    reg[args[0]] = (now.hour * 3600 + now.minute * 60 + now.second) * 1000 + now.microsecond // 1000
    step_pc()


def op_date(args: list[int]):
    global reg
    today = datetime.date.today()
    # [31:13]: year  - 19 bits - 524287 values
    # [12:9]:  month - 4  bits - 16     values
    # [8:0]:   day   - 9  bits - 512    values
    reg[args[0]] = today.year << 13 | today.month << 9 | today.day
    step_pc()

def op_nop(_):
    step_pc()

# Arithmetic instructions

def op_neg(args: list[int]):
    global reg
    reg[args[0]] = tc_neg(reg[args[1]])
    step_pc()

def op_add(args: list[int]):
    global reg
    reg[args[0]] = tc_add(reg[args[1]], reg[args[2]])
    step_pc()

def op_sub(args: list[int]):
    global reg
    reg[args[0]] = tc_sub(reg[args[1]], reg[args[2]])
    step_pc()

def op_mul(args: list[int]):
    global reg
    reg[args[0]] = tc_mul(reg[args[1]], reg[args[2]])
    step_pc()

def op_div(args: list[int]):
    global reg
    reg[args[0]] = tc_div(reg[args[1]], reg[args[2]])
    step_pc()

def op_mod(args: list[int]):
    global reg
    reg[args[0]] = tc_mod(reg[args[1]],  reg[args[2]])
    step_pc()

def op_fneg(args: list[int]):
    global reg
    reg[args[0]] = reg[args[1]] ^ (1 << 31)
    step_pc()

def op_fadd(args: list[int]):
    global reg
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[1]]) + fp_f32_to_float(reg[args[2]]))
    step_pc()

def op_fsub(args: list[int]):
    global reg
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[1]]) - fp_f32_to_float(reg[args[2]]))
    step_pc()

def op_fmul(args: list[int]):
    global reg
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[1]]) * fp_f32_to_float(reg[args[2]]))
    step_pc()

def op_fdiv(args: list[int]):
    global reg
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[1]]) / fp_f32_to_float(reg[args[2]]))
    step_pc()

# Bitwise instructions

def op_and(args: list[int]):
    global reg
    reg[args[0]] = reg[args[1]] & reg[args[2]]
    step_pc()

def op_or(args: list[int]):
    global reg
    reg[args[0]] = reg[args[1]] | reg[args[2]]
    step_pc()

def op_xor(args: list[int]):
    global reg
    reg[args[0]] = reg[args[1]] ^ reg[args[2]]
    step_pc()

def op_not(args: list[int]):
    global reg
    reg[args[0]] = ~reg[args[1]]
    step_pc()

def op_lshl(args: list[int]):
    global reg
    reg[args[0]] = (reg[args[1]] << reg[args[2]]) & 0xffffffff
    step_pc()

def op_lshr(args: list[int]):
    global reg
    reg[args[0]] = (reg[args[1]] >> reg[args[2]]) & 0xffffffff
    step_pc()

# TODO: verify this
def op_ashl(args: list[int]):
    global reg
    sign = reg[args[1]] & (1 << 31)
    temp = ((reg[args[1]] ^ sign) << reg[args[2]]) & 0xffffffff
    reg[args[0]] = 0 if temp == 0 else temp | sign
    step_pc()

# TODO: verify this
def op_ashr(args: list[int]):
    global reg
    num_shifts = tc_b32_to_int(reg[args[2]])
    sign = reg[args[1]] & (1 << 31)
    sign_extend = int("1" * num_shifts, 2) if sign else 0
    temp = ((reg[args[1]] ^ sign) >> reg[args[2]]) | (sign_extend << (31 - num_shifts))
    reg[args[0]] = 0 if temp == 0 else temp | sign
    step_pc()

# Jump Instructions

def op_jumpn(args: list[int]):
    global pc
    pc = args[0] * WORD_SIZE

def op_jumpr(args: list[int]):
    global pc
    pc = reg[args[0]]

def op_jeqzn(args: list[int]):
    global pc
    pc = args[1] * WORD_SIZE if tc_b32_to_int(reg[args[0]]) == 0 else pc + WORD_SIZE

def op_jnezn(args: list[int]):
    global pc
    pc = args[1] * WORD_SIZE if tc_b32_to_int(reg[args[0]]) != 0 else pc + WORD_SIZE

def op_jgen(args: list[int]):
    global pc
    pc = args[2] * WORD_SIZE if tc_b32_to_int(reg[args[0]]) >= tc_b32_to_int(reg[args[1]]) else pc + WORD_SIZE

def op_jlen(args: list[int]):
    global pc
    pc = args[2] * WORD_SIZE if tc_b32_to_int(reg[args[0]]) <= tc_b32_to_int(reg[args[1]]) else pc + WORD_SIZE

def op_jeqn(args: list[int]):
    global pc
    pc = args[2] * WORD_SIZE if tc_b32_to_int(reg[args[0]]) == tc_b32_to_int(reg[args[1]]) else pc + WORD_SIZE

def op_jnen(args: list[int]):
    global pc
    pc = args[2] * WORD_SIZE if tc_b32_to_int(reg[args[0]]) != tc_b32_to_int(reg[args[1]]) else pc + WORD_SIZE

def op_jgtn(args: list[int]):
    global pc
    pc = args[2] * WORD_SIZE if tc_b32_to_int(reg[args[0]]) > tc_b32_to_int(reg[args[1]]) else pc + WORD_SIZE

def op_jltn(args: list[int]):
    global pc
    pc = args[2] * WORD_SIZE if tc_b32_to_int(reg[args[0]]) < tc_b32_to_int(reg[args[1]]) else pc + WORD_SIZE

def op_calln(args: list[int]):
    global reg, pc
    # I think this is guaranteed to never be over integer limit
    reg[args[0]] = pc + WORD_SIZE
    pc = args[1] * WORD_SIZE

# Register instructions

def op_seti(args: list[int]):
    global reg
    reg[args[0]] = tc_b16_to_b32(args[1])
    step_pc()

def op_addi(args: list[int]):
    global reg
    reg[args[0]] = tc_add(reg[args[0]], tc_b16_to_b32(args[1]))
    step_pc()

def op_setf(args: list[int]):
    global reg
    reg[args[0]] = fp_f16_to_f32(args[1])
    step_pc()

def op_addf(args: list[int]):
    global reg
    reg[args[0]] = fp_float_to_f32(fp_f32_to_float(reg[args[0]]) + fp_f32_to_float(reg[args[1]]))
    step_pc()

def op_copy(args: list[int]):
    global reg
    reg[args[0]] = reg[args[1]]
    step_pc()

# Stack instructions

# TODO: clean up generic code for single bytes etc

def op_pushrb(args: list[int]):
    global reg, mem
    if reg[reg_to_bin["sp"]] <= reg[reg_to_bin["gp"]]:
        sys.exit(f"Error: stack overflow attempting to execute instruction {pc // WORD_SIZE}; halting the machine")
    mem[reg[args[1]] - 0] = reg[args[0]] & 0xff
    reg[args[1]] = tc_sub(reg[args[1]], BYTE_SIZE)
    step_pc()

def op_poprb(args: list[int]):
    global reg
    reg[args[1]] = tc_add(reg[args[1]], BYTE_SIZE)
    word = mem[reg[args[1]] : reg[args[1]] - BYTE_SIZE : -1]
    reg[args[0]] = word[0]
    step_pc()

def op_pushrs(args: list[int]):
    global reg, mem
    if reg[reg_to_bin["sp"]] <= reg[reg_to_bin["gp"]]:
        sys.exit(f"Error: stack overflow attempting to execute instruction {pc // WORD_SIZE}; halting the machine")
    mem[reg[args[1]] - 0] = reg[args[0]] & 0xff
    mem[reg[args[1]] - 1] = (reg[args[0]] >> 8) & 0xff
    reg[args[1]] = tc_sub(reg[args[1]], SHORT_SIZE)
    step_pc()

def op_poprs(args: list[int]):
    global reg
    reg[args[1]] = tc_add(reg[args[1]], SHORT_SIZE)
    word = mem[reg[args[1]] : reg[args[1]] - SHORT_SIZE : -1]
    reg[args[0]] = word[1] << 8 | word[0]
    step_pc()

def op_pushrw(args: list[int]):
    global reg, mem
    if reg[reg_to_bin["sp"]] <= reg[reg_to_bin["gp"]]:
        sys.exit(f"Error: stack overflow attempting to execute instruction {pc // WORD_SIZE}; halting the machine")
    mem[reg[args[1]] - 0] = reg[args[0]] & 0xff
    mem[reg[args[1]] - 1] = (reg[args[0]] >> 8) & 0xff
    mem[reg[args[1]] - 2] = (reg[args[0]] >> 16) & 0xff
    mem[reg[args[1]] - 3] = (reg[args[0]] >> 24) & 0xff
    reg[args[1]] = tc_sub(reg[args[1]], WORD_SIZE)
    step_pc()

def op_poprw(args: list[int]):
    global reg
    reg[args[1]] = tc_add(reg[args[1]], WORD_SIZE)
    word = mem[reg[args[1]] : reg[args[1]] - WORD_SIZE : -1]
    reg[args[0]] = word[3] << 24 | word[2] << 16 | word[1] << 8 | word[0]
    step_pc()

# Load/store instructions

def op_loadnb(args: list[int]):
    global reg
    addr = tc_add(reg[args[1]], tc_b16_to_b32(args[2]))
    word = mem[addr : addr - BYTE_SIZE : -1]
    reg[args[0]] = word[0]
    step_pc()

def op_storenb(args: list[int]):
    global mem
    addr = tc_add(reg[args[1]], tc_b16_to_b32(args[2]))
    mem[addr - 0] = reg[args[0]] & 0xff
    step_pc()

def op_loadrb(args: list[int]):
    global reg
    word = mem[reg[args[1]] : reg[args[1]] - BYTE_SIZE : -1]
    reg[args[0]] = word[0]
    step_pc()

def op_storerb(args: list[int]):
    global mem
    mem[reg[args[1]] - 0] = reg[args[0]] & 0xff
    step_pc()

def op_loadns(args: list[int]):
    global reg
    addr = tc_add(reg[args[1]], tc_b16_to_b32(args[2]))
    word = mem[addr : addr - SHORT_SIZE : -1]
    reg[args[0]] = word[1] << 8 | word[0]
    step_pc()

def op_storens(args: list[int]):
    global mem
    addr = tc_add(reg[args[1]], tc_b16_to_b32(args[2]))
    mem[addr - 0] = reg[args[0]] & 0xff
    mem[addr - 1] = (reg[args[0]] >> 8) & 0xff
    step_pc()

def op_loadrs(args: list[int]):
    global reg
    word = mem[reg[args[1]] : reg[args[1]] - SHORT_SIZE : -1]
    reg[args[0]] = word[1] << 8 | word[0]
    step_pc()

def op_storers(args: list[int]):
    global mem
    mem[reg[args[1]] - 0] = reg[args[0]] & 0xff
    mem[reg[args[1]] - 1] = (reg[args[0]] >> 8) & 0xff
    step_pc()

# TODO: make explicit in documentation that loadn* loads from offset in bytes, not words
def op_loadnw(args: list[int]):
    global reg
    addr = tc_add(reg[args[1]], tc_b16_to_b32(args[2]))
    word = mem[addr : addr - WORD_SIZE : -1]
    reg[args[0]] = word[3] << 24 | word[2] << 16 | word[1] << 8 | word[0]
    step_pc()

def op_storenw(args: list[int]):
    global mem
    addr = tc_add(reg[args[1]], tc_b16_to_b32(args[2]))
    mem[addr - 0] = reg[args[0]] & 0xff
    mem[addr - 1] = (reg[args[0]] >> 8) & 0xff
    mem[addr - 2] = (reg[args[0]] >> 16) & 0xff
    mem[addr - 3] = (reg[args[0]] >> 24) & 0xff
    step_pc()

def op_loadrw(args: list[int]):
    global reg
    word = mem[reg[args[1]] : reg[args[1]] - WORD_SIZE : -1]
    reg[args[0]] = word[3] << 24 | word[2] << 16 | word[1] << 8 | word[0]
    step_pc()

def op_storerw(args: list[int]):
    global mem
    mem[reg[args[1]] - 0] = reg[args[0]] & 0xff
    mem[reg[args[1]] - 1] = (reg[args[0]] >> 8) & 0xff
    mem[reg[args[1]] - 2] = (reg[args[0]] >> 16) & 0xff
    mem[reg[args[1]] - 3] = (reg[args[0]] >> 24) & 0xff
    step_pc()

# TODO: how to handle accidentally reading uninitialized memory
def op_unimp(_):
    sys.exit(f"Error: operation {bin_to_opcode[ir >> 24]} unimplemented")

# TODO: where to put this? awkward
instructions = {}
for op in opcode_to_bin.keys():
    func = "op_" + op
    instructions[op] = globals()[func] if func in globals() else op_unimp

# Two's complement helper functions

# TODO: combine into one function tc_int_to_bin(val: int, width: int) -> int
# TODO: report under/overflow
# TODO: completely redo tc over/underflow - look at existing implementations
def tc_int_to_b16(val: int) -> int:
    # two's complement conversion
    if val < 0:
        val = -val
        val = (val ^ 0xffff) + 1
    return val & 0xffff

# TODO: report under/overflow
def tc_int_to_b32(val: int) -> int:
    # two's complement conversion
    if val < 0:
        val = -val
        val = (val ^ 0xffffffff) + 1
    return val & 0xffffffff

def tc_b16_to_b32(val: int) -> int:
    # sign extend negative numbers
    if val & (1 << 15):
        val = val | (0xffff << 16)
    return val

def tc_b16_to_int(val: int) -> int:
    if val & 1 << 15:
        val = (val - 1) ^ 0xffff
        val = -val
    return val

def tc_b32_to_int(val: int) -> int:
    if val & 1 << 31:
        val = (val - 1) ^ 0xffffffff
        val = -val
    return val

def tc_neg(val: int) -> int:
    return (0 - val) & 0xffffffff

def tc_add(val1: int, val2: int) -> int:
    return (val1 + val2) & 0xffffffff

def tc_sub(val1: int, val2: int) -> int:
    return (val1 - val2) & 0xffffffff

def tc_mul(val1: int, val2: int) -> int:
    if (val1 & (1 << 31)):
        val1 = val1 | (0xffffffff << 32)
    if (val2 & (1 << 31)):
        val2 = val2 | (0xffffffff << 32) 
    return (val1 * val2) & 0xffffffff

def tc_div(val1: int, val2: int) -> int:
    sign = ((val1 & (1 << 31))) ^ ((val2 & (1 << 31)))
    if val1 & (1 << 31):
        val1 = tc_neg(val1)
    if val2 & (1 << 31):
        val2 = tc_neg(val2)
    res = val1 // val2
    if sign:
        res = tc_neg(res)
    return res

def tc_mod(val1: int, val2: int) -> int:
    sign2 = ((val2 & (1 << 31)))
    if val1 & (1 << 31):
        val1 = tc_neg(val1)
    if val2 & (1 << 31):
        val2 = tc_neg(val2)
    res = val1 % val2
    if sign2:
        res = tc_neg(res)
    return res

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
        elif c == "n" or c == "f" or c == "l":
            ret.insert(0, ir & 0xffff)
            ir >>= 16
    return ret

# Misc. helper functions

# from https://stackoverflow.com/a/21659588
# TODO: allow for KeyboardInterrupt
def find_getch():
    # Windows getch
    if os.name == 'nt':
        import msvcrt
        return msvcrt.getch

    # Posix getch
    import termios
    import sys
    import tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            _ = tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch

getch = find_getch()

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
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
