import argparse
from dataclasses import dataclass
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
    "halt":   0b00000000, "readi":  0b00000001, "readf":  0b00000010, "readc": 0b00000011,
    "writei": 0b00000100, "writef": 0b00000101, "writec": 0b00000110, "seed":  0b00000111,
    "rand":   0b00001000, "time":   0b00001001, "date":   0b00001010, "nop":   0b00001111,
    # arithmetic instructions
    "addi":   0b00010000, "subi":   0b00010001, "muli":   0b00010010, "divi":  0b00010011,
    "modi":   0b00010100, "negi":   0b00010101, "addf":   0b00010110, "subf":  0b00010111,
    "mulf":   0b00011000, "divf":   0b00011001, "negf":   0b00011010,
    # bitwise instructions
    "and":    0b00100000, "or":     0b00100001, "xor":    0b00100010, "not":   0b00100011,
    "lshl":   0b00100100, "lshr":   0b00100101, "ashl":   0b00100110, "ashr":  0b00100111,
    # jump instructions
    "jump":   0b00110000, "jra":    0b00110001, "jeqz":   0b00110010, "jnez":  0b00110011,
    "jge":    0b00110100, "jle":    0b00110101, "jeq":    0b00110110, "jne":   0b00110111,
    "jgt":    0b00111000, "jlt":    0b00111001, "jsr":    0b00111010,
    # register instructions
    "seti":   0b01000000, "inci":   0b01000001, "setf":   0b01000010, "incf":  0b01000011,
    "copy":   0b01000100,
    # stack insructions
    "pushb":  0b01010000, "popb":   0b01010001,
    "pushs":  0b01010010, "pops":   0b01010011,
    "pushw":  0b01010100, "popw":   0b01010101,
    # load/store stack instructions
    "ldbn":   0b01100000, "ldsn":   0b01100001, "ldwn":   0b01100010,
    "stbn":   0b01100011, "stsn":   0b01100100, "stwn":   0b01100101,
    "ldbr":   0b01100110, "ldsr":   0b01100111, "ldwr":   0b01101000,
    "stbr":   0b01101001, "stsr":   0b01101010, "stwr":   0b01101011,
    # load/store heap instructions
    "ldba":   0b01110000, "ldsa":   0b01110001, "ldwa":   0b01110010,
    "stba":   0b01110011, "stsa":   0b01110100, "stwa":   0b01110101,
    # array instructions
}

# Maps 8-bit binary codes to the opcodes they represent.
bin_to_opcode = {opcode_to_bin[opcode]: opcode for opcode in opcode_to_bin.keys()}

# Maps opcodes to their symbolic argument masks.
opcode_to_argmask = {
    # system instructions
    "halt":   "",    "readi":  "r",   "readf":  "r",   "readc": "r",
    "writei": "r",   "writef": "r",   "writec": "r",   "seed":  "r",
    "rand":   "rrr", "time":   "r",   "date":   "r",   "nop":   "",
    # arithmetic instructions
    "addi":   "rrr", "subi":   "rrr", "muli":   "rrr", "divi":  "rrr",
    "modi":   "rrr", "negi":   "rr",  "addf":   "rrr", "subf":  "rrr",
    "mulf":   "rrr", "divf":   "rrr", "negf":   "rr",
    # bitwise instructions
    "and":    "rrr", "or":     "rrr", "xor":    "rrr", "not":   "rrr",
    "lshl":   "rr",  "lshr":   "rr",  "ashl":   "rr",  "ashr":  "rr",
    # jump instructions
    "jump":   "l",   "jra":    "r",   "jeqz":   "rl",  "jnez":  "rl",
    "jge":    "rrl", "jle":    "rrl", "jeq":    "rrl", "jne":   "rrl",
    "jgt":    "rrl", "jlt":    "rrl", "jsr":    "rl",
    # register instructions
    "seti":   "rn",  "inci":   "rn",  "setf":   "rf",  "incf":  "rf",
    "copy":   "rr",
    # stack insructions
    "pushb":  "rr",  "popb":   "rr",
    "pushs":  "rr",  "pops":   "rr",
    "pushw":  "rr",  "popw":   "rr",
    # load/store stack instructions
    "ldbn":   "rrn", "stbn":   "rrn", "ldbr":   "rr",  "stbr":  "rr",
    "ldsn":   "rrn", "stsn":   "rrn", "ldsr":   "rr",  "stsr":  "rr",
    "ldwn":   "rrn", "stwn":   "rrn", "ldwr":   "rr",  "stwr":  "rr",
    # load/store heap instructions
    "ldba":   "ra",  "ldsa":   "ra",  "ldwa":   "ra",
    "stba":   "ra",  "stsa":   "ra",  "stwa":   "ra"
    # array instructions
}

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

# Maps opcodes to their costs.
opcode_to_cost = {opcode: 2 if opcode_to_bin[opcode] >> 4 == 0b0101 else 1 for opcode in opcode_to_bin.keys()}

# Maps opcodes to their call counts.
opcode_to_calls: dict[str, int] = {opcode: 0 for opcode in opcode_to_bin.keys()}

def main():
    # Process command-line inputs and exit if they are not as expected.
    parser = argparse.ArgumentParser(description=description)
    _ = parser.add_argument("filename", help="input .marv file")
    _ = parser.add_argument("-d", "--debug",   action="store_true", help="enable debug mode")
    _ = parser.add_argument("-c", "--count",   action="store_true", help="count instruction calls")
    args = parser.parse_args()

    inFile = args.filename
    debug = args.debug
    count_calls = args.count

    if not inFile.endswith(".marv") or not os.path.exists(inFile):
        sys.exit(f"Error: invalid file '{inFile}'")

    # TODO: ParsingError exception class instead of prints
    parser = Parser(inFile)
    program = parser.parse()

    if not program.machine_code:
        sys.exit()

    cpu = CPU(program, debug, count_calls)
    cpu.run()

@dataclass
class Program:
    machine_code: list[tuple[int, int, int, int]]
    lines: list[str]
    labels: dict[str, int]
    data_ids: dict[str, tuple[str, int, int]]
    pc_to_abs_line: dict[int, int]
    abs_to_pc_line: dict[int, int]

WORD_SIZE  = 4
SHORT_SIZE = 2
BYTE_SIZE  = 1

STACK_MAX = 65535
HEAP_START = 8192
DATA_START = HEAP_START

class CPU:
    def __init__(self, program: Program, debug: bool, count_calls: bool):
        self.program: Program = program
        self.debug: bool = debug
        self.count_calls: bool = count_calls

        self.reg: list[int] = [0] * 16
        self.mem: list[int] = [0] * 65536
        self.pc: int = 0
        self.ir: int = 0

        self.breakpoints: set[int] = set()
        self.debug_continue_flag: bool = False

        # Load the machine code into memory starting at location 0.
        for i, v in enumerate(self.program.machine_code):
            for b in range(4):
                self.mem[i * 4 + b] = v[b]

        # Load the data section variables into memory at the start of the heap.
        heap_offset = 0
        for data_type, data_val, data_offset in self.program.data_ids.values():
            if data_type in [".byte"]:
                self.mem[DATA_START + data_offset] = data_val & 0xff
                heap_offset += 1
            elif data_type in [".short", ".char"]:
                self.mem[DATA_START + data_offset + 0] = (data_val >> 8) & 0xff
                self.mem[DATA_START + data_offset + 1] = data_val & 0xff
                heap_offset += 2
            elif data_type in [".int", ".float"]:
                self.mem[DATA_START + data_offset + 0] = data_val >> 24
                self.mem[DATA_START + data_offset + 1] = (data_val >> 16) & 0xff
                self.mem[DATA_START + data_offset + 2] = (data_val >> 8) & 0xff
                self.mem[DATA_START + data_offset + 3] = data_val & 0xff
                heap_offset += 4
            elif data_type in [".string"]:
                # TODO: string support
                pass

        # Initialize the frame, stack, and global pointers
        self.reg[reg_to_bin["fp"]] = self.reg[reg_to_bin["sp"]] = STACK_MAX
        self.reg[reg_to_bin["gp"]] = DATA_START + heap_offset

    def run(self):
        while True:
            try:
                self.ir = (
                      self.mem[self.pc]     << 24
                    | self.mem[self.pc + 1] << 16
                    | self.mem[self.pc + 2] << 8
                    | self.mem[self.pc + 3]
                )
            except IndexError:
                sys.exit(f"Error: attempted to execute instruction {self.pc // WORD_SIZE}; halting the machine")

            try:
                self.debug_exec()
            except EOFError:
                sys.exit()

            opcode = bin_to_opcode[self.ir >> 24]
            args = extract_args(self.ir, opcode_to_argmask[opcode])

            # TODO: implement default case
            op_fn = getattr(self, f"op_{opcode}")
            op_fn(args)

            # TODO: actually delay and visualize
            if self.count_calls:
                opcode_to_calls[opcode] += 1

    def debug_exec(self):
        if not self.debug:
            return
        if self.pc // 4 in self.breakpoints:
            pass
        elif self.debug_continue_flag:
            return

        print(f"{self.pc // 4} {self.program.lines[self.program.pc_to_abs_line[self.pc // 4]]}", end="\n\n")

        # Get debug input.
        while cmd := input("(dbg) "):
            if not (tokens := cmd.split()):
                break
            cmd = tokens[0]
            args = [] if len(tokens) == 1 else tokens[1:]
            if cmd == "c" or cmd == "continue":
                self.debug_continue_flag = True
                break
            elif cmd == "s" or cmd == "step":
                self.debug_continue_flag = False
                break
            elif cmd == "b" or cmd == "break":
                if not args:
                    self.breakpoints.add(self.pc // 4)
                try:
                    self.breakpoints.add(int(args[0]))
                except ValueError:
                    print(f"Invalid breakpoint: {args[0]}")
            elif cmd == "d" or cmd == "disable" or cmd == "delete":
                if not args:
                    self.breakpoints = set()
                try:
                    self.breakpoints.remove(int(args[0]))
                except ValueError:
                    print(f"Invalid breakpoint: {args[0]}")
            elif cmd == "l" or cmd == "list":
                print(f"breakpoints: {", ".join([str(bp) for bp in self.breakpoints])}")
            elif cmd == "p" or cmd == "print":
                if not args:
                    self.print_regs()
                    self.print_stack(STACK_MAX, tc_b32_to_int(self.reg[reg_to_bin["sp"]]))
                    continue
                if args[0] not in {"stack", "reg", "mem", "program", "s", "r", "m", "p"}:
                    print(f"Invalid print object: {args[0]}")
                    continue
                if args[0] in {"stack", "s"}:
                    # case 1: print entire stack
                    if len(args) == 1:
                        self.print_stack(STACK_MAX, tc_b32_to_int(self.reg[reg_to_bin["sp"]]))
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
                    self.print_stack(int(args[2]), int(args[1]))
                elif args[0] in {"reg", "r"}:
                    # case 1: print all regs
                    # print reg
                    if len(args) == 1:
                        self.print_regs()
                        continue
                    # case 2: specific reg
                    # print reg <reg>
                    if args[1] not in reg_to_bin.keys():
                        print(f"Invalid register: {args[1]}")
                        continue
                    if len(args) == 2:
                        print(tc_b32_to_int(self.reg[reg_to_bin[args[1]]]))
                        continue
                    # case 3: print reg <reg> <type>
                    if len(args) != 3:
                        print(f"Invalid syntax: {"TODO"}")
                        continue
                    if args[2] not in {"int", "float", "char", "bin", "i", "f", "c", "b"}:
                        print(f"Invalid type: {args[2]}")
                        continue
                    if args[2] in {"int", "i"}:
                        print(tc_b32_to_int(self.reg[reg_to_bin[args[1]]]))
                    elif args[2] in {"float", "f"}:
                        print(fp_f32_to_float(self.reg[reg_to_bin[args[1]]]))
                    elif args[2] in {"char", "c"}:
                        print(chr(self.reg[reg_to_bin[args[1]]]))
                    elif args[2] in {"bin", "b"}:
                        print(format(self.reg[reg_to_bin[args[1]]], "032b"))
                elif args[0] in {"mem", "m"}:
                    # TODO: make more robust
                    if len(args) == 2:
                        print(format(self.mem[int(args[1])], "08b"))
                        continue
                    if len(args) == 3:
                        for i in range(int(args[1]) , int(args[2]) + 1, 4):
                            print(f"{i:04x}: {" ".join(format(byte, "08b") for byte in self.mem[i : i + 4])}")
                        continue
                elif args[0] in {"program", "p"}:
                    self.print_program()

            elif cmd == "q" or cmd == "quit":
                self.debug = False
                return
            elif cmd == "h" or cmd == "help":
                # TODO: de-indent this some other way
                help_str = """
List of commands:

    help, h -- print this command
        Usage: (h | help)

    quit, q -- exit the debugger
        Usage: (q | quit)

    continue, c -- continue to next breakpoint
        Usage: (c | continue)

    step, s -- step to next instruction
        Usage: (s | step)

    break, b -- set a breakpoint
        Usage: (b | break) <instruction>

    delete, d -- delete a breakpoint
        Usage: (d | delete) <instruction>

    list, l -- list breakpoints
        Usage: (l | list)

    print, p -- print register or stack information
        Usage: (p | print) [(s | stack) [<start> <stop> | <address>]] |
               (p | print) [(r | reg) [(<reg>) [<type>]]] |
               (p | print) [(p | program)]
                """
                print(help_str)
            else:
                print(f"Invalid command: {cmd}")
        print()

    def print_program(self):
        print()
        for i, line in enumerate(self.program.lines):
            if (pc := self.program.abs_to_pc_line.get(i, -1)) != -1:
                # TODO: handle spacing more robustly
                # TODO: hex memory address instead of instruction number?
                if pc == self.pc // 4:
                    print(f">{pc :>3} {line :<30}", end="")
                else:
                    print(f"{pc :>4} {line :<30}", end="")
                print(" ".join([format(b, "08b") for b in self.mem[i:i+4]]))
            else:
                print(line)
        print()


    def print_regs(self):
        print(f"r0:  [{self.format_reg(0)  :>10}] r1:  [{self.format_reg(1)  :>10}] r2:  [{self.format_reg(2)  :>10}] r3:  [{self.format_reg(3)  :>10}]")
        print(f"r4:  [{self.format_reg(4)  :>10}] r5:  [{self.format_reg(5)  :>10}] r6:  [{self.format_reg(6)  :>10}] r7:  [{self.format_reg(7)  :>10}]")
        print(f"r8:  [{self.format_reg(8)  :>10}] r9:  [{self.format_reg(9)  :>10}] r10: [{self.format_reg(10) :>10}] ra:  [{self.format_reg(11) :>10}]")
        print(f"rv:  [{self.format_reg(12) :>10}] fp:  [{self.format_reg(13) :>10}] sp:  [{self.format_reg(14) :>10}] gp:  [{self.format_reg(15) :>10}]")
        print()

    def print_stack(self, hi: int, lo: int):
        # TODO: bounds checking
        for i in range(hi, lo, -4):
            sym = ["*" if i == self.reg[reg_to_bin["fp"]] else " ", ">" if i == self.reg[reg_to_bin["sp"]] else " "]
            print(f"{''.join(sym)} {i:04x}: {' '.join(format(byte, '08b') for byte in self.mem[i - 3 : i + 1])}")
        print()

    def format_reg(self, regbin: int) -> str:
        return f"{tc_b32_to_int(self.reg[regbin])}"

    def step_pc(self):
        self.pc += WORD_SIZE

    # System Instructions

    def op_halt(self, _):
        if self.count_calls:
            print_opcode_cost()
        sys.exit()

    def op_readi(self, args: list[int]):
        while True:
            try:
                x = int(input())
                if (valid_int(x)):
                    break
                raise ValueError
            except ValueError:
                print("Illegal input: input must be a number must be in [-32768, 32767]")
        self.reg[args[0]] = tc_int_to_b32(x)
        self.step_pc()


    def op_readf(self, args: list[int]):
        while True:
            try:
                x = float(input())
                break
            except ValueError:
                # TODO: better error message
                print("Illegal input: input must be a number")
        self.reg[args[0]] = fp_float_to_f32(x)
        self.step_pc()

    def op_readc(self, args: list[int]):
        byte_list = list(str(getch()).encode("utf-16-be"))
        self.reg[args[0]] = byte_list[0] << 8 | byte_list[1]
        self.step_pc()

    def op_writei(self, args: list[int]):
        print(tc_b32_to_int(self.reg[args[0]]))
        self.step_pc()

    def op_writef(self, args: list[int]):
        print(fp_f32_to_float(self.reg[args[0]]))
        self.step_pc()

    def op_writec(self, args: list[int]):
        # TODO: how to handle value outside of range? clamp?
        print(self.reg[args[0]].to_bytes(2, 'big').decode("utf-16-be"))
        self.step_pc()

    def op_seed(self, args: list[int]):
        random.seed((self.reg[args[0]]))
        self.step_pc()

    def op_rand(self, args: list[int]):
        lo = tc_b32_to_int(self.reg[args[0]])
        hi = tc_b32_to_int(self.reg[args[1]])
        self.reg[args[2]] = tc_int_to_b32(random.randint(lo, hi))
        self.step_pc()

    def op_time(self, args: list[int]):
        now = datetime.datetime.now()
        self.reg[args[0]] = (now.hour * 3600 + now.minute * 60 + now.second) * 1000 + now.microsecond // 1000
        self.step_pc()


    def op_date(self, args: list[int]):
        today = datetime.date.today()
        # [31:13]: year  - 19 bits - 524287 values
        # [12:9]:  month - 4  bits - 16     values
        # [8:0]:   day   - 9  bits - 512    values
        self.reg[args[0]] = today.year << 13 | today.month << 9 | today.day
        self.step_pc()

    def op_nop(self, _):
        self.step_pc()

# Arithmetic instructions

    def op_negi(self, args: list[int]):
        self.reg[args[0]] = tc_neg(self.reg[args[1]])
        self.step_pc()

    def op_addi(self, args: list[int]):
        self.reg[args[0]] = tc_add(self.reg[args[1]], self.reg[args[2]])
        self.step_pc()

    def op_subi(self, args: list[int]):
        self.reg[args[0]] = tc_sub(self.reg[args[1]], self.reg[args[2]])
        self.step_pc()

    def op_muli(self, args: list[int]):
        self.reg[args[0]] = tc_mul(self.reg[args[1]], self.reg[args[2]])
        self.step_pc()

    def op_divi(self, args: list[int]):
        self.reg[args[0]] = tc_div(self.reg[args[1]], self.reg[args[2]])
        self.step_pc()

    def op_modi(self, args: list[int]):
        self.reg[args[0]] = tc_mod(self.reg[args[1]], self.reg[args[2]])
        self.step_pc()

    def op_negf(self, args: list[int]):
        self.reg[args[0]] = self.reg[args[1]] ^ (1 << 31)
        self.step_pc()

    def op_addf(self, args: list[int]):
        self.reg[args[0]] = fp_float_to_f32(fp_f32_to_float(self.reg[args[1]]) + fp_f32_to_float(self.reg[args[2]]))
        self.step_pc()

    def op_subf(self, args: list[int]):
        self.reg[args[0]] = fp_float_to_f32(fp_f32_to_float(self.reg[args[1]]) - fp_f32_to_float(self.reg[args[2]]))
        self.step_pc()

    def op_mulf(self, args: list[int]):
        self.reg[args[0]] = fp_float_to_f32(fp_f32_to_float(self.reg[args[1]]) * fp_f32_to_float(self.reg[args[2]]))
        self.step_pc()

    def op_divf(self, args: list[int]):
        self.reg[args[0]] = fp_float_to_f32(fp_f32_to_float(self.reg[args[1]]) / fp_f32_to_float(self.reg[args[2]]))
        self.step_pc()

    # Bitwise instructions

    def op_and(self, args: list[int]):
        self.reg[args[0]] = self.reg[args[1]] & self.reg[args[2]]
        self.step_pc()

    def op_or(self, args: list[int]):
        self.reg[args[0]] = self.reg[args[1]] | self.reg[args[2]]
        self.step_pc()

    def op_xor(self, args: list[int]):
        self.reg[args[0]] = self.reg[args[1]] ^ self.reg[args[2]]
        self.step_pc()

    def op_not(self, args: list[int]):
        self.reg[args[0]] = ~self.reg[args[1]]
        self.step_pc()

    def op_lshl(self, args: list[int]):
        self.reg[args[0]] = (self.reg[args[1]] << self.reg[args[2]]) & 0xffffffff
        self.step_pc()

    def op_lshr(self, args: list[int]):
        self.reg[args[0]] = (self.reg[args[1]] >> self.reg[args[2]]) & 0xffffffff
        self.step_pc()

    # TODO: verify this
    def op_ashl(self, args: list[int]):
        sign = self.reg[args[1]] & (1 << 31)
        temp = ((self.reg[args[1]] ^ sign) << self.reg[args[2]]) & 0xffffffff
        self.reg[args[0]] = 0 if temp == 0 else temp | sign
        self.step_pc()

    # TODO: verify this
    def op_ashr(self, args: list[int]):
        num_shifts = tc_b32_to_int(self.reg[args[2]])
        sign = self.reg[args[1]] & (1 << 31)
        sign_extend = int("1" * num_shifts, 2) if sign else 0
        temp = ((self.reg[args[1]] ^ sign) >> self.reg[args[2]]) | (sign_extend << (31 - num_shifts))
        self.reg[args[0]] = 0 if temp == 0 else temp | sign
        self.step_pc()

    # Jump Instructions

    def op_jump(self, args: list[int]):
        self.pc = args[0] * WORD_SIZE

    def op_jra(self, args: list[int]):
        self.pc = self.reg[args[0]]

    def op_jeqz(self, args: list[int]):
        self.pc = args[1] * WORD_SIZE if tc_b32_to_int(self.reg[args[0]]) == 0 else self.pc + WORD_SIZE

    def op_jnez(self, args: list[int]):
        self.pc = args[1] * WORD_SIZE if tc_b32_to_int(self.reg[args[0]]) != 0 else self.pc + WORD_SIZE

    def op_jge(self, args: list[int]):
        self.pc = (
            args[2] * WORD_SIZE
            if tc_b32_to_int(self.reg[args[0]]) >= tc_b32_to_int(self.reg[args[1]])
            else self.pc + WORD_SIZE
        )

    def op_jle(self, args: list[int]):
        self.pc = (
            args[2] * WORD_SIZE
            if tc_b32_to_int(self.reg[args[0]]) <= tc_b32_to_int(self.reg[args[1]])
            else self.pc + WORD_SIZE
        )

    def op_jeq(self, args: list[int]):
        self.pc = (
            args[2] * WORD_SIZE
            if tc_b32_to_int(self.reg[args[0]]) == tc_b32_to_int(self.reg[args[1]])
            else self.pc + WORD_SIZE
        )

    def op_jne(self, args: list[int]):
        self.pc = (
            args[2] * WORD_SIZE
            if tc_b32_to_int(self.reg[args[0]]) != tc_b32_to_int(self.reg[args[1]])
            else self.pc + WORD_SIZE
        )

    def op_jgt(self, args: list[int]):
        self.pc = (
            args[2] * WORD_SIZE
            if tc_b32_to_int(self.reg[args[0]]) > tc_b32_to_int(self.reg[args[1]])
            else self.pc + WORD_SIZE
        )

    def op_jlt(self, args: list[int]):
        self.pc = (
            args[2] * WORD_SIZE
            if tc_b32_to_int(self.reg[args[0]]) < tc_b32_to_int(self.reg[args[1]])
            else self.pc + WORD_SIZE
        )

    def op_jsr(self, args: list[int]):
        # I think this is guaranteed to never be over integer limit
        self.reg[args[0]] = self.pc + WORD_SIZE
        self.pc = args[1] * WORD_SIZE

    # Register instructions

    def op_seti(self, args: list[int]):
        self.reg[args[0]] = tc_b16_to_b32(args[1])
        self.step_pc()

    def op_inci(self, args: list[int]):
        self.reg[args[0]] = tc_add(self.reg[args[0]], tc_b16_to_b32(args[1]))
        self.step_pc()

    def op_setf(self, args: list[int]):
        self.reg[args[0]] = fp_f16_to_f32(args[1])
        self.step_pc()

    def op_incf(self, args: list[int]):
        self.reg[args[0]] = fp_float_to_f32(fp_f32_to_float(self.reg[args[0]]) + fp_f32_to_float(self.reg[args[1]]))
        self.step_pc()

    def op_copy(self, args: list[int]):
        self.reg[args[0]] = self.reg[args[1]]
        self.step_pc()

    # Stack instructions

    # TODO: clean up generic code for single bytes etc

    def op_pushb(self, args: list[int]):
        if self.reg[reg_to_bin["sp"]] <= self.reg[reg_to_bin["gp"]]:
            sys.exit(f"Error: stack overflow attempting to execute instruction {self.pc // WORD_SIZE}; halting the machine")
        self.mem[self.reg[args[1]] - 0] = self.reg[args[0]] & 0xff
        self.reg[args[1]] = tc_sub(self.reg[args[1]], BYTE_SIZE)
        self.step_pc()

    def op_popb(self, args: list[int]):
        self.reg[args[1]] = tc_add(self.reg[args[1]], BYTE_SIZE)
        word = self.mem[self.reg[args[1]] : self.reg[args[1]] - BYTE_SIZE : -1]
        self.reg[args[0]] = word[0]
        self.step_pc()

    def op_pushs(self, args: list[int]):
        if self.reg[reg_to_bin["sp"]] <= self.reg[reg_to_bin["gp"]]:
            sys.exit(f"Error: stack overflow attempting to execute instruction {self.pc // WORD_SIZE}; halting the machine")
        self.mem[self.reg[args[1]] - 0] = self.reg[args[0]] & 0xff
        self.mem[self.reg[args[1]] - 1] = (self.reg[args[0]] >> 8) & 0xff
        self.reg[args[1]] = tc_sub(self.reg[args[1]], SHORT_SIZE)
        self.step_pc()

    def op_pops(self, args: list[int]):
        self.reg[args[1]] = tc_add(self.reg[args[1]], SHORT_SIZE)
        word = self.mem[self.reg[args[1]] : self.reg[args[1]] - SHORT_SIZE : -1]
        self.reg[args[0]] = word[1] << 8 | word[0]
        self.step_pc()

    def op_pushw(self, args: list[int]):
        if self.reg[reg_to_bin["sp"]] <= self.reg[reg_to_bin["gp"]]:
            sys.exit(f"Error: stack overflow attempting to execute instruction {self.pc // WORD_SIZE}; halting the machine")
        self.mem[self.reg[args[1]] - 0] = self.reg[args[0]] & 0xff
        self.mem[self.reg[args[1]] - 1] = (self.reg[args[0]] >> 8) & 0xff
        self.mem[self.reg[args[1]] - 2] = (self.reg[args[0]] >> 16) & 0xff
        self.mem[self.reg[args[1]] - 3] = (self.reg[args[0]] >> 24) & 0xff
        self.reg[args[1]] = tc_sub(self.reg[args[1]], WORD_SIZE)
        self.step_pc()

    def op_popw(self, args: list[int]):
        global reg
        self.reg[args[1]] = tc_add(self.reg[args[1]], WORD_SIZE)
        word = self.mem[self.reg[args[1]] : self.reg[args[1]] - WORD_SIZE : -1]
        self.reg[args[0]] = word[3] << 24 | word[2] << 16 | word[1] << 8 | word[0]
        self.step_pc()

    # Load/store stack instructions

    # TODO: add 0xff mask to left shifted bytes?

    # TODO: change from addr : addr - ? to addr - ? + 1 : addr + 1 ?
    # - to avoid case where addr - ? is -1 and last byte gets cut off
    def op_ldbn(self, args: list[int]):
        addr = tc_sub(self.reg[args[1]], tc_b16_to_b32(args[2]))
        word = self.mem[addr : addr - BYTE_SIZE : -1]
        self.reg[args[0]] = word[0]
        self.step_pc()

    def op_ldsn(self, args: list[int]):
        addr = tc_sub(self.reg[args[1]], tc_b16_to_b32(args[2]))
        word = self.mem[addr : addr - SHORT_SIZE : -1]
        self.reg[args[0]] = word[1] << 8 | word[0]
        self.step_pc()

    def op_ldwn(self, args: list[int]):
        addr = tc_sub(self.reg[args[1]], tc_b16_to_b32(args[2]))
        word = self.mem[addr : addr - WORD_SIZE : -1]
        self.reg[args[0]] = word[3] << 24 | word[2] << 16 | word[1] << 8 | word[0]
        self.step_pc()

    def op_stbn(self, args: list[int]):
        addr = tc_sub(self.reg[args[1]], tc_b16_to_b32(args[2]))
        self.mem[addr - 0] = self.reg[args[0]] & 0xff
        self.step_pc()

    def op_stsn(self, args: list[int]):
        addr = tc_sub(self.reg[args[1]], tc_b16_to_b32(args[2]))
        self.mem[addr - 0] = self.reg[args[0]] & 0xff
        self.mem[addr - 1] = (self.reg[args[0]] >> 8) & 0xff
        self.step_pc()

    def op_stwn(self, args: list[int]):
        addr = tc_sub(self.reg[args[1]], tc_b16_to_b32(args[2]))
        self.mem[addr - 0] = self.reg[args[0]] & 0xff
        self.mem[addr - 1] = (self.reg[args[0]] >> 8) & 0xff
        self.mem[addr - 2] = (self.reg[args[0]] >> 16) & 0xff
        self.mem[addr - 3] = (self.reg[args[0]] >> 24) & 0xff
        self.step_pc()

    def op_ldbr(self, args: list[int]):
        word = self.mem[self.reg[args[1]] : self.reg[args[1]] - BYTE_SIZE : -1]
        self.reg[args[0]] = word[0]
        self.step_pc()

    def op_ldsr(self, args: list[int]):
        word = self.mem[self.reg[args[1]] : self.reg[args[1]] - SHORT_SIZE : -1]
        self.reg[args[0]] = word[1] << 8 | word[0]
        self.step_pc()

    def op_ldwr(self, args: list[int]):
        word = self.mem[self.reg[args[1]] : self.reg[args[1]] - WORD_SIZE : -1]
        self.reg[args[0]] = word[3] << 24 | word[2] << 16 | word[1] << 8 | word[0]
        self.step_pc()

    def op_stbr(self, args: list[int]):
        self.mem[self.reg[args[1]] - 0] = self.reg[args[0]] & 0xff
        self.step_pc()

    def op_stsr(self, args: list[int]):
        self.mem[self.reg[args[1]] - 0] = self.reg[args[0]] & 0xff
        self.mem[self.reg[args[1]] - 1] = (self.reg[args[0]] >> 8) & 0xff
        self.step_pc()

    def op_stwr(self, args: list[int]):
        self.mem[self.reg[args[1]] - 0] = self.reg[args[0]] & 0xff
        self.mem[self.reg[args[1]] - 1] = (self.reg[args[0]] >> 8) & 0xff
        self.mem[self.reg[args[1]] - 2] = (self.reg[args[0]] >> 16) & 0xff
        self.mem[self.reg[args[1]] - 3] = (self.reg[args[0]] >> 24) & 0xff
        self.step_pc()

    # Load/store heap instruction

    def op_ldba(self, args: list[int]):
        byte = self.mem[args[1]]
        self.reg[args[0]] = byte
        self.step_pc()

    def op_ldsa(self, args: list[int]):
        short = self.mem[args[1] : args[1] + SHORT_SIZE]
        self.reg[args[0]] = short[0] << 8 | short[1]
        self.step_pc()

    def op_ldwa(self, args:list[int]):
        word = self.mem[args[1] : args[1] + WORD_SIZE]
        self.reg[args[0]] = word[0] << 24 | word[1] << 16 | word[2] << 8 | word[3]
        self.step_pc()

    def op_stba(self, args: list[int]):
        self.mem[args[1]] = args[0] & 0xff
        self.step_pc()

    def op_stsa(self, args: list[int]):
        addr = args[1]
        short = self.reg[args[0]]
        self.mem[addr]     = (short >> 8) & 0xff
        self.mem[addr + 1] = short & 0xff

    def op_stwa(self, args: list[int]):
        addr = args[1]
        word = self.reg[args[0]]
        self.mem[addr]     = (word >> 24) & 0xff
        self.mem[addr + 1] = (word >> 16) & 0xff
        self.mem[addr + 2] = (word >> 8) & 0xff
        self.mem[addr + 3] = word & 0xff

    # TODO: how to handle accidentally reading uninitialized memory?
    def op_unimp(self, _):
        sys.exit(f"Error: operation {bin_to_opcode[self.ir >> 24]} unimplemented")


class Parser:
    def __init__(self, inFile: str):
        self.inFile: str = inFile

        self.lines: list[str] = []
        self.pc_to_abs_line: dict[int, int] = {}
        self.abs_to_pc_line: dict[int, int] = {}

        self.machine_code: list[tuple[int, int, int, int]] = []
        self.labels: dict[str, int] = {}
        self.data_ids: dict[str, tuple[str, int, int]] = {}

    def parse(self) -> Program:
        tokens = self._tokenize()
        self._assemble(tokens)
        return Program(self.machine_code, self.lines, self.labels, self.data_ids, self.pc_to_abs_line, self.abs_to_pc_line)

    def _tokenize(self) -> list[tuple[str, *tuple[str, ...]]]:
        tuples: list[tuple[str, *tuple[str, ...]]] = []

        data_start = 0
        data_end = 0
        data_found = False

        text_start = 0
        text_end = 0
        text_found = False

        instruction_number = 0

        with open(self.inFile, "r") as fh:
            lines: list[str] = fh.readlines()

        # Scan for label validation and section bounds.
        for i, line in enumerate(lines):
            line = line.strip().lower()
            lineno = i + 1

            # Skip empty lines and comments.
            if not line or line.startswith("#"):
                continue

            # Remove inlined comment if any.
            if "#" in line:
                line = line[:line.find("#")].strip()

            # Find start and end of text and data sections.
            if line.startswith("."):
                toks = line.split()
                section = toks[0]
                if section == ".data":
                    if text_found:
                        text_end = i
                    if data_found:
                        sys.exit(f"Error {self.inFile}@{lineno}: duplicate data section {line}")
                    data_found = True
                    data_start = i + 1

                elif section == ".text":
                    if data_found:
                        data_end = i
                    if text_found:
                        sys.exit(f"Error {self.inFile}@{lineno}: duplicate text section {line}")
                    text_found = True
                    text_start = i + 1
                elif section not in [".byte", ".short", ".int", ".float", ".char", ".string"]:
                    # TODO: handle types here?
                    sys.exit(f"Error {self.inFile}@{lineno}: invalid section {line}")

            # Find labels.
            if ":" in line:
                if not line.endswith(":"):
                    sys.exit(f"Error {self.inFile}@{lineno}: invalid label {line}")
                label = line[:-1]
                if label in self.labels.keys():
                    sys.exit(f"Error: {self.inFile}@{lineno}: duplicate label {label}")
                self.labels[label] = instruction_number - 1
                continue

            instruction_number += 1

        if not text_start:
            # TODO: is this descriptive enough?
            sys.exit(f"Error {self.inFile}: missing text section")

        if not text_end:
            # TODO: -1 necessary?
            text_end = len(lines)

        if data_start and not data_end:
            # TODO: -1 necessary?
            data_end = len(lines)

        data_offset = 0

        # Validate data section.
        for line in lines[data_start:data_end]:
            line = line.strip().lower()
            lineno = data_start + 1

            # Skip empty lines, comments, and self.labels.
            if not line or line.startswith("#") or line.endswith(":"):
                continue

            # Remove inlined comment if any.
            if "#" in line:
                line = line[:line.find("#")].strip()

            toks = line.split()

            if len(toks) != 4:
                sys.exit(f"Error {self.inFile}@{lineno}: invalid number of tokens '{toks[0]}'")

            if toks[0] not in [".byte", ".short", ".int", ".float", ".char", ".string"]:
                sys.exit(f"Error {self.inFile}@{lineno}: invalid type directive '{toks[0]}'")

            if toks[1] in self.data_ids.keys():
                sys.exit(f"Error {self.inFile}@{lineno}: duplicate variable indentifier '{toks[1]}'")

            if toks[2] != "=":
                sys.exit(f"Error {self.inFile}@{lineno}: invalid assignment operator '{toks[2]}'")

            if toks[0] in [".byte", ".short", ".int"]:
                if is_int(toks[3]):
                    self.data_ids[toks[1]] = (toks[0], tc_int_to_b32(int(toks[3])), data_offset)
                    # TODO: change this for varying width types
                    data_offset += 4
            elif toks[0] == ".float":
                if is_float(toks[3]):
                    self.data_ids[toks[1]] = (toks[0], fp_float_to_f32(float(toks[3])), data_offset)
                    data_offset += 4
            elif toks[0] == ".char":
                # TODO: require quotes?
                byte_list = list(toks[3].encode("utf-16-be"))
                ch = byte_list[0] << 8 | byte_list[1]
                if len(byte_list) != 2:
                    sys.exit(f"Error {self.inFile}@{lineno}: invalid unicode character '{toks[3]}'")
                self.data_ids[toks[1]] = (toks[0], ch, data_offset)
                data_offset += 2
            elif toks[0] == ".string":
                # TODO: implement strings
                sys.exit(f"Error {self.inFile}@{lineno}: unimplemented type '{toks[0]}'")

        pc_line = 0
        abs_line = 0

        halt_found = False

        # Validate text section.
        for line in lines[text_start:text_end]:
            line = line.strip().lower()
            lineno = text_start + 1

            # Skip empty lines, comments, and self.labels.
            if not line or line.startswith("#"):
                continue

            if line.endswith(":"):
                self.lines.append(line)
                abs_line += 1
                continue

            # Remove inlined comment if any.
            if "#" in line:
                line = line[:line.find("#")].strip()

            toks = line.split()

            # Exit with error if the instruction is invalid.
            if toks[0] not in opcode_to_bin:
                sys.exit(f"Error {self.inFile}@{lineno}: invalid instruction '{toks[0]}'")

            # Validate the instruction arguments.
            opcode, args = toks[0], toks[1:]
            if len(args) != len(opcode_to_argmask[opcode]):
                argmask = opcode_to_argmask[opcode]
                # TODO: argmask to str?
                sys.exit(f"Error {self.inFile}@{lineno}: opcode expects {len(argmask)} arguments: '{argmask}'")
            for i, c in enumerate(opcode_to_argmask[opcode]):
                # TODO: do we need a 'c' case for chars? how to handle strings?
                if c == "r":
                    if not valid_reg(args[i]):
                        sys.exit(f"Error {self.inFile}@{lineno}: invalid register '{args[i]}'")
                elif c == "n":
                    if not is_int(args[i]):
                        sys.exit(f"Error {self.inFile}@{lineno}: invalid number '{args[i]}'")
                elif c == "f":
                    if not is_float(args[i]):
                        sys.exit(f"Error {self.inFile}@{lineno}: invalid number '{args[i]}'")
                elif c == "l":
                    if not self.valid_label(args[i]):
                        sys.exit(f"Error {self.inFile}@{lineno}: invalid label '{args[i]}'")
                    if self.labels[args[i]] >= instruction_number:
                        sys.exit(f"Error {self.inFile}@{lineno}: missing instruction after label '{args[i]}'")
                elif c == "a":
                    if not self.valid_variable(args[i]) and not self.valid_address(args[i]):
                        sys.exit(f"Error {self.inFile}@{lineno}: invalid address '{args[i]}'")

            # Append valid tuple.
            tuples.append((opcode, *args))
            self.lines.append(line)
            self.pc_to_abs_line[pc_line] = abs_line
            self.abs_to_pc_line[abs_line] = pc_line
            pc_line += 1
            abs_line += 1

            if opcode == "halt":
                halt_found = True

        if not halt_found:
            sys.exit(f"Error: {self.inFile}: halt instruction not found")

        return tuples

    def _assemble(self, tokens: list[tuple[str, *tuple[str, ...]]]):
        for _, t in enumerate(tokens):
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
                    if self.valid_variable(args[i]):
                        val = tc_int_to_b16(self.data_ids[args[i]][2])
                    else:
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
                    val = tc_int_to_b16(self.labels[args[i]])
                    byte_list[curr_byte] = val & 0xff
                    byte_list[curr_byte - 1] = val >> 8
                    curr_byte -= 2
                elif c == "a":
                    if self.valid_variable(args[i]):
                        val = tc_int_to_b16(DATA_START + self.data_ids[args[i]][2])
                        byte_list[curr_byte] = val & 0xff
                        byte_list[curr_byte - 1] = val >> 8
                        curr_byte -= 2
                    else:
                        val = tc_int_to_b16(int(args[i]))
                        byte_list[curr_byte] = val & 0xff
                        byte_list[curr_byte - 1] = val >> 8
                        curr_byte -= 2

                i -= 1

            code = (byte_list[0], byte_list[1], byte_list[2], byte_list[3])
            self.machine_code.append(code)

    def valid_label(self, label: str) -> bool:
        return label in self.labels.keys()

    def valid_variable(self, variable: str) -> bool:
        return variable in self.data_ids.keys()

    def valid_address(self, address: str) -> bool:
        try:
            addr = int(address)
            return 0 <= addr and addr <= STACK_MAX
        except ValueError:
            return False

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
        elif c == "n" or c == "f" or c == "l" or c == "a":
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

# determine getch function
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

# TODO: reimplement verbose output
# def print_verbose_output():
#     for s in verbose_output:
#         print(s)
#     print()

def print_opcode_cost():
    for k, v in opcode_to_calls.items():
        if v > 0:
            print(f"{k: <6}: {v * opcode_to_cost[k]}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
