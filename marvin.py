#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import datetime
import os
import random
import shlex
import struct
import sys

description = """
This program serves as an emulator for a register-based machine called Marvin (named 
after the paranoid android character, Marvin, from The Hitchhiker's Guide to the 
Galaxy by Douglas Adams). The program accepts a .marv file as input, assembles and 
simulates the instructions within, and prints any output to stdout.
"""

# Maps opcodes to their binary 8-bit codes.
opcode_to_bin = {
    # system instructions
    "halt":   0b00000000, 
    "readi":  0b00000001, "readf":  0b00000010, "readc":  0b00000011,
    "writei": 0b00000100, "writef": 0b00000101, "writec": 0b00000110, "writes": 0b00000111,
    "seed":   0b00001000, "rand":   0b00001001, "time":   0b00001010, "date":   0b00001011,
    "nop":    0b00001100,
    # arithmetic instructions
    "addi":   0b00010000, "subi":   0b00010001, "muli":   0b00010010, "divi":  0b00010011,
    "modi":   0b00010100, "negi":   0b00010101, "addf":   0b00010110, "subf":  0b00010111,
    "mulf":   0b00011000, "divf":   0b00011001, "negf":   0b00011010,
    # bitwise instructions
    "and":    0b00100000, "or":     0b00100001, "xor":    0b00100010, "not":   0b00100011,
    "lshl":   0b00100100, "lshr":   0b00100101, "ashl":   0b00100110, "ashr":  0b00100111,
    # jump instructions
    "j":      0b00110000, "jr":     0b00110001, "jeqz":   0b00110010, "jnez":  0b00110011,
    "jge":    0b00110100, "jle":    0b00110101, "jeq":    0b00110110, "jne":   0b00110111,
    "jgt":    0b00111000, "jlt":    0b00111001, "jsr":    0b00111010,
    # register instructions
    "seti":   0b01000000, "inci":   0b01000001,
    "copy":   0b01000010,
    # stack insructions
    "pushb":  0b01010000, "popb":   0b01010001,
    "pushs":  0b01010010, "pops":   0b01010011,
    "pushw":  0b01010100, "popw":   0b01010101,
    # load/store instructions
    "lda":    0b01100000,
    "ldb":    0b01100001, "lds":    0b01100010, "ldw":    0b01100011,
    "stb":    0b01100100, "sts":    0b01100101, "stw":    0b01100110,
    # array instructions
    "anewb":  0b01110000, "anews":  0b01110001, "aneww":  0b01110010,
    "aldb":   0b01110011, "alds":   0b01110100, "aldw":   0b01110101,
    "astb":   0b01110110, "asts":   0b01110111, "astw":   0b01111000,
    "alen":   0b01111001,
    # conversion instructions
    "i2f":    0b10000000,
    "i2c":    0b10000001,
    "f2i":    0b10000010
}

# Maps 8-bit binary codes to the opcodes they represent.
bin_to_opcode = {opcode_to_bin[opcode]: opcode for opcode in opcode_to_bin.keys()}

# Maps opcodes to their symbolic argument masks.
opcode_to_argmask = {
    # system instructions
    "halt":   "",
    "readi":  "r",   "readf":  "r",   "readc": "r",
    "writei": "r",   "writef": "r",   "writec": "r",   "writes": "r",
    "seed":   "r",
    "rand":   "rrr", "time":   "r",   "date":   "r",   "nop":    "",
    # arithmetic instructions
    "addi":   "rrr", "subi":   "rrr", "muli":   "rrr", "divi":   "rrr",
    "modi":   "rrr", "negi":   "rr",  "addf":   "rrr", "subf":   "rrr",
    "mulf":   "rrr", "divf":   "rrr", "negf":   "rr",
    # bitwise instructions
    "and":    "rrr", "or":     "rrr", "xor":    "rrr", "not":    "rr",
    "lshl":   "rr",  "lshr":   "rr",  "ashl":   "rr",  "ashr":   "rr",
    # jump instructions
    "j":   "l",   "jr":    "r",   "jeqz":   "rl",  "jnez":   "rl",
    "jge":    "rrl", "jle":    "rrl", "jeq":    "rrl", "jne":    "rrl",
    "jgt":    "rrl", "jlt":    "rrl", "jsr":    "rl",
    # register instructions
    "seti":   "rn",  "inci":   "rn",
    "copy":   "rr",
    # stack insructions
    "pushb":  "rr",  "popb":   "rr",
    "pushs":  "rr",  "pops":   "rr",
    "pushw":  "rr",  "popw":   "rr",
    # load/store instructions
    "lda":    "ra",
    "ldb":    "rrn", "stb":    "rrn",
    "lds":    "rrn", "sts":    "rrn",
    "ldw":    "rrn", "stw":    "rrn",
    # array instructions
    "anewb":  "rr",  "anews":  "rr",  "aneww":  "rr",
    "aldb":   "rrr", "alds":   "rrr", "aldw":   "rrr",
    "astb":   "rrr", "asts":   "rrr", "astw":   "rrr",
    "alen":   "rr",
    # conversion instructions
    "i2f":    "rr",
    "i2c":    "rr",
    "f2i":    "rr"
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
    # TODO: consider using nargs="*" or argparse.REMAINDER
    _ = parser.add_argument("-a", "--args",    nargs="+",           help="pass cli arguments")
    args = parser.parse_args()

    inFile = args.filename
    debug = args.debug
    count_calls = args.count
    cli_args: list[str] = args.args

    if not inFile.endswith(".marv") or not os.path.exists(inFile):
        sys.exit(f"Error: invalid file '{inFile}'")

    # TODO: ParsingError exception class instead of prints
    parser = Parser(inFile)
    program = parser.parse()

    if not program.machine_code:
        sys.exit()

    cpu = CPU(program, debug, count_calls, cli_args)
    cpu.run()

@dataclass
class Program:
    machine_code: list[tuple[int, int, int, int]]
    lines: list[str]
    labels: dict[str, int]
    data_ids: dict[str, tuple[str, list[int], int]]
    pc_to_abs_line: dict[int, int]
    abs_to_pc_line: dict[int, int]

WORD_SIZE  = 4
SHORT_SIZE = 2
BYTE_SIZE  = 1

STACK_START = 65532
HEAP_START = 8192
DATA_START = HEAP_START

class CPU:
    def __init__(self, program: Program, debug: bool, count_calls: bool, cli_args: list[str]):
        self.program: Program = program
        self.debug: bool = debug
        self.count_calls: bool = count_calls
        self.cli_args = cli_args

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
        for data_type, data_vals, data_offset in self.program.data_ids.values():
            if data_type in [".byte"]:
                data_val = data_vals[0]
                self.store_byte(DATA_START + data_offset, data_val)
                heap_offset += BYTE_SIZE
            elif data_type in [".short", ".char"]:
                data_val = data_vals[0]
                self.store_short(DATA_START + data_offset, data_val)
                heap_offset += SHORT_SIZE
            elif data_type in [".int", ".float"]:
                data_val = data_vals[0]
                self.store_word(DATA_START + data_offset, data_val)
                heap_offset += WORD_SIZE
            elif data_type in [".string"]:
                utf16_chars = data_vals
                arglen = len(utf16_chars)
                self.store_short(DATA_START + data_offset, arglen)
                for i in range(arglen):
                    addr = DATA_START + data_offset + SHORT_SIZE + i * SHORT_SIZE
                    self.store_short(addr, utf16_chars[i])
                heap_offset += (SHORT_SIZE + arglen * SHORT_SIZE)

        if cli_args:
            # Start argv array - store length, incremeent offset
            argc = len(cli_args)
            self.store_short(DATA_START + heap_offset, argc)
            argv_addr = DATA_START + heap_offset
            heap_offset += SHORT_SIZE

            # Store argv addrs + cli args arrays
            cli_args_arrays_start = DATA_START + heap_offset + SHORT_SIZE * argc # reserve argv array space
            cli_arg_addr = cli_args_arrays_start                                 # addr of first cli_arg array
            heap_offset += SHORT_SIZE * argc                                     # increment offset to match
            for i, arg in enumerate(cli_args):
                # TODO: replace more escaped characters
                arg = arg.replace(r"\n", "\n")
                self.store_short(argv_addr + SHORT_SIZE + SHORT_SIZE * i, cli_arg_addr) # store addr of cli arg in argv array
                # initialize array length
                arglen = len(arg)
                self.store_short(cli_arg_addr, arglen)
                heap_offset += SHORT_SIZE
                # store cli arg in its array
                for char in list(arg):
                    utf16_bytes = list(char.encode("utf-16-be"))
                    utf16_char = utf16_bytes[0] << 8 | utf16_bytes[1]
                    self.store_short(DATA_START + heap_offset, utf16_char)
                    heap_offset += SHORT_SIZE
                cli_arg_addr = DATA_START + heap_offset
        else:
            # initialize argv to 0 if not cli args
            self.store_short(DATA_START + heap_offset, 0)
            heap_offset += SHORT_SIZE

        # Initialize the frame, stack, and global pointers
        self.reg[reg_to_bin["fp"]] = self.reg[reg_to_bin["sp"]] = STACK_START
        self.reg[reg_to_bin["gp"]] = DATA_START + heap_offset

    def run(self):
        while True:
            try:
                self.ir = self.load_word(self.pc)
            except IndexError:
                sys.exit(f"Error: attempted to execute instruction {self.get_pc_line()}; halting the machine")

            self.debug_exec()

            opcode = bin_to_opcode[self.ir >> 24]
            op_fn = getattr(self, f"op_{opcode}")
            args = self.extract_args(opcode_to_argmask[opcode])
            op_fn(*args) if args else op_fn()

            # TODO: actually delay and visualize
            if self.count_calls:
                opcode_to_calls[opcode] += 1

    def debug_exec(self):
        if not self.debug:
            return
        if self.get_pc_line() in self.breakpoints:
            pass
        elif self.debug_continue_flag:
            return

        print(f"{self.get_pc_line()} {self.program.lines[self.program.pc_to_abs_line[self.get_pc_line()]]}")
        print()

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
                    self.breakpoints.add(self.get_pc_line())
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
                    self.print_stack(STACK_START, tc_b32_to_int(self.reg[reg_to_bin["sp"]]))
                    continue
                if args[0] not in {"stack", "reg", "mem", "program", "s", "r", "m", "p"}:
                    print(f"Invalid print object: {args[0]}")
                    continue
                if args[0] in {"stack", "s"}:
                    # case 1: print entire stack
                    if len(args) == 1:
                        self.print_stack(STACK_START, tc_b32_to_int(self.reg[reg_to_bin["sp"]]))
                        continue
                    # case 2: print stack range
                    # print stack <start> <stop>
                    if len(args) != 3:
                        print(f"Invalid syntax: {"TODO"}")
                        continue
                    if int(args[1]) > STACK_START or int(args[1]) < 0 \
                        or int(args[2]) > STACK_START or int(args[2]) < 0 \
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
                line_prefix = ""
                line_prefix += ">" if pc == self.get_pc_line() else " "
                line_prefix += "*" if pc in self.breakpoints else " "
                print(f"{line_prefix}{pc :>4} {line :<30}", end="")
                print(" ".join([format(b, "08b") for b in self.mem[pc * 4:pc * 4 + 4]]))
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

    def op_halt(self):
        if self.count_calls:
            print_opcode_cost()
        sys.exit()

    def op_readi(self, rX: int):
        while True:
            try:
                x = int(input())
                if (valid_int(x)):
                    break
                raise ValueError
            except ValueError:
                print("Illegal input: input must be a number must be in [-32768, 32767]")
        self.reg[rX] = tc_int_to_b32(x)
        self.step_pc()


    def op_readf(self, rX: int):
        while True:
            try:
                x = float(input())
                break
            except ValueError:
                # TODO: better error message
                print("Illegal input: input must be a number")
        self.reg[rX] = fp_float_to_f32(x)
        self.step_pc()

    def op_readc(self, rX: int):
        byte_list = list(str(getch()).encode("utf-16-be"))
        self.reg[rX] = byte_list[0] << 8 | byte_list[1]
        self.step_pc()

    def op_writei(self, rX: int):
        print(tc_b32_to_int(self.reg[rX]))
        self.step_pc()

    def op_writef(self, rX: int):
        print(fp_f32_to_float(self.reg[rX]))
        self.step_pc()

    def op_writec(self, rX: int):
        # TODO: how to handle value outside of range? clamp?
        print(self.reg[rX].to_bytes(2, 'big').decode("utf-16-be"), end="")
        self.step_pc()

    def op_writes(self, rX: int):
        addr = tc_b32_to_int(self.reg[rX])
        arrlen = self.load_short(addr)
        # TODO: this approach is ugly, stop doing this
        for i in range(SHORT_SIZE, SHORT_SIZE + SHORT_SIZE * arrlen, SHORT_SIZE):
            char = self.load_short(addr + i).to_bytes(2, 'big').decode("utf-16-be")
            print(char, end="")
        self.step_pc()

    def op_seed(self, rX: int):
        random.seed((self.reg[rX]))
        self.step_pc()

    def op_rand(self, rX: int, rY: int, rZ: int):
        lo = tc_b32_to_int(self.reg[rY])
        hi = tc_b32_to_int(self.reg[rZ])
        self.reg[rX] = tc_int_to_b32(random.randint(lo, hi))
        self.step_pc()

    def op_time(self, rX: int):
        now = datetime.datetime.now()
        self.reg[rX] = (
            now.hour * 3600 + now.minute * 60 + now.second
        ) * 1000 + now.microsecond // 1000
        self.step_pc()


    def op_date(self, rX: int):
        today = datetime.date.today()
        # [31:13]: year  - 19 bits - 524287 values
        # [12:9]:  month - 4  bits - 16     values
        # [8:0]:   day   - 9  bits - 512    values
        self.reg[rX] = today.year << 13 | today.month << 9 | today.day
        self.step_pc()

    def op_nop(self):
        self.step_pc()

    # Arithmetic instructions

    def op_negi(self, rX: int, rY: int):
        self.reg[rX] = tc_neg(self.reg[rY])
        self.step_pc()

    def op_addi(self, rX: int, rY: int, rZ:int):
        self.reg[rX] = tc_add(self.reg[rY], self.reg[rZ])
        self.step_pc()

    def op_subi(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = tc_sub(self.reg[rY], self.reg[rZ])
        self.step_pc()

    def op_muli(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = tc_mul(self.reg[rY], self.reg[rZ])
        self.step_pc()

    def op_divi(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = tc_div(self.reg[rY], self.reg[rZ])
        self.step_pc()

    def op_modi(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = tc_mod(self.reg[rY], self.reg[rZ])
        self.step_pc()

    def op_negf(self, rX: int, rY: int):
        self.reg[rX] = self.reg[rY] ^ (1 << 31)
        self.step_pc()

    def op_addf(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = fp_float_to_f32(fp_f32_to_float(self.reg[rY]) + fp_f32_to_float(self.reg[rZ]))
        self.step_pc()

    def op_subf(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = fp_float_to_f32(fp_f32_to_float(self.reg[rY]) - fp_f32_to_float(self.reg[rZ]))
        self.step_pc()

    def op_mulf(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = fp_float_to_f32(fp_f32_to_float(self.reg[rY]) * fp_f32_to_float(self.reg[rZ]))
        self.step_pc()

    def op_divf(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = fp_float_to_f32(fp_f32_to_float(self.reg[rY]) / fp_f32_to_float(self.reg[rZ]))
        self.step_pc()

    # Bitwise instructions

    def op_and(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = self.reg[rY] & self.reg[rZ]
        self.step_pc()

    def op_or(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = self.reg[rY] | self.reg[rZ]
        self.step_pc()

    def op_xor(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = self.reg[rY] ^ self.reg[rZ]
        self.step_pc()

    def op_not(self, rX: int, rY: int):
        self.reg[rX] = ~self.reg[rY]
        self.step_pc()

    def op_lshl(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = (self.reg[rY] << self.reg[rZ]) & 0xffffffff
        self.step_pc()

    def op_lshr(self, rX: int, rY: int, rZ: int):
        self.reg[rX] = (self.reg[rY] >> self.reg[rZ]) & 0xffffffff
        self.step_pc()

    # TODO: verify this
    def op_ashl(self, rX: int, rY: int, rZ: int):
        sign = self.reg[rY] & (1 << 31)
        temp = ((self.reg[rY] ^ sign) << self.reg[rZ]) & 0xffffffff
        self.reg[rX] = 0 if temp == 0 else temp | sign
        self.step_pc()

    # TODO: verify this
    def op_ashr(self, rX: int, rY: int, rZ: int):
        num_shifts = tc_b32_to_int(self.reg[rZ])
        sign = self.reg[rY] & (1 << 31)
        sign_extend = int("1" * num_shifts, 2) if sign else 0
        temp = ((self.reg[rY] ^ sign) >> num_shifts) | (sign_extend << (31 - num_shifts))
        self.reg[rX] = 0 if temp == 0 else temp | sign
        self.step_pc()

    # Jump Instructions

    def op_j(self, addr: int):
        self.pc = addr

    def op_jr(self, rX: int):
        self.pc = self.reg[rX]

    def op_jeqz(self, rX: int, addr: int):
        self.pc = (
            addr
            if tc_b32_to_int(self.reg[rX]) == 0
            else self.pc + WORD_SIZE
        )

    def op_jnez(self, rX: int, addr: int):
        self.pc = (
            addr
            if tc_b32_to_int(self.reg[rX]) != 0
            else self.pc + WORD_SIZE
        )

    def op_jge(self, rX: int, rY: int, addr: int):
        self.pc = (
            addr
            if tc_b32_to_int(self.reg[rX]) >= tc_b32_to_int(self.reg[rY])
            else self.pc + WORD_SIZE
        )

    def op_jle(self, rX: int, rY: int, addr: int):
        self.pc = (
            addr
            if tc_b32_to_int(self.reg[rX]) <= tc_b32_to_int(self.reg[rY])
            else self.pc + WORD_SIZE
        )

    def op_jeq(self, rX: int, rY: int, addr: int):
        self.pc = (
            addr
            if tc_b32_to_int(self.reg[rX]) == tc_b32_to_int(self.reg[rY])
            else self.pc + WORD_SIZE
        )

    def op_jne(self, rX: int, rY: int, addr: int):
        self.pc = (
            addr
            if tc_b32_to_int(self.reg[rX]) != tc_b32_to_int(self.reg[rY])
            else self.pc + WORD_SIZE
        )

    def op_jgt(self, rX: int, rY: int, addr: int):
        self.pc = (
            addr
            if tc_b32_to_int(self.reg[rX]) > tc_b32_to_int(self.reg[rY])
            else self.pc + WORD_SIZE
        )

    def op_jlt(self, rX: int, rY: int, addr: int):
        self.pc = (
            addr
            if tc_b32_to_int(self.reg[rX]) < tc_b32_to_int(self.reg[rY])
            else self.pc + WORD_SIZE
        )

    def op_jsr(self, rX: int, addr: int):
        self.reg[rX] = tc_int_to_b32(self.pc + WORD_SIZE)
        self.pc = addr

    # Register instructions

    def op_seti(self, rX: int, val: int):
        self.reg[rX] = tc_b16_to_b32(val)
        self.step_pc()

    def op_inci(self, rX: int, val: int):
        self.reg[rX] = tc_add(self.reg[rX], tc_b16_to_b32(val))
        self.step_pc()

    def op_copy(self, rX: int, rY: int):
        self.reg[rX] = self.reg[rY]
        self.step_pc()

    # Stack instructions

    def op_pushb(self, rX: int, rY: int):
        if self.reg[reg_to_bin["sp"]] <= self.reg[reg_to_bin["gp"]]:
            sys.exit(f"Error: stack overflow attempting to execute instruction {self.get_pc_line()}; halting the machine")
        addr = tc_b32_to_int(self.reg[rY])
        byte = self.reg[rX]
        self.store_byte(addr, byte)
        self.reg[rY] = tc_int_to_b32(addr - BYTE_SIZE)
        self.step_pc()

    def op_pushs(self, rX: int, rY: int):
        if self.reg[reg_to_bin["sp"]] <= self.reg[reg_to_bin["gp"]]:
            sys.exit(f"Error: stack overflow attempting to execute instruction {self.get_pc_line()}; halting the machine")
        addr = tc_b32_to_int(self.reg[rY])
        short = self.reg[rX]
        self.store_short(addr, short)
        self.reg[rY] = tc_int_to_b32(addr - SHORT_SIZE)
        self.step_pc()

    def op_pushw(self, rX: int, rY: int):
        if self.reg[reg_to_bin["sp"]] <= self.reg[reg_to_bin["gp"]]:
            sys.exit(f"Error: stack overflow attempting to execute instruction {self.get_pc_line()}; halting the machine")
        addr = tc_b32_to_int(self.reg[rY])
        word = self.reg[rX]
        self.store_word(addr, word)
        self.reg[rY] = tc_int_to_b32(addr - WORD_SIZE)
        self.step_pc()

    def op_popb(self, rX: int, rY: int):
        addr = tc_b32_to_int(self.reg[rY]) + BYTE_SIZE
        self.reg[rY] = tc_int_to_b32(addr)
        self.reg[rX] = self.load_byte(addr)
        self.step_pc()

    def op_pops(self, rX: int, rY: int):
        addr = tc_b32_to_int(self.reg[rY]) + SHORT_SIZE
        self.reg[rY] = tc_int_to_b32(addr)
        self.reg[rX] = self.load_short(addr)
        self.step_pc()

    def op_popw(self, rX: int, rY: int):
        addr = tc_b32_to_int(self.reg[rY]) + WORD_SIZE
        self.reg[rY] = tc_int_to_b32(addr)
        self.reg[rX] = self.load_word(addr)
        self.step_pc()

    # Load/store stack instructions

    def op_lda(self, rX: int, val: int):
        self.reg[rX] = tc_int_to_b32(val)
        self.step_pc()

    def op_ldb(self, rX: int, rY: int, offset: int):
        addr = tc_b32_to_int(self.reg[rY])
        offset = tc_b16_to_int(offset)
        self.reg[rX] = self.load_byte(addr + offset)
        self.step_pc()

    def op_lds(self, rX: int, rY: int, offset: int):
        addr = tc_b32_to_int(self.reg[rY])
        offset = tc_b16_to_int(offset)
        self.reg[rX] = self.load_short(addr + offset)
        self.step_pc()

    def op_ldw(self, rX: int, rY: int, offset: int):
        addr = tc_b32_to_int(self.reg[rY])
        offset = tc_b16_to_int(offset)
        self.reg[rX] = self.load_word(addr + offset)
        self.step_pc()

    def op_stb(self, rX: int, rY: int, offset: int):
        addr = tc_b32_to_int(self.reg[rY])
        offset = tc_b16_to_int(offset)
        byte = self.reg[rX]
        self.store_byte(addr + offset, byte)
        self.step_pc()

    def op_sts(self, rX: int, rY: int, offset: int):
        addr = tc_b32_to_int(self.reg[rY])
        offset = tc_b16_to_int(offset)
        short = self.reg[rX]
        self.store_short(addr + offset, short)
        self.step_pc()

    def op_stw(self, rX: int, rY: int, offset: int):
        addr = tc_b32_to_int(self.reg[rY])
        offset = tc_b16_to_int(offset)
        word = self.reg[rX]
        self.store_word(addr + offset, word)
        self.step_pc()

    # array instructions

    def op_anewb(self, rX: int, rY: int):
        addr = tc_b32_to_int(self.reg[rX])
        len = self.reg[rY]
        self.store_short(addr, len)
        # TODO: ensure len > 0
        for i in range(2, 2 + len):
            self.mem[addr + i] = 0
        self.step_pc()

    def op_anews(self, rX: int, rY: int):
        addr = tc_b32_to_int(self.reg[rX])
        len = self.reg[rY]
        self.store_short(addr, len)
        # TODO: ensure len > 0
        for i in range(SHORT_SIZE, SHORT_SIZE + len * SHORT_SIZE):
            self.mem[addr + i] = 0
        self.step_pc()

    def op_aneww(self, rX: int, rY: int):
        addr = tc_b32_to_int(self.reg[rX])
        len = self.reg[rY]
        self.store_short(addr, len)
        # TODO: ensure len > 0
        for i in range(SHORT_SIZE, SHORT_SIZE + len * WORD_SIZE):
            self.mem[addr + i] = 0
        self.step_pc()

    def op_aldb(self, rX: int, rY: int, rZ: int):
        addr = tc_b32_to_int(self.reg[rY])
        arrlen = self.load_short(addr)
        index = tc_b16_to_int(self.reg[rZ])
        if index > arrlen - 1:
            sys.exit(f"Error: out of bounds array access at instruction {self.get_pc_line()}; halting the machine")
        offset = SHORT_SIZE + index
        self.reg[rX] = self.load_byte(addr + offset)
        self.step_pc()

    def op_alds(self, rX: int, rY: int, rZ: int):
        addr = tc_b32_to_int(self.reg[rY])
        arrlen = self.load_short(addr)
        index = tc_b16_to_int(self.reg[rZ])
        if index > arrlen - 1:
            sys.exit(f"Error: out of bounds array access at instruction {self.get_pc_line()}; halting the machine")
        offset = SHORT_SIZE + index * SHORT_SIZE
        self.reg[rX] = self.load_short(addr + offset)
        self.step_pc()

    def op_aldw(self, rX: int, rY: int, rZ: int):
        addr = tc_b32_to_int(self.reg[rY])
        arrlen = self.load_short(addr)
        index = tc_b16_to_int(self.reg[rZ])
        if index > arrlen - 1:
            sys.exit(f"Error: out of bounds array access at instruction {self.get_pc_line()}; halting the machine")
        offset = SHORT_SIZE + index * WORD_SIZE
        self.reg[rX] = self.load_word(addr + offset)
        self.step_pc()

    def op_astb(self, rX: int, rY: int, rZ: int):
        addr = tc_b32_to_int(self.reg[rY])
        arrlen = self.load_short(addr)
        index = tc_b16_to_int(self.reg[rZ])
        if index > arrlen - 1:
            sys.exit(f"Error: out of bounds array access at instruction {self.get_pc_line()}; halting the machine")
        offset = SHORT_SIZE + index
        self.store_byte(addr + offset, self.reg[rX])
        self.step_pc()

    def op_asts(self, rX: int, rY: int, rZ: int):
        addr = tc_b32_to_int(self.reg[rY])
        arrlen = self.load_short(addr)
        index = tc_b16_to_int(self.reg[rZ])
        if index > arrlen - 1:
            sys.exit(f"Error: out of bounds array access at instruction {self.get_pc_line()}; halting the machine")
        offset = SHORT_SIZE + index * SHORT_SIZE
        self.store_short(addr + offset, self.reg[rX])
        self.step_pc()

    def op_astw(self, rX: int, rY: int, rZ: int):
        addr = tc_b32_to_int(self.reg[rY])
        arrlen = self.load_short(addr)
        index = tc_b16_to_int(self.reg[rZ])
        if index > arrlen - 1:
            sys.exit(f"Error: out of bounds array access at instruction {self.get_pc_line()}; halting the machine")
        offset = SHORT_SIZE + index * WORD_SIZE
        self.store_word(addr + offset, self.reg[rX])
        self.step_pc()

    def op_alen(self, rX: int, rY: int):
        addr = tc_b32_to_int(self.reg[rY])
        self.reg[rX] = self.load_short(addr)
        self.step_pc()

    def op_i2f(self, rX: int, rY: int):
        i = tc_b32_to_int(self.reg[rY])
        self.reg[rX] = fp_float_to_f32(i)
        self.step_pc()

    def op_i2c(self, rX: int, rY: int):
        i = tc_b32_to_int(self.reg[rY])
        # TODO: what am i converting based on?
        self.reg[rX] = i
        self.step_pc()

    def op_f2i(self, rX: int, rY: int):
        f = fp_f32_to_float(self.reg[rY])
        self.reg[rX] = int(f)
        self.step_pc()

    def store_byte(self, addr: int, byte: int):
        if not self.valid_address(addr):
            sys.exit(f"Error: attempted to store outside of address range at instruction {self.get_pc_line()}; halting the machine")
        self.mem[addr] = byte & 0xff

    def store_short(self, addr: int, short: int):
        if not self.valid_address(addr):
            sys.exit(f"Error: attempted to store outside of address range at instruction {self.get_pc_line()}; halting the machine")
        self.mem[addr]     = (short >> 8) & 0xff
        self.mem[addr + 1] =  short       & 0xff

    def store_word(self, addr: int, word: int):
        if not self.valid_address(addr):
            sys.exit(f"Error: attempted to store outside of address range at instruction {self.get_pc_line()}; halting the machine")
        self.mem[addr]     = (word >> 24) & 0xff
        self.mem[addr + 1] = (word >> 16) & 0xff
        self.mem[addr + 2] = (word >>  8) & 0xff
        self.mem[addr + 3] =  word        & 0xff

    def load_byte(self, addr: int) -> int:
        if not self.valid_address(addr):
            sys.exit(f"Error: attempted to load outside of address range at instruction {self.get_pc_line()}; halting the machine")
        return self.mem[addr]

    def load_short(self, addr: int) -> int:
        if not self.valid_address(addr):
            sys.exit(f"Error: attempted to load outside of address range at instruction {self.get_pc_line()}; halting the machine")
        return (
            self.mem[addr] << 8
          | self.mem[addr + 1]
        )

    def load_word(self, addr: int) -> int:
        if not self.valid_address(addr):
            sys.exit(f"Error: attempted to load outside of address range at instruction {self.get_pc_line()}; halting the machine")
        return (
            self.mem[addr]     << 24
          | self.mem[addr + 1] << 16
          | self.mem[addr + 2] << 8
          | self.mem[addr + 3]
        )

    def valid_address(self, addr: int):
        return 0 <= addr and addr <= STACK_START

    def get_pc_line(self):
        return self.pc // WORD_SIZE

    def extract_args(self, mask: str) -> list[int]:
        ir = self.ir
        ret: list[int] = []
        for c in reversed(mask):
            if c == "r":
                ret.insert(0, ir & 0xf)
                ir >>= 4
            elif c == "n" or c == "f" or c == "l" or c == "a":
                ret.insert(0, ir & 0xffff)
                ir >>= 16
        return ret

class Parser:
    def __init__(self, inFile: str):
        self.inFile: str = inFile

        self.lines: list[str] = []
        self.pc_to_abs_line: dict[int, int] = {}
        self.abs_to_pc_line: dict[int, int] = {}

        self.machine_code: list[tuple[int, int, int, int]] = []
        self.labels: dict[str, int] = {}

        self.data_ids: dict[str, tuple[str, list[int], int]] = {}
        self.data_offset: int = 0

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
            line = line.strip()
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

            if text_found:
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

        # Validate data section.
        for line in lines[data_start:data_end]:
            line = line.strip()
            lineno = data_start + 1

            # Skip empty lines, comments, and self.labels.
            if not line or line.startswith("#") or line.endswith(":"):
                continue

            # Remove inlined comment if any.
            if "#" in line:
                line = line[:line.find("#")].strip()

            toks = shlex.split(line)

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
                    self.data_ids[toks[1]] = (toks[0], [tc_int_to_b32(int(toks[3]))], self.data_offset)
                    # TODO: change this for varying width types
                    self.data_offset += 4
            elif toks[0] == ".float":
                if is_float(toks[3]):
                    self.data_ids[toks[1]] = (toks[0], [fp_float_to_f32(float(toks[3]))], self.data_offset)
                    self.data_offset += 4
            elif toks[0] == ".char":
                # TODO: require quotes?
                utf16_bytes = list(toks[3].encode("utf-16-be"))
                utf16_char = utf16_bytes[0] << 8 | utf16_bytes[1]
                if len(utf16_bytes) != 2:
                    sys.exit(f"Error {self.inFile}@{lineno}: invalid unicode character '{toks[3]}'")
                self.data_ids[toks[1]] = (toks[0], [utf16_char], self.data_offset)
                self.data_offset += 2
            elif toks[0] == ".string":
                string = toks[3]
                string = string.replace(r"\n", "\n")
                utf16_chars: list[int] = []
                for c in list(string):
                    utf16_bytes = list(c.encode("utf-16-be"))
                    utf16_char = utf16_bytes[0] << 8 | utf16_bytes[1]
                    utf16_chars.append(utf16_char)
                self.data_ids[toks[1]] = (toks[0], utf16_chars, self.data_offset)
                self.data_offset += len(toks[3]) * SHORT_SIZE

        pc_line = 0
        abs_line = 0

        halt_found = False

        lineno = text_start + 1
        # Validate text section.
        for line in lines[text_start:text_end]:
            line = line.strip()

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
                sys.exit(f"Error {self.inFile}@{lineno}: opcode {opcode} expects {len(argmask)} arguments: '{argmask}'")
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
            lineno += 1

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
                elif c == "l":
                    addr = self.labels[args[i]] * WORD_SIZE
                    byte_list[curr_byte] = addr & 0xff
                    byte_list[curr_byte - 1] = addr >> 8
                    curr_byte -= 2
                elif c == "a":
                    if self.valid_variable(args[i]):
                        if args[i] == "argc":
                            addr = DATA_START + self.data_offset
                        elif args[i] == "argv":
                            addr = DATA_START + self.data_offset
                        else:
                            addr = DATA_START + self.data_ids[args[i]][2]
                        byte_list[curr_byte] = addr & 0xff
                        byte_list[curr_byte - 1] = addr >> 8
                        curr_byte -= 2
                    else:
                        addr = int(args[i])
                        byte_list[curr_byte] = addr & 0xff
                        byte_list[curr_byte - 1] = addr >> 8
                        curr_byte -= 2

                i -= 1

            code = (byte_list[0], byte_list[1], byte_list[2], byte_list[3])
            self.machine_code.append(code)

    def valid_label(self, label: str) -> bool:
        return label in self.labels.keys()

    def valid_variable(self, variable: str) -> bool:
        return variable in self.data_ids.keys() or variable == "argc" or variable == "argv"

    def valid_address(self, address: str) -> bool:
        try:
            addr = int(address)
            return 0 <= addr and addr <= STACK_START
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

def fp_float_to_f32(val: float) -> int:
    # TODO: test speed between struct and manual conversion
    bstr = "".join(format(byte, "08b") for byte in struct.pack("!f", val))
    return int(bstr, 2)

def fp_f32_to_float(val: int) -> float:
    # TODO: test speed between struct and manual conversion
    return struct.unpack("!f", val.to_bytes(4))[0]

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
    except (KeyboardInterrupt, EOFError):
        sys.exit()
