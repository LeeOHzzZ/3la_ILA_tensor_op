
import json
import sys
import timeit

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, NewType, Tuple, Callable, Union, Mapping

from src.converter import Converter
from src.utils import tool as Simulator

# Preliminaries
IlaASMInstr = Mapping[str, str]
IlaASMDataLib = Mapping[str, str]
IlaASMProgram = Tuple[List[IlaASMInstr], IlaASMDataLib]
IlaSimOutputFile = NewType('IlaSimOutputFile', str)

# Contexts

LocalContext = Mapping[str, Any]

@dataclass
class DataBlock:
  parent_store: str
  address: int
  size: int
  type_info: Any

AcceleratorContext = Mapping[str, DataBlock]

DriverContext = Tuple[LocalContext, AcceleratorContext]
# TODO: properly load data out of simulator, don't just hand
# simulator to the DriverAction function.
DriverAction = Callable[[IlaSimOutputFile, DriverContext], DriverContext]

# References to data in contexts

@dataclass
class LocalRef:
  name: str
  
@dataclass
class AccelRef:
  name: str

InstrArg = Union[LocalRef, AccelRef, Any]
ArgParser = Callable[[str], InstrArg]

# Instructions manipulating data in driver/accelerator contexts

@dataclass(frozen=True)
class MemInstruction:
  name: str
  args: List[Tuple[str, ArgParser]]
  # TODO: have proper type checking, don't just use Any
  execute: Callable[[DriverContext, List[InstrArg]], 
    Tuple[IlaASMProgram, Union[AcceleratorContext, DriverAction]]]
    # instructions can be chained in a single program fragment unless one 
    # of them creates a new DriverContext

@dataclass(frozen=True)
class ASMInstruction:
  name: str
  args: List[Tuple[str, ArgParser]]
  # TODO: have proper type checking, don't just use Any
  execute: Callable[[AcceleratorContext, List[InstrArg]],
    Tuple[IlaASMProgram, AcceleratorContext]]

@dataclass(frozen=True)
class MemoryModel:
  init: Callable[[], AcceleratorContext]
  instructions: List[MemInstruction]

@dataclass(frozen=True)
class AcceleratorASM:
  mem_model: MemoryModel
  instructions: List[ASMInstruction]


def build_driver(asm: AcceleratorASM, namespace=None, mem_namespace=None):
  
  namespace = (namespace + '.') if namespace else ''
  mem_namespace = namespace + ((mem_namespace + '.') if mem_namespace else '')

  instr_map = {}
  for i in asm.instructions:
    iname = namespace + i.name
    if iname in instr_map:
      raise KeyError("Duplicate instruction: ", iname)
    instr_map[iname] = i
  for i in asm.mem_model.instructions:
    iname = mem_namespace + i.name
    if iname in instr_map:
      raise KeyError("Duplicate instruction: ", iname)
    instr_map[iname] = i

  def parse_instr(instr_dict):
    try:
      instr = instr_map[instr_dict['name']]
    except KeyError:
      raise KeyError("unrecognized instruction: ", instr_dict['name'])
    args = []
    for (name, parser) in instr.args:
      try:
        args.append(parser(instr_dict[name]))
      except KeyError:
        raise KeyError("missing argument: ", name)
      except ValueError as e:
        raise ValueError(f"failed to parse argument '{name}': {e}")
    return instr, args

  class Driver:

    def __init__(self, program, local_ctx):
      self.program = program
      self.loc_ctx = local_ctx
      self.progloc = 0
      self.simulator = Simulator()

    def try_finish(self):

      print('\n--------------------------------------------------------------')
      print(f'\tCompiling fragment beginning at instr. {self.progloc}')
      print('--------------------------------------------------------------\n')

      # TODO: stop resetting the accelerator
      print("! WARNING: Resetting accelerator. !")
      acc_ctx = asm.mem_model.init()

      progf = []
      datalib = {}
      while self.progloc < len(self.program):
        instr, args = parse_instr(self.program[self.progloc])
        (code, data), update = instr.execute((self.loc_ctx, acc_ctx), *args)
        self.progloc += 1
        progf += code
        datalib.update(data)
        if callable(update):
          # TODO, FIXME, HACK for isinstance(update, DriverAction)
          break
        else:
          # TODO, FIXME, HACK for isinstance(update, AcceleratorContext):
          acc_ctx = update

      asm_file = f'./test/{namespace}_asm.json'
      with open(asm_file, 'w') as fout:
        json.dump({'asm': progf}, fout, indent=4)
      print(f'*** ILA tensor assembly has been dumped to {asm_file} ***')

      dlib_file = f'./test/{namespace}_data_lib.json'
      with open(dlib_file, 'w') as fout:
        json.dump(datalib, fout, indent=4)
      print(f'\n*** data_lib has been dump to {dlib_file}***\n')

      cvtr = Converter(asm_file, dlib_file)
      cvtr.dump_ila_asm(f'./test/{namespace}_ila_asm.json')

      pfrag_file = f'./test/{namespace}_prog_frag_in.json'
      cvtr.dump_ila_prog_frag(pfrag_file)
      print(f'*** ILA program fragment has been dumped to {pfrag_file}***\n')

      print('\n--------------------------------------------------------------')
      print('\tinvoking ILA simulator')
      print('--------------------------------------------------------------\n')
      # measure the time of ila simulation
      start_time = timeit.default_timer()
      self.simulator.call_ila_simulator('float32', pfrag_file, './test/adpf_result.tmp')
      end_time = timeit.default_timer()
      print('\n********* ILA simulator performance ***********')
      print('ILA simulator execution time is {:04f}s'.format(end_time - start_time))
      
      if callable(update):
        # TODO, FIXME, HACK for isinstance(update, DriverAction)
        (new_loc_ctx, new_acc_ctx) = update(
          IlaSimOutputFile('./test/adpf_result.tmp'), 
          (self.loc_ctx, acc_ctx))
        
        # TODO: stop resetting the accelerator
        self.loc_ctx = new_loc_ctx

      return (self.progloc == len(self.program))

    def local_ctx(self):
      return self.loc_ctx

  return Driver




