# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: risenlighten/lasvsim/train_sim/api/trainsim/scenario.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.api import field_behavior_pb2 as google_dot_api_dot_field__behavior__pb2
from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2
from openapi.v3 import annotations_pb2 as openapi_dot_v3_dot_annotations__pb2
from risenlighten.lasvsim.train_sim.api.trainsim import external_struct_pb2 as risenlighten_dot_lasvsim_dot_train__sim_dot_api_dot_trainsim_dot_external__struct__pb2
from risenlighten.lasvsim.api.qxmap import qxmap_pb2 as api_dot_v1_dot_qxmap_dot_qxmap__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n:risenlighten/lasvsim/train_sim/api/trainsim/scenario.proto\x12+risenlighten.lasvsim.train_sim.api.trainsim\x1a\x1fgoogle/api/field_behavior.proto\x1a\x1cgoogle/api/annotations.proto\x1a\x1copenapi/v3/annotations.proto\x1a\x41risenlighten/lasvsim/train_sim/api/trainsim/external_struct.proto\x1a\x18\x61pi/v1/qxmap/qxmap.proto\"*\n\x0bGetHdMapReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\"\x90\x01\n\x0bGetHdMapRes\x12\x41\n\x05hdmap\x18\x01 \x01(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.HdMap\x12>\n\x04\x64\x61ta\x18\x02 \x01(\x0b\x32\x30.risenlighten.lasvsim.api.datahub.qxmap.v1.Qxmap2\x8d\x01\n\x08Scenario\x12\x80\x01\n\x08GetHdMap\x12\x38.risenlighten.lasvsim.train_sim.api.trainsim.GetHdMapReq\x1a\x38.risenlighten.lasvsim.train_sim.api.trainsim.GetHdMapRes\"\x00\x42@Z>git.risenlighten.com/lasvsim/train_sim/api/trainsim;trainsimv1b\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'risenlighten.lasvsim.train_sim.api.trainsim.scenario_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z>git.risenlighten.com/lasvsim/train_sim/api/trainsim;trainsimv1'
  _globals['_GETHDMAPREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETHDMAPREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETHDMAPREQ']._serialized_start=293
  _globals['_GETHDMAPREQ']._serialized_end=335
  _globals['_GETHDMAPRES']._serialized_start=338
  _globals['_GETHDMAPRES']._serialized_end=482
  _globals['_SCENARIO']._serialized_start=485
  _globals['_SCENARIO']._serialized_end=626
# @@protoc_insertion_point(module_scope)
