# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: risenlighten/lasvsim/train_sim/api/trainsim/trainsim.proto
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
from validate2 import validate_pb2 as validate_dot_validate__pb2
from risenlighten.lasvsim.train_sim.api.trainsim import external_struct_pb2 as risenlighten_dot_lasvsim_dot_train__sim_dot_api_dot_trainsim_dot_external__struct__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n:risenlighten/lasvsim/train_sim/api/trainsim/trainsim.proto\x12+risenlighten.lasvsim.train_sim.api.trainsim\x1a\x1fgoogle/api/field_behavior.proto\x1a\x1cgoogle/api/annotations.proto\x1a\x1copenapi/v3/annotations.proto\x1a\x17validate/validate.proto\x1a\x41risenlighten/lasvsim/train_sim/api/trainsim/external_struct.proto\"$\n\x07InitReq\x12\x19\n\x0bscenario_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\".\n\x07InitRes\x12\x15\n\rsimulation_id\x18\x01 \x01(\t\x12\x0c\n\x04\x61\x64\x64r\x18\x02 \x01(\t\"&\n\x07StepReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\"(\n\x07StepRes\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\"&\n\x07StopReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\"\t\n\x07StopRes\"\'\n\x08ResetReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\"\n\n\x08ResetRes\"/\n\x10GetTrafficMapReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\"V\n\x10GetTrafficMapRes\x12\x42\n\x06hd_map\x18\x01 \x01(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.HdMap\"2\n\x13GetVehicleIdListReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\"#\n\x13GetVehicleIdListRes\x12\x0c\n\x04list\x18\x01 \x03(\t\"6\n\x17GetTestVehicleIdListReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\"\'\n\x17GetTestVehicleIdListRes\x12\x0c\n\x04list\x18\x01 \x03(\t\"P\n\x15GetVehicleBaseInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x1a\n\x07id_list\x18\x02 \x03(\tB\t\xfa\x42\x06\x92\x01\x03\x10\xe8\x07\"\xb3\x03\n\x15GetVehicleBaseInfoRes\x12\x63\n\tinfo_dict\x18\x01 \x03(\x0b\x32P.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleBaseInfoRes.InfoDictEntry\x1a\x83\x01\n\rInfoDictEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x61\n\x05value\x18\x02 \x01(\x0b\x32R.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleBaseInfoRes.VehicleBaseInfo:\x02\x38\x01\x1a\xae\x01\n\x0fVehicleBaseInfo\x12K\n\tbase_info\x18\x01 \x01(\x0b\x32\x38.risenlighten.lasvsim.train_sim.api.trainsim.ObjBaseInfo\x12N\n\x0c\x64ynamic_info\x18\x02 \x01(\x0b\x32\x38.risenlighten.lasvsim.train_sim.api.trainsim.DynamicInfo\"O\n\x15GetVehiclePositionReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x19\n\x07id_list\x18\x02 \x03(\tB\x08\xfa\x42\x05\x92\x01\x02\x10\x64\"\xf0\x01\n\x15GetVehiclePositionRes\x12k\n\rposition_dict\x18\x01 \x03(\x0b\x32T.risenlighten.lasvsim.train_sim.api.trainsim.GetVehiclePositionRes.PositionDictEntry\x1aj\n\x11PositionDictEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x44\n\x05value\x18\x02 \x01(\x0b\x32\x35.risenlighten.lasvsim.train_sim.api.trainsim.Position:\x02\x38\x01\"Q\n\x17GetVehicleMovingInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x19\n\x07id_list\x18\x02 \x03(\tB\x08\xfa\x42\x05\x92\x01\x02\x10\x64\"\x80\x02\n\x17GetVehicleMovingInfoRes\x12r\n\x10moving_info_dict\x18\x01 \x03(\x0b\x32X.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleMovingInfoRes.MovingInfoDictEntry\x1aq\n\x13MovingInfoDictEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12I\n\x05value\x18\x02 \x01(\x0b\x32:.risenlighten.lasvsim.train_sim.api.trainsim.ObjMovingInfo:\x02\x38\x01\"R\n\x18GetVehicleControlInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x19\n\x07id_list\x18\x02 \x03(\tB\x08\xfa\x42\x05\x92\x01\x02\x10\x64\"\x83\x02\n\x18GetVehicleControlInfoRes\x12u\n\x11\x63ontrol_info_dict\x18\x01 \x03(\x0b\x32Z.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleControlInfoRes.ControlInfoDictEntry\x1ap\n\x14\x43ontrolInfoDictEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12G\n\x05value\x18\x02 \x01(\x0b\x32\x38.risenlighten.lasvsim.train_sim.api.trainsim.ControlInfo:\x02\x38\x01\"T\n\x1bGetVehiclePerceptionInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\"\x8c\x03\n\x1bGetVehiclePerceptionInfoRes\x12\x64\n\x04list\x18\x01 \x03(\x0b\x32V.risenlighten.lasvsim.train_sim.api.trainsim.GetVehiclePerceptionInfoRes.PerceptionObj\x1a\x86\x02\n\rPerceptionObj\x12\x0e\n\x06obj_id\x18\x01 \x01(\t\x12K\n\tbase_info\x18\x02 \x01(\x0b\x32\x38.risenlighten.lasvsim.train_sim.api.trainsim.ObjBaseInfo\x12O\n\x0bmoving_info\x18\x03 \x01(\x0b\x32:.risenlighten.lasvsim.train_sim.api.trainsim.ObjMovingInfo\x12G\n\x08position\x18\x04 \x01(\x0b\x32\x35.risenlighten.lasvsim.train_sim.api.trainsim.Position\"T\n\x1bGetVehicleReferenceLinesReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\"r\n\x1bGetVehicleReferenceLinesRes\x12S\n\x0freference_lines\x18\x01 \x03(\x0b\x32:.risenlighten.lasvsim.train_sim.api.trainsim.ReferenceLine\"R\n\x19GetVehiclePlanningInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\"f\n\x19GetVehiclePlanningInfoRes\x12I\n\rplanning_path\x18\x01 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\"S\n\x1aGetVehicleTrackPathInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\"d\n\x1aGetVehicleTrackPathInfoRes\x12\x46\n\ntrack_path\x18\x01 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\"S\n\x1aGetVehicleCollisionInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\"4\n\x1aGetVehicleCollisionInfoRes\x12\x16\n\x0e\x63ollision_flag\x18\x01 \x01(\x08\"T\n\x1bGetVehicleNavigationInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\"s\n\x1bGetVehicleNavigationInfoRes\x12T\n\x0fnavigation_info\x18\x01 \x01(\x0b\x32;.risenlighten.lasvsim.train_sim.api.trainsim.NavigationInfo\"\x9d\x01\n\x19SetVehiclePlanningInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\x12I\n\rplanning_path\x18\x03 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\"\x1b\n\x19SetVehiclePlanningInfoRes\"V\n\x1dGetVehicleDynamicParamInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\"o\n\x1dGetVehicleDynamicParamInfoRes\x12N\n\x0c\x64ynamic_info\x18\x01 \x01(\x0b\x32\x38.risenlighten.lasvsim.train_sim.api.trainsim.DynamicInfo\"\x99\x01\n\x18SetVehicleControlInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\x12\x16\n\tste_wheel\x18\x03 \x01(\x01H\x00\x88\x01\x01\x12\x14\n\x07lon_acc\x18\x04 \x01(\x01H\x01\x88\x01\x01\x42\x0c\n\n_ste_wheelB\n\n\x08_lon_acc\"\x1a\n\x18SetVehicleControlInfoRes\"\xba\x01\n\x15SetVehiclePositionReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\x12\x46\n\x05point\x18\x03 \x01(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.PointH\x00\x88\x01\x01\x12\x10\n\x03phi\x18\x04 \x01(\x01H\x01\x88\x01\x01\x42\x08\n\x06_pointB\x06\n\x04_phi\"\x17\n\x15SetVehiclePositionRes\"\xec\x01\n\x17SetVehicleMovingInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\x12\x0e\n\x01u\x18\x03 \x01(\x01H\x00\x88\x01\x01\x12\x0e\n\x01v\x18\x04 \x01(\x01H\x01\x88\x01\x01\x12\x0e\n\x01w\x18\x05 \x01(\x01H\x02\x88\x01\x01\x12\x12\n\x05u_acc\x18\x06 \x01(\x01H\x03\x88\x01\x01\x12\x12\n\x05v_acc\x18\x07 \x01(\x01H\x04\x88\x01\x01\x12\x12\n\x05w_acc\x18\x08 \x01(\x01H\x05\x88\x01\x01\x42\x04\n\x02_uB\x04\n\x02_vB\x04\n\x02_wB\x08\n\x06_u_accB\x08\n\x06_v_accB\x08\n\x06_w_acc\"\x19\n\x17SetVehicleMovingInfoRes\"\x94\x02\n\x15SetVehicleBaseInfoReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\x12P\n\tbase_info\x18\x03 \x01(\x0b\x32\x38.risenlighten.lasvsim.train_sim.api.trainsim.ObjBaseInfoH\x00\x88\x01\x01\x12S\n\x0c\x64ynamic_info\x18\x04 \x01(\x0b\x32\x38.risenlighten.lasvsim.train_sim.api.trainsim.DynamicInfoH\x01\x88\x01\x01\x42\x0c\n\n_base_infoB\x0f\n\r_dynamic_info\"\x17\n\x15SetVehicleBaseInfoRes\"\xaf\x01\n\x18SetVehicleDestinationReq\x12\x1b\n\rsimulation_id\x18\x01 \x01(\tB\x04\xe2\x41\x01\x02\x12\x18\n\nvehicle_id\x18\x02 \x01(\tB\x04\xe2\x41\x01\x02\x12L\n\x0b\x64\x65stination\x18\x03 \x01(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.PointH\x00\x88\x01\x01\x42\x0e\n\x0c_destination\"\x1a\n\x18SetVehicleDestinationRes2\x98\x1e\n\nSimulation\x12t\n\x04Init\x12\x34.risenlighten.lasvsim.train_sim.api.trainsim.InitReq\x1a\x34.risenlighten.lasvsim.train_sim.api.trainsim.InitRes\"\x00\x12t\n\x04Step\x12\x34.risenlighten.lasvsim.train_sim.api.trainsim.StepReq\x1a\x34.risenlighten.lasvsim.train_sim.api.trainsim.StepRes\"\x00\x12t\n\x04Stop\x12\x34.risenlighten.lasvsim.train_sim.api.trainsim.StopReq\x1a\x34.risenlighten.lasvsim.train_sim.api.trainsim.StopRes\"\x00\x12w\n\x05Reset\x12\x35.risenlighten.lasvsim.train_sim.api.trainsim.ResetReq\x1a\x35.risenlighten.lasvsim.train_sim.api.trainsim.ResetRes\"\x00\x12\x8f\x01\n\rGetTrafficMap\x12=.risenlighten.lasvsim.train_sim.api.trainsim.GetTrafficMapReq\x1a=.risenlighten.lasvsim.train_sim.api.trainsim.GetTrafficMapRes\"\x00\x12\x98\x01\n\x10GetVehicleIdList\x12@.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleIdListReq\x1a@.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleIdListRes\"\x00\x12\xa4\x01\n\x14GetTestVehicleIdList\x12\x44.risenlighten.lasvsim.train_sim.api.trainsim.GetTestVehicleIdListReq\x1a\x44.risenlighten.lasvsim.train_sim.api.trainsim.GetTestVehicleIdListRes\"\x00\x12\x9e\x01\n\x12GetVehicleBaseInfo\x12\x42.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleBaseInfoReq\x1a\x42.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleBaseInfoRes\"\x00\x12\x9e\x01\n\x12GetVehiclePosition\x12\x42.risenlighten.lasvsim.train_sim.api.trainsim.GetVehiclePositionReq\x1a\x42.risenlighten.lasvsim.train_sim.api.trainsim.GetVehiclePositionRes\"\x00\x12\xa4\x01\n\x14GetVehicleMovingInfo\x12\x44.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleMovingInfoReq\x1a\x44.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleMovingInfoRes\"\x00\x12\xa7\x01\n\x15GetVehicleControlInfo\x12\x45.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleControlInfoReq\x1a\x45.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleControlInfoRes\"\x00\x12\xb0\x01\n\x18GetVehiclePerceptionInfo\x12H.risenlighten.lasvsim.train_sim.api.trainsim.GetVehiclePerceptionInfoReq\x1aH.risenlighten.lasvsim.train_sim.api.trainsim.GetVehiclePerceptionInfoRes\"\x00\x12\xb0\x01\n\x18GetVehicleReferenceLines\x12H.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleReferenceLinesReq\x1aH.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleReferenceLinesRes\"\x00\x12\xaa\x01\n\x16GetVehiclePlanningInfo\x12\x46.risenlighten.lasvsim.train_sim.api.trainsim.GetVehiclePlanningInfoReq\x1a\x46.risenlighten.lasvsim.train_sim.api.trainsim.GetVehiclePlanningInfoRes\"\x00\x12\xb0\x01\n\x18GetVehicleNavigationInfo\x12H.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleNavigationInfoReq\x1aH.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleNavigationInfoRes\"\x00\x12\xad\x01\n\x17GetVehicleTrackPathInfo\x12G.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleTrackPathInfoReq\x1aG.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleTrackPathInfoRes\"\x00\x12\xaa\x01\n\x16SetVehiclePlanningInfo\x12\x46.risenlighten.lasvsim.train_sim.api.trainsim.SetVehiclePlanningInfoReq\x1a\x46.risenlighten.lasvsim.train_sim.api.trainsim.SetVehiclePlanningInfoRes\"\x00\x12\xb7\x01\n\x1bGetVehicleDynamicParamsInfo\x12J.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleDynamicParamInfoReq\x1aJ.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleDynamicParamInfoRes\"\x00\x12\xad\x01\n\x17GetVehicleCollisionInfo\x12G.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleCollisionInfoReq\x1aG.risenlighten.lasvsim.train_sim.api.trainsim.GetVehicleCollisionInfoRes\"\x00\x12\xa7\x01\n\x15SetVehicleControlInfo\x12\x45.risenlighten.lasvsim.train_sim.api.trainsim.SetVehicleControlInfoReq\x1a\x45.risenlighten.lasvsim.train_sim.api.trainsim.SetVehicleControlInfoRes\"\x00\x12\x9e\x01\n\x12SetVehiclePosition\x12\x42.risenlighten.lasvsim.train_sim.api.trainsim.SetVehiclePositionReq\x1a\x42.risenlighten.lasvsim.train_sim.api.trainsim.SetVehiclePositionRes\"\x00\x12\xa4\x01\n\x14SetVehicleMovingInfo\x12\x44.risenlighten.lasvsim.train_sim.api.trainsim.SetVehicleMovingInfoReq\x1a\x44.risenlighten.lasvsim.train_sim.api.trainsim.SetVehicleMovingInfoRes\"\x00\x12\x9e\x01\n\x12SetVehicleBaseInfo\x12\x42.risenlighten.lasvsim.train_sim.api.trainsim.SetVehicleBaseInfoReq\x1a\x42.risenlighten.lasvsim.train_sim.api.trainsim.SetVehicleBaseInfoRes\"\x00\x12\xa7\x01\n\x15SetVehicleDestination\x12\x45.risenlighten.lasvsim.train_sim.api.trainsim.SetVehicleDestinationReq\x1a\x45.risenlighten.lasvsim.train_sim.api.trainsim.SetVehicleDestinationRes\"\x00\x42@Z>git.risenlighten.com/lasvsim/train_sim/api/trainsim;trainsimv1b\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'risenlighten.lasvsim.train_sim.api.trainsim.trainsim_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z>git.risenlighten.com/lasvsim/train_sim/api/trainsim;trainsimv1'
  _globals['_INITREQ'].fields_by_name['scenario_id']._options = None
  _globals['_INITREQ'].fields_by_name['scenario_id']._serialized_options = b'\342A\001\002'
  _globals['_STEPREQ'].fields_by_name['simulation_id']._options = None
  _globals['_STEPREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_STOPREQ'].fields_by_name['simulation_id']._options = None
  _globals['_STOPREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_RESETREQ'].fields_by_name['simulation_id']._options = None
  _globals['_RESETREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETTRAFFICMAPREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETTRAFFICMAPREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEIDLISTREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLEIDLISTREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETTESTVEHICLEIDLISTREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETTESTVEHICLEIDLISTREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEBASEINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLEBASEINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEBASEINFOREQ'].fields_by_name['id_list']._options = None
  _globals['_GETVEHICLEBASEINFOREQ'].fields_by_name['id_list']._serialized_options = b'\372B\006\222\001\003\020\350\007'
  _globals['_GETVEHICLEBASEINFORES_INFODICTENTRY']._options = None
  _globals['_GETVEHICLEBASEINFORES_INFODICTENTRY']._serialized_options = b'8\001'
  _globals['_GETVEHICLEPOSITIONREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLEPOSITIONREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEPOSITIONREQ'].fields_by_name['id_list']._options = None
  _globals['_GETVEHICLEPOSITIONREQ'].fields_by_name['id_list']._serialized_options = b'\372B\005\222\001\002\020d'
  _globals['_GETVEHICLEPOSITIONRES_POSITIONDICTENTRY']._options = None
  _globals['_GETVEHICLEPOSITIONRES_POSITIONDICTENTRY']._serialized_options = b'8\001'
  _globals['_GETVEHICLEMOVINGINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLEMOVINGINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEMOVINGINFOREQ'].fields_by_name['id_list']._options = None
  _globals['_GETVEHICLEMOVINGINFOREQ'].fields_by_name['id_list']._serialized_options = b'\372B\005\222\001\002\020d'
  _globals['_GETVEHICLEMOVINGINFORES_MOVINGINFODICTENTRY']._options = None
  _globals['_GETVEHICLEMOVINGINFORES_MOVINGINFODICTENTRY']._serialized_options = b'8\001'
  _globals['_GETVEHICLECONTROLINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLECONTROLINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLECONTROLINFOREQ'].fields_by_name['id_list']._options = None
  _globals['_GETVEHICLECONTROLINFOREQ'].fields_by_name['id_list']._serialized_options = b'\372B\005\222\001\002\020d'
  _globals['_GETVEHICLECONTROLINFORES_CONTROLINFODICTENTRY']._options = None
  _globals['_GETVEHICLECONTROLINFORES_CONTROLINFODICTENTRY']._serialized_options = b'8\001'
  _globals['_GETVEHICLEPERCEPTIONINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLEPERCEPTIONINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEPERCEPTIONINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_GETVEHICLEPERCEPTIONINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEREFERENCELINESREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLEREFERENCELINESREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEREFERENCELINESREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_GETVEHICLEREFERENCELINESREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEPLANNINGINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLEPLANNINGINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEPLANNINGINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_GETVEHICLEPLANNINGINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLETRACKPATHINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLETRACKPATHINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLETRACKPATHINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_GETVEHICLETRACKPATHINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLECOLLISIONINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLECOLLISIONINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLECOLLISIONINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_GETVEHICLECOLLISIONINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLENAVIGATIONINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLENAVIGATIONINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLENAVIGATIONINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_GETVEHICLENAVIGATIONINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEPLANNINGINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_SETVEHICLEPLANNINGINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEPLANNINGINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_SETVEHICLEPLANNINGINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEDYNAMICPARAMINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_GETVEHICLEDYNAMICPARAMINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_GETVEHICLEDYNAMICPARAMINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_GETVEHICLEDYNAMICPARAMINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLECONTROLINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_SETVEHICLECONTROLINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLECONTROLINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_SETVEHICLECONTROLINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEPOSITIONREQ'].fields_by_name['simulation_id']._options = None
  _globals['_SETVEHICLEPOSITIONREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEPOSITIONREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_SETVEHICLEPOSITIONREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEMOVINGINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_SETVEHICLEMOVINGINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEMOVINGINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_SETVEHICLEMOVINGINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEBASEINFOREQ'].fields_by_name['simulation_id']._options = None
  _globals['_SETVEHICLEBASEINFOREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEBASEINFOREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_SETVEHICLEBASEINFOREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEDESTINATIONREQ'].fields_by_name['simulation_id']._options = None
  _globals['_SETVEHICLEDESTINATIONREQ'].fields_by_name['simulation_id']._serialized_options = b'\342A\001\002'
  _globals['_SETVEHICLEDESTINATIONREQ'].fields_by_name['vehicle_id']._options = None
  _globals['_SETVEHICLEDESTINATIONREQ'].fields_by_name['vehicle_id']._serialized_options = b'\342A\001\002'
  _globals['_INITREQ']._serialized_start=292
  _globals['_INITREQ']._serialized_end=328
  _globals['_INITRES']._serialized_start=330
  _globals['_INITRES']._serialized_end=376
  _globals['_STEPREQ']._serialized_start=378
  _globals['_STEPREQ']._serialized_end=416
  _globals['_STEPRES']._serialized_start=418
  _globals['_STEPRES']._serialized_end=458
  _globals['_STOPREQ']._serialized_start=460
  _globals['_STOPREQ']._serialized_end=498
  _globals['_STOPRES']._serialized_start=500
  _globals['_STOPRES']._serialized_end=509
  _globals['_RESETREQ']._serialized_start=511
  _globals['_RESETREQ']._serialized_end=550
  _globals['_RESETRES']._serialized_start=552
  _globals['_RESETRES']._serialized_end=562
  _globals['_GETTRAFFICMAPREQ']._serialized_start=564
  _globals['_GETTRAFFICMAPREQ']._serialized_end=611
  _globals['_GETTRAFFICMAPRES']._serialized_start=613
  _globals['_GETTRAFFICMAPRES']._serialized_end=699
  _globals['_GETVEHICLEIDLISTREQ']._serialized_start=701
  _globals['_GETVEHICLEIDLISTREQ']._serialized_end=751
  _globals['_GETVEHICLEIDLISTRES']._serialized_start=753
  _globals['_GETVEHICLEIDLISTRES']._serialized_end=788
  _globals['_GETTESTVEHICLEIDLISTREQ']._serialized_start=790
  _globals['_GETTESTVEHICLEIDLISTREQ']._serialized_end=844
  _globals['_GETTESTVEHICLEIDLISTRES']._serialized_start=846
  _globals['_GETTESTVEHICLEIDLISTRES']._serialized_end=885
  _globals['_GETVEHICLEBASEINFOREQ']._serialized_start=887
  _globals['_GETVEHICLEBASEINFOREQ']._serialized_end=967
  _globals['_GETVEHICLEBASEINFORES']._serialized_start=970
  _globals['_GETVEHICLEBASEINFORES']._serialized_end=1405
  _globals['_GETVEHICLEBASEINFORES_INFODICTENTRY']._serialized_start=1097
  _globals['_GETVEHICLEBASEINFORES_INFODICTENTRY']._serialized_end=1228
  _globals['_GETVEHICLEBASEINFORES_VEHICLEBASEINFO']._serialized_start=1231
  _globals['_GETVEHICLEBASEINFORES_VEHICLEBASEINFO']._serialized_end=1405
  _globals['_GETVEHICLEPOSITIONREQ']._serialized_start=1407
  _globals['_GETVEHICLEPOSITIONREQ']._serialized_end=1486
  _globals['_GETVEHICLEPOSITIONRES']._serialized_start=1489
  _globals['_GETVEHICLEPOSITIONRES']._serialized_end=1729
  _globals['_GETVEHICLEPOSITIONRES_POSITIONDICTENTRY']._serialized_start=1623
  _globals['_GETVEHICLEPOSITIONRES_POSITIONDICTENTRY']._serialized_end=1729
  _globals['_GETVEHICLEMOVINGINFOREQ']._serialized_start=1731
  _globals['_GETVEHICLEMOVINGINFOREQ']._serialized_end=1812
  _globals['_GETVEHICLEMOVINGINFORES']._serialized_start=1815
  _globals['_GETVEHICLEMOVINGINFORES']._serialized_end=2071
  _globals['_GETVEHICLEMOVINGINFORES_MOVINGINFODICTENTRY']._serialized_start=1958
  _globals['_GETVEHICLEMOVINGINFORES_MOVINGINFODICTENTRY']._serialized_end=2071
  _globals['_GETVEHICLECONTROLINFOREQ']._serialized_start=2073
  _globals['_GETVEHICLECONTROLINFOREQ']._serialized_end=2155
  _globals['_GETVEHICLECONTROLINFORES']._serialized_start=2158
  _globals['_GETVEHICLECONTROLINFORES']._serialized_end=2417
  _globals['_GETVEHICLECONTROLINFORES_CONTROLINFODICTENTRY']._serialized_start=2305
  _globals['_GETVEHICLECONTROLINFORES_CONTROLINFODICTENTRY']._serialized_end=2417
  _globals['_GETVEHICLEPERCEPTIONINFOREQ']._serialized_start=2419
  _globals['_GETVEHICLEPERCEPTIONINFOREQ']._serialized_end=2503
  _globals['_GETVEHICLEPERCEPTIONINFORES']._serialized_start=2506
  _globals['_GETVEHICLEPERCEPTIONINFORES']._serialized_end=2902
  _globals['_GETVEHICLEPERCEPTIONINFORES_PERCEPTIONOBJ']._serialized_start=2640
  _globals['_GETVEHICLEPERCEPTIONINFORES_PERCEPTIONOBJ']._serialized_end=2902
  _globals['_GETVEHICLEREFERENCELINESREQ']._serialized_start=2904
  _globals['_GETVEHICLEREFERENCELINESREQ']._serialized_end=2988
  _globals['_GETVEHICLEREFERENCELINESRES']._serialized_start=2990
  _globals['_GETVEHICLEREFERENCELINESRES']._serialized_end=3104
  _globals['_GETVEHICLEPLANNINGINFOREQ']._serialized_start=3106
  _globals['_GETVEHICLEPLANNINGINFOREQ']._serialized_end=3188
  _globals['_GETVEHICLEPLANNINGINFORES']._serialized_start=3190
  _globals['_GETVEHICLEPLANNINGINFORES']._serialized_end=3292
  _globals['_GETVEHICLETRACKPATHINFOREQ']._serialized_start=3294
  _globals['_GETVEHICLETRACKPATHINFOREQ']._serialized_end=3377
  _globals['_GETVEHICLETRACKPATHINFORES']._serialized_start=3379
  _globals['_GETVEHICLETRACKPATHINFORES']._serialized_end=3479
  _globals['_GETVEHICLECOLLISIONINFOREQ']._serialized_start=3481
  _globals['_GETVEHICLECOLLISIONINFOREQ']._serialized_end=3564
  _globals['_GETVEHICLECOLLISIONINFORES']._serialized_start=3566
  _globals['_GETVEHICLECOLLISIONINFORES']._serialized_end=3618
  _globals['_GETVEHICLENAVIGATIONINFOREQ']._serialized_start=3620
  _globals['_GETVEHICLENAVIGATIONINFOREQ']._serialized_end=3704
  _globals['_GETVEHICLENAVIGATIONINFORES']._serialized_start=3706
  _globals['_GETVEHICLENAVIGATIONINFORES']._serialized_end=3821
  _globals['_SETVEHICLEPLANNINGINFOREQ']._serialized_start=3824
  _globals['_SETVEHICLEPLANNINGINFOREQ']._serialized_end=3981
  _globals['_SETVEHICLEPLANNINGINFORES']._serialized_start=3983
  _globals['_SETVEHICLEPLANNINGINFORES']._serialized_end=4010
  _globals['_GETVEHICLEDYNAMICPARAMINFOREQ']._serialized_start=4012
  _globals['_GETVEHICLEDYNAMICPARAMINFOREQ']._serialized_end=4098
  _globals['_GETVEHICLEDYNAMICPARAMINFORES']._serialized_start=4100
  _globals['_GETVEHICLEDYNAMICPARAMINFORES']._serialized_end=4211
  _globals['_SETVEHICLECONTROLINFOREQ']._serialized_start=4214
  _globals['_SETVEHICLECONTROLINFOREQ']._serialized_end=4367
  _globals['_SETVEHICLECONTROLINFORES']._serialized_start=4369
  _globals['_SETVEHICLECONTROLINFORES']._serialized_end=4395
  _globals['_SETVEHICLEPOSITIONREQ']._serialized_start=4398
  _globals['_SETVEHICLEPOSITIONREQ']._serialized_end=4584
  _globals['_SETVEHICLEPOSITIONRES']._serialized_start=4586
  _globals['_SETVEHICLEPOSITIONRES']._serialized_end=4609
  _globals['_SETVEHICLEMOVINGINFOREQ']._serialized_start=4612
  _globals['_SETVEHICLEMOVINGINFOREQ']._serialized_end=4848
  _globals['_SETVEHICLEMOVINGINFORES']._serialized_start=4850
  _globals['_SETVEHICLEMOVINGINFORES']._serialized_end=4875
  _globals['_SETVEHICLEBASEINFOREQ']._serialized_start=4878
  _globals['_SETVEHICLEBASEINFOREQ']._serialized_end=5154
  _globals['_SETVEHICLEBASEINFORES']._serialized_start=5156
  _globals['_SETVEHICLEBASEINFORES']._serialized_end=5179
  _globals['_SETVEHICLEDESTINATIONREQ']._serialized_start=5182
  _globals['_SETVEHICLEDESTINATIONREQ']._serialized_end=5357
  _globals['_SETVEHICLEDESTINATIONRES']._serialized_start=5359
  _globals['_SETVEHICLEDESTINATIONRES']._serialized_end=5385
  _globals['_SIMULATION']._serialized_start=5388
  _globals['_SIMULATION']._serialized_end=9252
# @@protoc_insertion_point(module_scope)
