# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: risenlighten/lasvsim/train_sim/api/trainsim/external_struct.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nArisenlighten/lasvsim/train_sim/api/trainsim/external_struct.proto\x12+risenlighten.lasvsim.train_sim.api.trainsim\"(\n\x05Point\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\"\xbd\x01\n\x06Header\x12\r\n\x05north\x18\x01 \x01(\x01\x12\r\n\x05south\x18\x02 \x01(\x01\x12\x0c\n\x04\x65\x61st\x18\x03 \x01(\x01\x12\x0c\n\x04west\x18\x04 \x01(\x01\x12H\n\x0c\x63\x65nter_point\x18\x05 \x01(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x0f\n\x07version\x18\x06 \x01(\t\x12\x0c\n\x04zone\x18\x07 \x01(\x01\x12\x10\n\x08use_bias\x18\x08 \x01(\x08\"\xd5\x01\n\x08Position\x12\x41\n\x05point\x18\x01 \x01(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x0b\n\x03phi\x18\x02 \x01(\x01\x12\x0f\n\x07lane_id\x18\x03 \x01(\t\x12\x0f\n\x07link_id\x18\x04 \x01(\t\x12\x13\n\x0bjunction_id\x18\x05 \x01(\t\x12\x12\n\nsegment_id\x18\x06 \x01(\t\x12\x17\n\x0f\x64is_to_lane_end\x18\x07 \x01(\x01\x12\x15\n\rposition_type\x18\x08 \x01(\x05\"\x87\x01\n\x0e\x44irectionPoint\x12\x41\n\x05point\x18\x01 \x01(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x10\n\x08pitching\x18\x02 \x01(\x01\x12\x0f\n\x07heading\x18\x03 \x01(\x01\x12\x0f\n\x07rolling\x18\x04 \x01(\x01\"\xf8\x01\n\x04Turn\x12T\n\x0f\x64irection_point\x18\x01 \x01(\x0b\x32;.risenlighten.lasvsim.train_sim.api.trainsim.DirectionPoint\x12\x0c\n\x04turn\x18\x02 \x01(\t\x12X\n\x0cturn_mapping\x18\x03 \x03(\x0b\x32\x42.risenlighten.lasvsim.train_sim.api.trainsim.Turn.TurnMappingEntry\x1a\x32\n\x10TurnMappingEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"R\n\x05Speed\x12\t\n\x01s\x18\x01 \x01(\x01\x12\x0e\n\x06length\x18\x02 \x01(\x01\x12\x0e\n\x05value\x18\xe8\x07 \x01(\x01\x12\r\n\x04uint\x18\xe9\x07 \x01(\t\x12\x0f\n\x06source\x18\xea\x07 \x01(\t\"\xaa\x01\n\x08LaneMark\x12\x42\n\x05shape\x18\xe8\x07 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12Z\n\x0flane_mark_attrs\x18\xe9\x07 \x03(\x0b\x32@.risenlighten.lasvsim.train_sim.api.trainsim.LaneMarkAttribution\"\x88\x01\n\x13LaneMarkAttribution\x12\x0e\n\x06length\x18\x01 \x01(\x01\x12\t\n\x01s\x18\x02 \x01(\x01\x12\x13\n\x0bstart_index\x18\x03 \x01(\x05\x12\x11\n\tend_index\x18\x04 \x01(\x05\x12\x0e\n\x05style\x18\xe8\x07 \x01(\t\x12\x0e\n\x05\x63olor\x18\xe9\x07 \x01(\t\x12\x0e\n\x05width\x18\xea\x07 \x01(\x01\"\xeb\x01\n\x08Junction\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x13\n\x0bjunction_id\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\x10\n\x08link_ids\x18\x05 \x03(\t\x12\x41\n\x05shape\x18\x07 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x18\n\x10traffic_light_id\x18\x08 \x01(\t\x12\x16\n\x0ein_segment_ids\x18\t \x03(\t\x12\x17\n\x0fout_segment_ids\x18\n \x03(\t\"\xdb\x01\n\x07Segment\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x12\n\nsegment_id\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x18\n\x10ordered_link_ids\x18\x04 \x03(\t\x12\x19\n\x11start_junction_id\x18\x06 \x01(\t\x12\x17\n\x0f\x65nd_junction_id\x18\x07 \x01(\t\x12\x0e\n\x06length\x18\x08 \x01(\x01\x12\x0f\n\x07heading\x18\t \x01(\x01\x12\x1d\n\x15traffic_light_pole_id\x18\n \x01(\t\x12\x10\n\x08s_offset\x18\x0b \x01(\x01\"\xa8\x04\n\x04Link\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x0f\n\x07link_id\x18\x02 \x01(\t\x12\x0f\n\x07pair_id\x18\x03 \x01(\t\x12\r\n\x05width\x18\x04 \x01(\x01\x12\x18\n\x10ordered_lane_ids\x18\x05 \x03(\t\x12\x10\n\x08lane_num\x18\x07 \x01(\x05\x12G\n\x0bstart_point\x18\x08 \x01(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x45\n\tend_point\x18\t \x01(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x10\n\x08gradient\x18\n \x01(\x01\x12\x12\n\nsegment_id\x18\x0b \x01(\t\x12\x0e\n\x06length\x18\x0c \x01(\x01\x12\x0c\n\x04type\x18\r \x01(\t\x12\x0f\n\x07heading\x18\x0e \x01(\x01\x12\x12\n\nlink_order\x18\x0f \x01(\x05\x12I\n\rleft_boundary\x18\x10 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12J\n\x0eright_boundary\x18\x11 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x11\n\troad_type\x18\x12 \x01(\t\x12\x10\n\x08s_offset\x18\x13 \x01(\x01\"\xe2\x04\n\x04Lane\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x0f\n\x07lane_id\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x13\n\x0blane_offset\x18\x04 \x01(\x05\x12\x0f\n\x07link_id\x18\x05 \x01(\t\x12?\n\x04turn\x18\x06 \x01(\x0b\x32\x31.risenlighten.lasvsim.train_sim.api.trainsim.Turn\x12\x42\n\x06speeds\x18\x07 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Speed\x12\x13\n\x0bstopline_id\x18\x08 \x01(\t\x12\r\n\x05width\x18\t \x01(\x01\x12G\n\x0b\x63\x65nter_line\x18\n \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x18\n\x10\x63onnect_link_ids\x18\x0b \x03(\t\x12M\n\x0eleft_lane_mark\x18\x0c \x01(\x0b\x32\x35.risenlighten.lasvsim.train_sim.api.trainsim.LaneMark\x12N\n\x0fright_lane_mark\x18\r \x01(\x0b\x32\x35.risenlighten.lasvsim.train_sim.api.trainsim.LaneMark\x12\x19\n\x11upstream_link_ids\x18\x0e \x03(\t\x12\x1b\n\x13\x64ownstream_link_ids\x18\x0f \x03(\t\x12\x12\n\ncut_s_list\x18\x10 \x03(\x01\x12\x0e\n\x06length\x18\x11 \x01(\x01\"r\n\x08Stopline\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x13\n\x0bstopline_id\x18\x02 \x01(\t\x12\x41\n\x05shape\x18\x03 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\"\x9a\x01\n\x0bTrafficSign\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x17\n\x0ftraffic_sign_id\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12T\n\x0f\x64irection_point\x18\x04 \x01(\x0b\x32;.risenlighten.lasvsim.train_sim.api.trainsim.DirectionPoint\"\x85\x05\n\x0cTrafficLight\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x18\n\x10traffic_light_id\x18\x02 \x01(\t\x12\x13\n\x0bjunction_id\x18\x03 \x01(\t\x12\r\n\x05\x63ycle\x18\x07 \x01(\x05\x12\x0e\n\x06offset\x18\x08 \x01(\x05\x12\x11\n\tis_yellow\x18\t \x01(\x08\x12h\n\x10movement_signals\x18\x63 \x03(\x0b\x32N.risenlighten.lasvsim.train_sim.api.trainsim.TrafficLight.MovementSignalsEntry\x1a\x96\x02\n\x0eMovementSignal\x12\x13\n\x0bmovement_id\x18\x01 \x01(\t\x12\x1d\n\x15traffic_light_pole_id\x18\x02 \x01(\t\x12o\n\x0fsignal_of_green\x18\x64 \x03(\x0b\x32V.risenlighten.lasvsim.train_sim.api.trainsim.TrafficLight.MovementSignal.SignalOfGreen\x1a_\n\rSignalOfGreen\x12\x13\n\x0bgreen_start\x18\x01 \x01(\x05\x12\x16\n\x0egreen_duration\x18\x02 \x01(\x05\x12\x0e\n\x06yellow\x18\x03 \x01(\x05\x12\x11\n\tred_clean\x18\x04 \x01(\x05\x1a\x80\x01\n\x14MovementSignalsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12W\n\x05value\x18\x02 \x01(\x0b\x32H.risenlighten.lasvsim.train_sim.api.trainsim.TrafficLight.MovementSignal:\x02\x38\x01\"\x92\x01\n\x08Movement\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x13\n\x0bmovement_id\x18\x02 \x01(\t\x12\x18\n\x10upstream_link_id\x18\x03 \x01(\t\x12\x1a\n\x12\x64ownstream_link_id\x18\x04 \x01(\t\x12\x13\n\x0bjunction_id\x18\x05 \x01(\t\x12\x16\n\x0e\x66low_direction\x18\x06 \x01(\t\"\xb1\x02\n\nConnection\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x15\n\rconnection_id\x18\x02 \x01(\t\x12\x13\n\x0bjunction_id\x18\x03 \x01(\t\x12\x13\n\x0bmovement_id\x18\x04 \x01(\t\x12\x18\n\x10upstream_lane_id\x18\x05 \x01(\t\x12\x1a\n\x12\x64ownstream_lane_id\x18\x06 \x01(\t\x12@\n\x04path\x18\x07 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x16\n\x0e\x66low_direction\x18\x08 \x01(\t\x12\x18\n\x10upstream_link_id\x18\t \x01(\t\x12\x1a\n\x12\x64ownstream_link_id\x18\n \x01(\t\x12\x0c\n\x04type\x18\x0b \x01(\t\"\x86\x01\n\tCrosswalk\x12\x0e\n\x06map_id\x18\x01 \x01(\x04\x12\x14\n\x0c\x63rosswalk_id\x18\x02 \x01(\t\x12\x0f\n\x07heading\x18\x03 \x01(\x01\x12\x42\n\x05shape\x18\xe8\x07 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\"\xba\x06\n\x05HdMap\x12\x43\n\x06header\x18\x01 \x01(\x0b\x32\x33.risenlighten.lasvsim.train_sim.api.trainsim.Header\x12H\n\tjunctions\x18\x02 \x03(\x0b\x32\x35.risenlighten.lasvsim.train_sim.api.trainsim.Junction\x12\x46\n\x08segments\x18\x03 \x03(\x0b\x32\x34.risenlighten.lasvsim.train_sim.api.trainsim.Segment\x12@\n\x05links\x18\x04 \x03(\x0b\x32\x31.risenlighten.lasvsim.train_sim.api.trainsim.Link\x12@\n\x05lanes\x18\x05 \x03(\x0b\x32\x31.risenlighten.lasvsim.train_sim.api.trainsim.Lane\x12K\n\ncrosswalks\x18\xe8\x07 \x03(\x0b\x32\x36.risenlighten.lasvsim.train_sim.api.trainsim.Crosswalk\x12I\n\tstoplines\x18\xe9\x07 \x03(\x0b\x32\x35.risenlighten.lasvsim.train_sim.api.trainsim.Stopline\x12R\n\x0etraffic_lights\x18\xea\x07 \x03(\x0b\x32\x39.risenlighten.lasvsim.train_sim.api.trainsim.TrafficLight\x12P\n\rtraffic_signs\x18\xeb\x07 \x03(\x0b\x32\x38.risenlighten.lasvsim.train_sim.api.trainsim.TrafficSign\x12I\n\tmovements\x18\xec\x07 \x03(\x0b\x32\x35.risenlighten.lasvsim.train_sim.api.trainsim.Movement\x12M\n\x0b\x63onnections\x18\xed\x07 \x03(\x0b\x32\x37.risenlighten.lasvsim.train_sim.api.trainsim.Connection\"L\n\x0bObjBaseInfo\x12\r\n\x05width\x18\x01 \x01(\x01\x12\x0e\n\x06height\x18\x02 \x01(\x01\x12\x0e\n\x06length\x18\x03 \x01(\x01\x12\x0e\n\x06weight\x18\x04 \x01(\x01\"\xa4\x01\n\x0b\x44ynamicInfo\x12\x1d\n\x15\x66ront_wheel_stiffness\x18\x01 \x01(\x01\x12\x1c\n\x14rear_wheel_stiffness\x18\x02 \x01(\x01\x12\x1c\n\x14\x66ront_axle_to_center\x18\x03 \x01(\x01\x12\x1b\n\x13rear_axle_to_center\x18\x04 \x01(\x01\x12\x1d\n\x15yaw_moment_of_inertia\x18\x05 \x01(\x01\"]\n\rObjMovingInfo\x12\t\n\x01u\x18\x02 \x01(\x01\x12\r\n\x05u_acc\x18\x03 \x01(\x01\x12\t\n\x01v\x18\x04 \x01(\x01\x12\r\n\x05v_acc\x18\x05 \x01(\x01\x12\t\n\x01w\x18\x06 \x01(\x01\x12\r\n\x05w_acc\x18\x07 \x01(\x01\"}\n\x0b\x43ontrolInfo\x12\x11\n\tste_wheel\x18\x01 \x01(\x01\x12\x0f\n\x07lon_acc\x18\x02 \x01(\x01\x12\x11\n\tfl_torque\x18\x03 \x01(\x01\x12\x11\n\tfr_torque\x18\x04 \x01(\x01\x12\x11\n\trl_torque\x18\x05 \x01(\x01\x12\x11\n\trr_torque\x18\x06 \x01(\x01\"\x9f\x01\n\rReferenceLine\x12\x10\n\x08lane_ids\x18\x01 \x03(\t\x12\x12\n\nlane_types\x18\x02 \x03(\t\x12\x42\n\x06points\x18\x03 \x03(\x0b\x32\x32.risenlighten.lasvsim.train_sim.api.trainsim.Point\x12\x12\n\nlane_idxes\x18\x04 \x03(\x05\x12\x10\n\x08opposite\x18\x05 \x01(\x08\"\xc9\x01\n\x0eNavigationInfo\x12\x11\n\troute_nav\x18\x01 \x03(\t\x12\x10\n\x08link_nav\x18\x02 \x03(\t\x12\x46\n\x08lane_nav\x18\x03 \x03(\x0b\x32\x34.risenlighten.lasvsim.train_sim.api.trainsim.LaneNav\x12J\n\x0b\x64\x65stination\x18\x04 \x01(\x0b\x32\x35.risenlighten.lasvsim.train_sim.api.trainsim.Position\"\x81\x01\n\x07LaneNav\x12J\n\x03nav\x18\x01 \x03(\x0b\x32=.risenlighten.lasvsim.train_sim.api.trainsim.LaneNav.NavEntry\x1a*\n\x08NavEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x42x\n4proto.risenlighten.lasvsim.train_sim.api.trainsim.v1P\x01Z>git.risenlighten.com/lasvsim/train_sim/api/trainsim;trainsimv1b\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'risenlighten.lasvsim.train_sim.api.trainsim.external_struct_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n4proto.risenlighten.lasvsim.train_sim.api.trainsim.v1P\001Z>git.risenlighten.com/lasvsim/train_sim/api/trainsim;trainsimv1'
  _globals['_TURN_TURNMAPPINGENTRY']._options = None
  _globals['_TURN_TURNMAPPINGENTRY']._serialized_options = b'8\001'
  _globals['_TRAFFICLIGHT_MOVEMENTSIGNALSENTRY']._options = None
  _globals['_TRAFFICLIGHT_MOVEMENTSIGNALSENTRY']._serialized_options = b'8\001'
  _globals['_LANENAV_NAVENTRY']._options = None
  _globals['_LANENAV_NAVENTRY']._serialized_options = b'8\001'
  _globals['_POINT']._serialized_start=114
  _globals['_POINT']._serialized_end=154
  _globals['_HEADER']._serialized_start=157
  _globals['_HEADER']._serialized_end=346
  _globals['_POSITION']._serialized_start=349
  _globals['_POSITION']._serialized_end=562
  _globals['_DIRECTIONPOINT']._serialized_start=565
  _globals['_DIRECTIONPOINT']._serialized_end=700
  _globals['_TURN']._serialized_start=703
  _globals['_TURN']._serialized_end=951
  _globals['_TURN_TURNMAPPINGENTRY']._serialized_start=901
  _globals['_TURN_TURNMAPPINGENTRY']._serialized_end=951
  _globals['_SPEED']._serialized_start=953
  _globals['_SPEED']._serialized_end=1035
  _globals['_LANEMARK']._serialized_start=1038
  _globals['_LANEMARK']._serialized_end=1208
  _globals['_LANEMARKATTRIBUTION']._serialized_start=1211
  _globals['_LANEMARKATTRIBUTION']._serialized_end=1347
  _globals['_JUNCTION']._serialized_start=1350
  _globals['_JUNCTION']._serialized_end=1585
  _globals['_SEGMENT']._serialized_start=1588
  _globals['_SEGMENT']._serialized_end=1807
  _globals['_LINK']._serialized_start=1810
  _globals['_LINK']._serialized_end=2362
  _globals['_LANE']._serialized_start=2365
  _globals['_LANE']._serialized_end=2975
  _globals['_STOPLINE']._serialized_start=2977
  _globals['_STOPLINE']._serialized_end=3091
  _globals['_TRAFFICSIGN']._serialized_start=3094
  _globals['_TRAFFICSIGN']._serialized_end=3248
  _globals['_TRAFFICLIGHT']._serialized_start=3251
  _globals['_TRAFFICLIGHT']._serialized_end=3896
  _globals['_TRAFFICLIGHT_MOVEMENTSIGNAL']._serialized_start=3487
  _globals['_TRAFFICLIGHT_MOVEMENTSIGNAL']._serialized_end=3765
  _globals['_TRAFFICLIGHT_MOVEMENTSIGNAL_SIGNALOFGREEN']._serialized_start=3670
  _globals['_TRAFFICLIGHT_MOVEMENTSIGNAL_SIGNALOFGREEN']._serialized_end=3765
  _globals['_TRAFFICLIGHT_MOVEMENTSIGNALSENTRY']._serialized_start=3768
  _globals['_TRAFFICLIGHT_MOVEMENTSIGNALSENTRY']._serialized_end=3896
  _globals['_MOVEMENT']._serialized_start=3899
  _globals['_MOVEMENT']._serialized_end=4045
  _globals['_CONNECTION']._serialized_start=4048
  _globals['_CONNECTION']._serialized_end=4353
  _globals['_CROSSWALK']._serialized_start=4356
  _globals['_CROSSWALK']._serialized_end=4490
  _globals['_HDMAP']._serialized_start=4493
  _globals['_HDMAP']._serialized_end=5319
  _globals['_OBJBASEINFO']._serialized_start=5321
  _globals['_OBJBASEINFO']._serialized_end=5397
  _globals['_DYNAMICINFO']._serialized_start=5400
  _globals['_DYNAMICINFO']._serialized_end=5564
  _globals['_OBJMOVINGINFO']._serialized_start=5566
  _globals['_OBJMOVINGINFO']._serialized_end=5659
  _globals['_CONTROLINFO']._serialized_start=5661
  _globals['_CONTROLINFO']._serialized_end=5786
  _globals['_REFERENCELINE']._serialized_start=5789
  _globals['_REFERENCELINE']._serialized_end=5948
  _globals['_NAVIGATIONINFO']._serialized_start=5951
  _globals['_NAVIGATIONINFO']._serialized_end=6152
  _globals['_LANENAV']._serialized_start=6155
  _globals['_LANENAV']._serialized_end=6284
  _globals['_LANENAV_NAVENTRY']._serialized_start=6242
  _globals['_LANENAV_NAVENTRY']._serialized_end=6284
# @@protoc_insertion_point(module_scope)
