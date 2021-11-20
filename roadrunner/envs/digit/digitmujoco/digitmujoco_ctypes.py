# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/usr/include/clang/6.0/include', '-Iinclude']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes
import os
_dir_path = os.path.dirname(os.path.realpath(__file__))


_libraries = {}
_libraries['./libdigitmujoco.so'] = ctypes.CDLL(_dir_path + '/libdigitmujoco.so')
# if local wordsize is same as target, keep ctypes pointer function.
if ctypes.sizeof(ctypes.c_void_p) == 8:
    POINTER_T = ctypes.POINTER
else:
    # required to access _ctypes
    import _ctypes
    # Emulate a pointer class using the approriate c_int32/c_int64 type
    # The new class should have :
    # ['__module__', 'from_param', '_type_', '__dict__', '__weakref__', '__doc__']
    # but the class should be submitted to a unique instance for each base type
    # to that if A == B, POINTER_T(A) == POINTER_T(B)
    ctypes._pointer_t_type_cache = {}
    def POINTER_T(pointee):
        # a pointer should have the same length as LONG
        fake_ptr_base_type = ctypes.c_uint64 
        # specific case for c_void_p
        if pointee is None: # VOID pointer type. c_void_p.
            pointee = type(None) # ctypes.c_void_p # ctypes.c_ulong
            clsname = 'c_void'
        else:
            clsname = pointee.__name__
        if clsname in ctypes._pointer_t_type_cache:
            return ctypes._pointer_t_type_cache[clsname]
        # make template
        class _T(_ctypes._SimpleCData,):
            _type_ = 'L'
            _subtype_ = pointee
            def _sub_addr_(self):
                return self.value
            def __repr__(self):
                return '%s(%d)'%(clsname, self.value)
            def contents(self):
                raise TypeError('This is not a ctypes pointer.')
            def __init__(self, **args):
                raise TypeError('This is not a ctypes pointer. It is not instanciable.')
        _class = type('LP_%d_%s'%(8, clsname), (_T,),{}) 
        ctypes._pointer_t_type_cache[clsname] = _class
        return _class

c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



size_t = ctypes.c_uint64
socklen_t = ctypes.c_uint32
class struct_sockaddr(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('sa_family', ctypes.c_uint16),
    ('sa_data', ctypes.c_char * 14),
     ]

ssize_t = ctypes.c_int64

class struct_c__SA_digit_out_t(ctypes.Structure):
    pass

class struct_c__SA_digit_in_t(ctypes.Structure):
    pass

"""
"""
class struct_c__SA_elmo_in_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('controlWord', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 6),
    ('torque', ctypes.c_double),
     ]

elmo_in_t = struct_c__SA_elmo_in_t
class struct_c__SA_digit_leg_in_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('hipRollDrive', elmo_in_t),
    ('hipYawDrive', elmo_in_t),
    ('hipPitchDrive', elmo_in_t),
    ('kneeDrive', elmo_in_t),
    ('toeADrive', elmo_in_t),
    ('toeBDrive', elmo_in_t),
     ]

class struct_c__SA_digit_arm_in_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('shoulderRollDrive', elmo_in_t),
    ('shoulderYawDrive', elmo_in_t),
    ('shoulderPitchDrive', elmo_in_t),
    ('elbowDrive', elmo_in_t),
     ]

digit_leg_in_t = struct_c__SA_digit_leg_in_t
digit_arm_in_t = struct_c__SA_digit_arm_in_t
class struct_c__SA_radio_in_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('channel', ctypes.c_int16 * 14),
     ]

radio_in_t = struct_c__SA_radio_in_t
class struct_c__SA_digit_torso_in_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('radio', radio_in_t),
    ('sto', ctypes.c_bool),
    ('piezoState', ctypes.c_bool),
    ('piezoTone', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
     ]

digit_torso_in_t = struct_c__SA_digit_torso_in_t
struct_c__SA_digit_in_t._pack_ = True # source:False
struct_c__SA_digit_in_t._fields_ = [
    ('torso', digit_torso_in_t),
    ('leftLeg', digit_leg_in_t),
    ('rightLeg', digit_leg_in_t),
    ('leftArm', digit_arm_in_t),
    ('rightArm', digit_arm_in_t),
]

digit_in_t = struct_c__SA_digit_in_t
DiagnosticCodes = ctypes.c_int16
class struct_c__SA_battery_out_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('dataGood', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('stateOfCharge', ctypes.c_double),
    ('voltage', ctypes.c_double * 12),
    ('current', ctypes.c_double),
    ('temperature', ctypes.c_double * 4),
     ]

battery_out_t = struct_c__SA_battery_out_t
class struct_c__SA_digit_joint_out_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('position', ctypes.c_double),
    ('velocity', ctypes.c_double),
     ]

digit_joint_out_t = struct_c__SA_digit_joint_out_t
class struct_c__SA_elmo_out_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('statusWord', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 6),
    ('position', ctypes.c_double),
    ('velocity', ctypes.c_double),
    ('torque', ctypes.c_double),
    ('driveTemperature', ctypes.c_double),
    ('dcLinkVoltage', ctypes.c_double),
    ('torqueLimit', ctypes.c_double),
    ('gearRatio', ctypes.c_double),
     ]

elmo_out_t = struct_c__SA_elmo_out_t
class struct_c__SA_digit_leg_out_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('hipRollDrive', elmo_out_t),
    ('hipYawDrive', elmo_out_t),
    ('hipPitchDrive', elmo_out_t),
    ('kneeDrive', elmo_out_t),
    ('toeADrive', elmo_out_t),
    ('toeBDrive', elmo_out_t),
    ('shinJoint', digit_joint_out_t),
    ('tarsusJoint', digit_joint_out_t),
    ('medullaCounter', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('medullaCpuLoad', ctypes.c_uint16),
    ('reedSwitchState', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
     ]

digit_leg_out_t = struct_c__SA_digit_leg_out_t
class struct_c__SA_digit_arm_out_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('shoulderRollDrive', elmo_out_t),
    ('shoulderYawDrive', elmo_out_t),
    ('shoulderitchDrive', elmo_out_t),
    ('elbowDrive', elmo_out_t),
    ('medullaCounter', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('medullaCpuLoad', ctypes.c_uint16),
    ('reedSwitchState', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
     ]

digit_arm_out_t = struct_c__SA_digit_arm_out_t
class struct_c__SA_radio_out_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('radioReceiverSignalGood', ctypes.c_bool),
    ('receiverMedullaSignalGood', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 6),
    ('channel', ctypes.c_double * 16),
     ]

radio_out_t = struct_c__SA_radio_out_t
class struct_c__SA_target_pc_out_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('etherCatStatus', ctypes.c_int32 * 6),
    ('etherCatNotifications', ctypes.c_int32 * 21),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('taskExecutionTime', ctypes.c_double),
    ('overloadCounter', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('cpuTemperature', ctypes.c_double),
     ]

target_pc_out_t = struct_c__SA_target_pc_out_t
class struct_c__SA_vectornav_out_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('dataGood', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
    ('vpeStatus', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('pressure', ctypes.c_double),
    ('temperature', ctypes.c_double),
    ('magneticField', ctypes.c_double * 3),
    ('angularVelocity', ctypes.c_double * 3),
    ('linearAcceleration', ctypes.c_double * 3),
    ('orientation', ctypes.c_double * 4),
     ]

vectornav_out_t = struct_c__SA_vectornav_out_t
class struct_c__SA_digit_torso_out_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('targetPc', target_pc_out_t),
    ('battery', battery_out_t),
    ('radio', radio_out_t),
    ('vectorNav', vectornav_out_t),
    ('medullaCounter', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('medullaCpuLoad', ctypes.c_uint16),
    ('bleederState', ctypes.c_bool),
    ('leftReedSwitchState', ctypes.c_bool),
    ('rightReedSwitchState', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte),
    ('vtmTemperature', ctypes.c_double),
     ]

digit_torso_out_t = struct_c__SA_digit_torso_out_t
struct_c__SA_digit_out_t._pack_ = True # source:False
struct_c__SA_digit_out_t._fields_ = [
    ('torso', digit_torso_out_t),
    ('leftLeg', digit_leg_out_t),
    ('rightLeg', digit_leg_out_t),
    ('leftArm', digit_arm_out_t),
    ('rightArm', digit_arm_out_t),
    ('motor_pos', ctypes.c_double * 20),
    ('motor_vel', ctypes.c_double * 20),
    ('joint_pos', ctypes.c_double * 4),
    ('joint_vel', ctypes.c_double * 4),
    ('isCalibrated', ctypes.c_bool),
    #('PADDING_0', ctypes.c_ubyte),
    ('messages', ctypes.c_int16 * 4),
    #('PADDING_1', ctypes.c_ubyte * 6),
]

digit_out_t = struct_c__SA_digit_out_t
class struct_digit_sim(ctypes.Structure):
    pass

digit_sim_t = struct_digit_sim
class struct_digit_vis(ctypes.Structure):
    pass

digit_vis_t = struct_digit_vis
digit_mujoco_init = _libraries['./libdigitmujoco.so'].digit_mujoco_init
digit_mujoco_init.restype = ctypes.c_bool
digit_mujoco_init.argtypes = [POINTER_T(ctypes.c_char)]
digit_sim_init = _libraries['./libdigitmujoco.so'].digit_sim_init
digit_sim_init.restype = POINTER_T(struct_digit_sim)
digit_sim_init.argtypes = [ctypes.c_char_p, ctypes.c_bool]
digit_sim_out = _libraries['./libdigitmujoco.so'].digit_sim_out
digit_sim_out.restype = POINTER_T(digit_out_t)
digit_sim_out.argtypes = [POINTER_T(struct_digit_sim)]

if False: # These functions TODO
    digit_cleanup = _libraries['./libdigitmujoco.so'].digit_cleanup
    digit_cleanup.restype = None
    digit_cleanup.argtypes = []
    digit_reload_xml = _libraries['./libdigitmujoco.so'].digit_reload_xml
    digit_reload_xml.restype = ctypes.c_bool
    digit_reload_xml.argtypes = [ctypes.c_char_p]
digit_sim_duplicate = _libraries['./libdigitmujoco.so'].digit_sim_duplicate
digit_sim_duplicate.restype = POINTER_T(struct_digit_sim)
digit_sim_duplicate.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_copy = _libraries['./libdigitmujoco.so'].digit_sim_copy
digit_sim_copy.restype = None
digit_sim_copy.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(struct_digit_sim)]
digit_sim_free = _libraries['./libdigitmujoco.so'].digit_sim_free
digit_sim_free.restype = None
digit_sim_free.argtypes = [POINTER_T(struct_digit_sim)]

"""
# this probably shouldn't be available to the user
digit_sim_step_ethercat = _libraries['./libdigitmujoco.so'].digit_sim_step_ethercat
digit_sim_step_ethercat.restype = None
digit_sim_step_ethercat.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(struct_c__SA_digit_out_t), POINTER_T(struct_c__SA_digit_in_t)]
"""

class struct_c__SA_pd_in_t(ctypes.Structure):
    pass

class struct_c__SA_pd_in_t(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('torque', ctypes.c_double * 20),
    ('pTarget', ctypes.c_double * 20),
    ('dTarget', ctypes.c_double * 20),
    ('pGain', ctypes.c_double * 20),
    ('dGain', ctypes.c_double * 20),
    ('telemetry', ctypes.c_double * 9),
     ]
pd_in_t = struct_c__SA_pd_in_t


digit_sim_step_pd = _libraries['./libdigitmujoco.so'].digit_sim_step_pd
digit_sim_step_pd.restype = None
#digit_sim_step_pd.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(struct_c__SA_digit_out_t), POINTER_T(struct_c__SA_pd_in_t)]
digit_sim_step_pd.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(struct_c__SA_pd_in_t)]
digit_sim_time = _libraries['./libdigitmujoco.so'].digit_sim_time
digit_sim_time.restype = POINTER_T(ctypes.c_double)
digit_sim_time.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_qpos = _libraries['./libdigitmujoco.so'].digit_sim_qpos
digit_sim_qpos.restype = POINTER_T(ctypes.c_double)
digit_sim_qpos.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_qvel = _libraries['./libdigitmujoco.so'].digit_sim_qvel
digit_sim_qvel.restype = POINTER_T(ctypes.c_double)
digit_sim_qvel.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_qacc = _libraries['./libdigitmujoco.so'].digit_sim_qacc
digit_sim_qacc.restype = POINTER_T(ctypes.c_double)
digit_sim_qacc.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_mjmodel = _libraries['./libdigitmujoco.so'].digit_sim_mjmodel
digit_sim_mjmodel.restype = POINTER_T(None)
digit_sim_mjmodel.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_mjdata = _libraries['./libdigitmujoco.so'].digit_sim_mjdata
digit_sim_mjdata.restype = POINTER_T(None)
digit_sim_mjdata.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_foot_forces = _libraries['./libdigitmujoco.so'].digit_sim_foot_forces
digit_sim_foot_forces.restype = None
digit_sim_foot_forces.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_double * 12]
digit_sim_foot_positions = _libraries['./libdigitmujoco.so'].digit_sim_foot_positions
digit_sim_foot_positions.restype = None
digit_sim_foot_positions.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_double * 6]
if False: # These functions TODO
    digit_sim_check_obstacle_collision = _libraries['./libdigitmujoco.so'].digit_sim_check_obstacle_collision
    digit_sim_check_obstacle_collision.restype = ctypes.c_bool
    digit_sim_check_obstacle_collision.argtypes = [POINTER_T(struct_digit_sim)]
    digit_sim_check_self_collision = _libraries['./libdigitmujoco.so'].digit_sim_check_self_collision
    digit_sim_check_self_collision.restype = ctypes.c_bool
    digit_sim_check_self_collision.argtypes = [POINTER_T(struct_digit_sim)]
    digit_sim_foot_velocities = _libraries['./libdigitmujoco.so'].digit_sim_foot_velocities
    digit_sim_foot_velocities.restype = None
    digit_sim_foot_velocities.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_double * 12]
    digit_sim_foot_quat = _libraries['./libdigitmujoco.so'].digit_sim_foot_orient
    digit_sim_foot_quat.restype = None
    digit_sim_foot_quat.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_double * 4]
    digit_sim_body_vel = _libraries['./libdigitmujoco.so'].digit_sim_body_velocities
    digit_sim_body_vel.restype = None
    digit_sim_body_vel.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_double * 6, ctypes.c_char_p]
    digit_sim_apply_force = _libraries['./libdigitmujoco.so'].digit_sim_apply_force
    digit_sim_apply_force.restype = None
    digit_sim_apply_force.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_double * 6, ctypes.c_char_p]
    digit_sim_clear_forces = _libraries['./libdigitmujoco.so'].digit_sim_clear_forces
    digit_sim_clear_forces.restype = None
    digit_sim_clear_forces.argtypes = [POINTER_T(struct_digit_sim)]
    digit_sim_hold = _libraries['./libdigitmujoco.so'].digit_sim_hold
    digit_sim_hold.restype = None
    digit_sim_hold.argtypes = [POINTER_T(struct_digit_sim)]
    digit_sim_release = _libraries['./libdigitmujoco.so'].digit_sim_release
    digit_sim_release.restype = None
    digit_sim_release.argtypes = [POINTER_T(struct_digit_sim)]
    digit_sim_radio = _libraries['./libdigitmujoco.so'].digit_sim_radio
    digit_sim_radio.restype = None
    digit_sim_radio.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_double * 16]

digit_sim_full_reset = _libraries['./libdigitmujoco.so'].digit_sim_full_reset
digit_sim_full_reset.restype = None
digit_sim_full_reset.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_get_hfield_nrow = _libraries['./libdigitmujoco.so'].digit_sim_get_hfield_nrow
digit_sim_get_hfield_nrow.restype = ctypes.c_int32
digit_sim_get_hfield_nrow.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_get_hfield_ncol = _libraries['./libdigitmujoco.so'].digit_sim_get_hfield_ncol
digit_sim_get_hfield_ncol.restype = ctypes.c_int32
digit_sim_get_hfield_ncol.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_get_nhfielddata = _libraries['./libdigitmujoco.so'].digit_sim_get_nhfielddata
digit_sim_get_nhfielddata.restype = ctypes.c_int32
digit_sim_get_nhfielddata.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_get_hfield_size = _libraries['./libdigitmujoco.so'].digit_sim_get_hfield_size
digit_sim_get_hfield_size.restype = POINTER_T(ctypes.c_double)
digit_sim_get_hfield_size.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_set_hfield_size = _libraries['./libdigitmujoco.so'].digit_sim_set_hfield_size
digit_sim_set_hfield_size.restype = None
digit_sim_set_hfield_size.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_double * 4]
digit_sim_hfielddata = _libraries['./libdigitmujoco.so'].digit_sim_hfielddata
digit_sim_hfielddata.restype = POINTER_T(ctypes.c_float)
digit_sim_hfielddata.argtypes = [POINTER_T(struct_digit_sim)]
digit_sim_set_hfielddata = _libraries['./libdigitmujoco.so'].digit_sim_set_hfielddata
digit_sim_set_hfielddata.restype = None
digit_sim_set_hfielddata.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(ctypes.c_float)]

digit_sim_xquat = _libraries['./libdigitmujoco.so'].digit_sim_xquat
digit_sim_xquat.restype = POINTER_T(ctypes.c_double)
digit_sim_xquat.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p]
digit_vis_init = _libraries['./libdigitmujoco.so'].digit_vis_init
digit_vis_init.restype = POINTER_T(struct_digit_vis)
digit_vis_init.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p]
digit_vis_close = _libraries['./libdigitmujoco.so'].digit_vis_close
digit_vis_close.restype = None
digit_vis_close.argtypes = [POINTER_T(struct_digit_vis)]
digit_vis_free = _libraries['./libdigitmujoco.so'].digit_vis_free
digit_vis_free.restype = None
digit_vis_free.argtypes = [POINTER_T(struct_digit_vis)]
digit_vis_draw = _libraries['./libdigitmujoco.so'].digit_vis_draw
digit_vis_draw.restype = ctypes.c_bool
digit_vis_draw.argtypes = [POINTER_T(struct_digit_vis), POINTER_T(struct_digit_sim)]
digit_vis_set_cam = _libraries['./libdigitmujoco.so'].digit_vis_set_cam
digit_vis_set_cam.restype = None
digit_vis_set_cam.argtypes = [POINTER_T(struct_digit_vis), ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
digit_vis_valid = _libraries['./libdigitmujoco.so'].digit_vis_valid
digit_vis_valid.restype = ctypes.c_bool
digit_vis_valid.argtypes = [POINTER_T(struct_digit_vis)]
digit_vis_paused = _libraries['./libdigitmujoco.so'].digit_vis_paused
digit_vis_paused.restype = ctypes.c_bool
digit_vis_paused.argtypes = [POINTER_T(struct_digit_vis)]
digit_vis_apply_force = _libraries['./libdigitmujoco.so'].digit_vis_apply_force
digit_vis_apply_force.restype = None
digit_vis_apply_force.argtypes = [POINTER_T(struct_digit_vis), POINTER_T(ctypes.c_double), ctypes.c_char_p]
digit_vis_full_reset = _libraries['./libdigitmujoco.so'].digit_vis_full_reset
digit_vis_full_reset.restype = None
digit_vis_full_reset.argtypes = [POINTER_T(struct_digit_vis)]

digit_sim_read_rangefinder = _libraries['./libdigitmujoco.so'].digit_sim_read_rangefinder
digit_sim_read_rangefinder.restype = None
digit_sim_read_rangefinder.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_double * 6]

digit_sim_dof_damping = _libraries['./libdigitmujoco.so'].digit_sim_dof_damping
digit_sim_dof_damping.restype = POINTER_T(ctypes.c_double)
digit_sim_dof_damping.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_set_dof_damping = _libraries['./libdigitmujoco.so'].digit_sim_set_dof_damping
digit_sim_set_dof_damping.restype = None
digit_sim_set_dof_damping.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(ctypes.c_double)]

digit_sim_body_mass = _libraries['./libdigitmujoco.so'].digit_sim_body_mass
digit_sim_body_mass.restype = POINTER_T(ctypes.c_double)
digit_sim_body_mass.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_set_body_mass = _libraries['./libdigitmujoco.so'].digit_sim_set_body_mass
digit_sim_set_body_mass.restype = None
digit_sim_set_body_mass.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(ctypes.c_double)]

digit_sim_set_body_name_mass = _libraries['./libdigitmujoco.so'].digit_sim_set_body_name_mass
digit_sim_set_body_name_mass.restype = None
digit_sim_set_body_name_mass.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p, ctypes.c_double]

digit_sim_body_ipos = _libraries['./libdigitmujoco.so'].digit_sim_body_ipos
digit_sim_body_ipos.restype = POINTER_T(ctypes.c_double)
digit_sim_body_ipos.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_set_body_ipos = _libraries['./libdigitmujoco.so'].digit_sim_set_body_ipos
digit_sim_set_body_ipos.restype = None
digit_sim_set_body_ipos.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(ctypes.c_double)]

digit_sim_geom_friction = _libraries['./libdigitmujoco.so'].digit_sim_geom_friction
digit_sim_geom_friction.restype = POINTER_T(ctypes.c_double)
digit_sim_geom_friction.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_set_geom_friction = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_friction
digit_sim_set_geom_friction.restype = None
digit_sim_set_geom_friction.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(ctypes.c_double)]

digit_sim_set_geom_name_friction = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_name_friction
digit_sim_set_geom_name_friction.restype = None
digit_sim_set_geom_name_friction.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p, POINTER_T(ctypes.c_double)]

digit_sim_geom_rgba = _libraries['./libdigitmujoco.so'].digit_sim_geom_rgba
digit_sim_geom_rgba.restype = POINTER_T(ctypes.c_float)
digit_sim_geom_rgba.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_geom_name_rgba = _libraries['./libdigitmujoco.so'].digit_sim_geom_name_rgba
digit_sim_geom_name_rgba.restype = POINTER_T(ctypes.c_float)
digit_sim_geom_name_rgba.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p]

digit_sim_set_geom_rgba = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_rgba
digit_sim_set_geom_rgba.restype = None
digit_sim_set_geom_rgba.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(ctypes.c_float)]

digit_sim_set_geom_name_rgba = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_name_rgba
digit_sim_set_geom_name_rgba.restype = None
digit_sim_set_geom_name_rgba.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p, POINTER_T(ctypes.c_float)]

digit_sim_geom_quat = _libraries['./libdigitmujoco.so'].digit_sim_geom_quat
digit_sim_geom_quat.restype = POINTER_T(ctypes.c_double)
digit_sim_geom_quat.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_geom_name_quat = _libraries['./libdigitmujoco.so'].digit_sim_geom_name_quat
digit_sim_geom_name_quat.restype = POINTER_T(ctypes.c_double)
digit_sim_geom_name_quat.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p]

digit_sim_set_geom_quat = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_quat
digit_sim_set_geom_quat.restype = None
digit_sim_set_geom_quat.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(ctypes.c_double)]

digit_sim_set_geom_name_quat = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_name_quat
digit_sim_set_geom_name_quat.restype = None
digit_sim_set_geom_name_quat.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p, POINTER_T(ctypes.c_double)]

digit_sim_geom_pos = _libraries['./libdigitmujoco.so'].digit_sim_geom_pos
digit_sim_geom_pos.restype = POINTER_T(ctypes.c_double)
digit_sim_geom_pos.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_geom_name_pos = _libraries['./libdigitmujoco.so'].digit_sim_geom_name_pos
digit_sim_geom_name_pos.restype = POINTER_T(ctypes.c_double)
digit_sim_geom_name_pos.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p]

digit_sim_set_geom_pos = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_pos
digit_sim_set_geom_pos.restype = None
digit_sim_set_geom_pos.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(ctypes.c_double)]

digit_sim_set_geom_name_pos = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_name_pos
digit_sim_set_geom_name_pos.restype = None
digit_sim_set_geom_name_pos.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p, POINTER_T(ctypes.c_double)]

digit_sim_geom_size = _libraries['./libdigitmujoco.so'].digit_sim_geom_size
digit_sim_geom_size.restype = POINTER_T(ctypes.c_double)
digit_sim_geom_size.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_geom_name_size = _libraries['./libdigitmujoco.so'].digit_sim_geom_name_size
digit_sim_geom_name_size.restype = POINTER_T(ctypes.c_double)
digit_sim_geom_name_size.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p]

digit_sim_set_geom_size = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_size
digit_sim_set_geom_size.restype = None
digit_sim_set_geom_size.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(ctypes.c_double)]

digit_sim_set_geom_name_size = _libraries['./libdigitmujoco.so'].digit_sim_set_geom_name_size
digit_sim_set_geom_name_size.restype = None
digit_sim_set_geom_name_size.argtypes = [POINTER_T(struct_digit_sim), ctypes.c_char_p, POINTER_T(ctypes.c_double)]

digit_sim_set_const = _libraries['./libdigitmujoco.so'].digit_sim_set_const
digit_sim_set_const.restype = None
digit_sim_set_const.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_nv = _libraries['./libdigitmujoco.so'].digit_sim_nv
digit_sim_nv.restype = ctypes.c_int32
digit_sim_nv.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_nbody = _libraries['./libdigitmujoco.so'].digit_sim_nbody
digit_sim_nbody.restype = ctypes.c_int32
digit_sim_nbody.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_ngeom = _libraries['./libdigitmujoco.so'].digit_sim_ngeom
digit_sim_ngeom.restype = ctypes.c_int32
digit_sim_ngeom.argtypes = [POINTER_T(struct_digit_sim)]

digit_sim_nq = _libraries['./libdigitmujoco.so'].digit_sim_nq
digit_sim_nq.restype = ctypes.c_int32
digit_sim_nq.argtypes = [POINTER_T(struct_digit_sim)]

if False: # Code from this point on is obsolete due to not using proprietary agility code, but we might want to use later
    class struct_c__SA_state_out_t(ctypes.Structure):
        pass

    digit_core_sim_step = _libraries['./libdigitmujoco.so'].digit_core_sim_step
    digit_core_sim_step.restype = None
    digit_core_sim_step.argtypes = [POINTER_T(struct_CassieCoreSim), POINTER_T(struct_c__SA_digit_user_in_t), POINTER_T(struct_c__SA_digit_out_t), POINTER_T(struct_c__SA_digit_in_t)]
    pack_digit_in_t = _libraries['./libdigitmujoco.so'].pack_digit_in_t
    pack_digit_in_t.restype = None
    pack_digit_in_t.argtypes = [POINTER_T(struct_c__SA_digit_in_t), POINTER_T(ctypes.c_ubyte)]
    unpack_digit_in_t = _libraries['./libdigitmujoco.so'].unpack_digit_in_t
    unpack_digit_in_t.restype = None
    unpack_digit_in_t.argtypes = [POINTER_T(ctypes.c_ubyte), POINTER_T(struct_c__SA_digit_in_t)]
    class struct_CassieCoreSim(ctypes.Structure):
        pass

    digit_core_sim_t = struct_CassieCoreSim
    digit_core_sim_alloc = _libraries['./libdigitmujoco.so'].digit_core_sim_alloc
    digit_core_sim_alloc.restype = POINTER_T(struct_CassieCoreSim)
    digit_core_sim_alloc.argtypes = []
    digit_core_sim_copy = _libraries['./libdigitmujoco.so'].digit_core_sim_copy
    digit_core_sim_copy.restype = None
    digit_core_sim_copy.argtypes = [POINTER_T(struct_CassieCoreSim), POINTER_T(struct_CassieCoreSim)]
    digit_core_sim_free = _libraries['./libdigitmujoco.so'].digit_core_sim_free
    digit_core_sim_free.restype = None
    digit_core_sim_free.argtypes = [POINTER_T(struct_CassieCoreSim)]
    digit_core_sim_setup = _libraries['./libdigitmujoco.so'].digit_core_sim_setup
    digit_core_sim_setup.restype = None
    digit_core_sim_setup.argtypes = [POINTER_T(struct_CassieCoreSim)]
    class struct_c__SA_digit_user_in_t(ctypes.Structure):
        pass
    pack_digit_out_t = _libraries['./libdigitmujoco.so'].pack_digit_out_t
    pack_digit_out_t.restype = None
    pack_digit_out_t.argtypes = [POINTER_T(struct_c__SA_digit_out_t), POINTER_T(ctypes.c_ubyte)]
    unpack_digit_out_t = _libraries['./libdigitmujoco.so'].unpack_digit_out_t
    unpack_digit_out_t.restype = None
    unpack_digit_out_t.argtypes = [POINTER_T(ctypes.c_ubyte), POINTER_T(struct_c__SA_digit_out_t)]
    struct_c__SA_digit_user_in_t._pack_ = True # source:False
    struct_c__SA_digit_user_in_t._fields_ = [
        ('torque', ctypes.c_double * 10),
        ('telemetry', ctypes.c_int16 * 9),
        ('PADDING_0', ctypes.c_ubyte * 6),
    ]

    digit_user_in_t = struct_c__SA_digit_user_in_t
    pack_digit_user_in_t = _libraries['./libdigitmujoco.so'].pack_digit_user_in_t
    pack_digit_user_in_t.restype = None
    pack_digit_user_in_t.argtypes = [POINTER_T(struct_c__SA_digit_user_in_t), POINTER_T(ctypes.c_ubyte)]
    unpack_digit_user_in_t = _libraries['./libdigitmujoco.so'].unpack_digit_user_in_t
    unpack_digit_user_in_t.restype = None
    unpack_digit_user_in_t.argtypes = [POINTER_T(ctypes.c_ubyte), POINTER_T(struct_c__SA_digit_user_in_t)]
    class struct_digit_state(ctypes.Structure):
        pass

    digit_state_t = struct_digit_state
    digit_state_alloc = _libraries['./libdigitmujoco.so'].digit_state_alloc
    digit_state_alloc.restype = POINTER_T(struct_digit_state)
    digit_state_alloc.argtypes = []
    digit_state_duplicate = _libraries['./libdigitmujoco.so'].digit_state_duplicate
    digit_state_duplicate.restype = POINTER_T(struct_digit_state)
    digit_state_duplicate.argtypes = [POINTER_T(struct_digit_state)]
    digit_state_copy = _libraries['./libdigitmujoco.so'].digit_state_copy
    digit_state_copy.restype = None
    digit_state_copy.argtypes = [POINTER_T(struct_digit_state), POINTER_T(struct_digit_state)]
    digit_state_free = _libraries['./libdigitmujoco.so'].digit_state_free
    digit_state_free.restype = None
    digit_state_free.argtypes = [POINTER_T(struct_digit_state)]
    digit_state_time = _libraries['./libdigitmujoco.so'].digit_state_time
    digit_state_time.restype = POINTER_T(ctypes.c_double)
    digit_state_time.argtypes = [POINTER_T(struct_digit_state)]
    digit_state_qpos = _libraries['./libdigitmujoco.so'].digit_state_qpos
    digit_state_qpos.restype = POINTER_T(ctypes.c_double)
    digit_state_qpos.argtypes = [POINTER_T(struct_digit_state)]
    digit_state_qvel = _libraries['./libdigitmujoco.so'].digit_state_qvel
    digit_state_qvel.restype = POINTER_T(ctypes.c_double)
    digit_state_qvel.argtypes = [POINTER_T(struct_digit_state)]
    digit_get_state = _libraries['./libdigitmujoco.so'].digit_get_state
    digit_get_state.restype = None
    digit_get_state.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(struct_digit_state)]
    digit_set_state = _libraries['./libdigitmujoco.so'].digit_set_state
    digit_set_state.restype = None
    digit_set_state.argtypes = [POINTER_T(struct_digit_sim), POINTER_T(struct_digit_state)]
    pack_pd_in_t = _libraries['./libdigitmujoco.so'].pack_pd_in_t
    pack_pd_in_t.restype = None
    pack_pd_in_t.argtypes = [POINTER_T(struct_c__SA_pd_in_t), POINTER_T(ctypes.c_ubyte)]
    unpack_pd_in_t = _libraries['./libdigitmujoco.so'].unpack_pd_in_t
    unpack_pd_in_t.restype = None
    unpack_pd_in_t.argtypes = [POINTER_T(ctypes.c_ubyte), POINTER_T(struct_c__SA_pd_in_t)]
    class struct_c__SA_state_battery_out_t(ctypes.Structure):
        _pack_ = True # source:False
        _fields_ = [
        ('stateOfCharge', ctypes.c_double),
        ('current', ctypes.c_double),
         ]

    state_battery_out_t = struct_c__SA_state_battery_out_t
    class struct_c__SA_state_foot_out_t(ctypes.Structure):
        _pack_ = True # source:False
        _fields_ = [
        ('position', ctypes.c_double * 3),
        ('orientation', ctypes.c_double * 4),
        ('footRotationalVelocity', ctypes.c_double * 3),
        ('footTranslationalVelocity', ctypes.c_double * 3),
        ('toeForce', ctypes.c_double * 3),
        ('heelForce', ctypes.c_double * 3),
         ]

    state_foot_out_t = struct_c__SA_state_foot_out_t
    class struct_c__SA_state_joint_out_t(ctypes.Structure):
        _pack_ = True # source:False
        _fields_ = [
        ('position', ctypes.c_double * 6),
        ('velocity', ctypes.c_double * 6),
         ]

    state_joint_out_t = struct_c__SA_state_joint_out_t
    class struct_c__SA_state_motor_out_t(ctypes.Structure):
        _pack_ = True # source:False
        _fields_ = [
        ('position', ctypes.c_double * 10),
        ('velocity', ctypes.c_double * 10),
        ('torque', ctypes.c_double * 10),
         ]

    state_motor_out_t = struct_c__SA_state_motor_out_t
    class struct_c__SA_state_torso_out_t(ctypes.Structure):
        _pack_ = True # source:False
        _fields_ = [
        ('position', ctypes.c_double * 3),
        ('orientation', ctypes.c_double * 4),
        ('rotationalVelocity', ctypes.c_double * 3),
        ('translationalVelocity', ctypes.c_double * 3),
        ('translationalAcceleration', ctypes.c_double * 3),
        ('externalMoment', ctypes.c_double * 3),
        ('externalForce', ctypes.c_double * 3),
         ]

    state_torso_out_t = struct_c__SA_state_torso_out_t
    class struct_c__SA_state_radio_out_t(ctypes.Structure):
        _pack_ = True # source:False
        _fields_ = [
        ('channel', ctypes.c_double * 16),
        ('signalGood', ctypes.c_bool),
        ('PADDING_0', ctypes.c_ubyte * 7),
         ]

    state_radio_out_t = struct_c__SA_state_radio_out_t
    class struct_c__SA_state_terrain_out_t(ctypes.Structure):
        _pack_ = True # source:False
        _fields_ = [
        ('height', ctypes.c_double),
        ('slope', ctypes.c_double * 2),
         ]

    state_terrain_out_t = struct_c__SA_state_terrain_out_t
    struct_c__SA_state_out_t._pack_ = True # source:False
    struct_c__SA_state_out_t._fields_ = [
        ('torso', state_torso_out_t),
        ('leftFoot', state_foot_out_t),
        ('rightFoot', state_foot_out_t),
        ('terrain', state_terrain_out_t),
        ('motor', state_motor_out_t),
        ('joint', state_joint_out_t),
        ('radio', state_radio_out_t),
        ('battery', state_battery_out_t),
    ]

    state_out_t = struct_c__SA_state_out_t
    pack_state_out_t = _libraries['./libdigitmujoco.so'].pack_state_out_t
    pack_state_out_t.restype = None
    pack_state_out_t.argtypes = [POINTER_T(struct_c__SA_state_out_t), POINTER_T(ctypes.c_ubyte)]
    unpack_state_out_t = _libraries['./libdigitmujoco.so'].unpack_state_out_t
    unpack_state_out_t.restype = None
    unpack_state_out_t.argtypes = [POINTER_T(ctypes.c_ubyte), POINTER_T(struct_c__SA_state_out_t)]
    class struct_StateOutput(ctypes.Structure):
        pass

    state_output_t = struct_StateOutput
    state_output_alloc = _libraries['./libdigitmujoco.so'].state_output_alloc
    state_output_alloc.restype = POINTER_T(struct_StateOutput)
    state_output_alloc.argtypes = []
    state_output_copy = _libraries['./libdigitmujoco.so'].state_output_copy
    state_output_copy.restype = None
    state_output_copy.argtypes = [POINTER_T(struct_StateOutput), POINTER_T(struct_StateOutput)]
    state_output_free = _libraries['./libdigitmujoco.so'].state_output_free
    state_output_free.restype = None
    state_output_free.argtypes = [POINTER_T(struct_StateOutput)]
    state_output_setup = _libraries['./libdigitmujoco.so'].state_output_setup
    state_output_setup.restype = None
    state_output_setup.argtypes = [POINTER_T(struct_StateOutput)]
    state_output_step = _libraries['./libdigitmujoco.so'].state_output_step
    state_output_step.restype = None
    state_output_step.argtypes = [POINTER_T(struct_StateOutput), POINTER_T(struct_c__SA_digit_out_t), POINTER_T(struct_c__SA_state_out_t)]
    class struct_c__SA_packet_header_info_t(ctypes.Structure):
        _pack_ = True # source:False
        _fields_ = [
        ('seq_num_out', ctypes.c_char),
        ('seq_num_in_last', ctypes.c_char),
        ('delay', ctypes.c_char),
        ('seq_num_in_diff', ctypes.c_char),
         ]

    packet_header_info_t = struct_c__SA_packet_header_info_t
    process_packet_header = _libraries['./libdigitmujoco.so'].process_packet_header
    process_packet_header.restype = None
    process_packet_header.argtypes = [POINTER_T(struct_c__SA_packet_header_info_t), POINTER_T(ctypes.c_ubyte), POINTER_T(ctypes.c_ubyte)]
    udp_init_host = _libraries['./libdigitmujoco.so'].udp_init_host
    udp_init_host.restype = ctypes.c_int32
    udp_init_host.argtypes = [POINTER_T(ctypes.c_char), POINTER_T(ctypes.c_char)]
    udp_init_client = _libraries['./libdigitmujoco.so'].udp_init_client
    udp_init_client.restype = ctypes.c_int32
    udp_init_client.argtypes = [POINTER_T(ctypes.c_char), POINTER_T(ctypes.c_char), POINTER_T(ctypes.c_char), POINTER_T(ctypes.c_char)]
    udp_close = _libraries['./libdigitmujoco.so'].udp_close
    udp_close.restype = None
    udp_close.argtypes = [ctypes.c_int32]
    get_newest_packet = _libraries['./libdigitmujoco.so'].get_newest_packet
    get_newest_packet.restype = ssize_t
    get_newest_packet.argtypes = [ctypes.c_int32, POINTER_T(None), size_t, POINTER_T(struct_sockaddr), POINTER_T(ctypes.c_uint32)]
    wait_for_packet = _libraries['./libdigitmujoco.so'].wait_for_packet
    wait_for_packet.restype = ssize_t
    wait_for_packet.argtypes = [ctypes.c_int32, POINTER_T(None), size_t, POINTER_T(struct_sockaddr), POINTER_T(ctypes.c_uint32)]
    send_packet = _libraries['./libdigitmujoco.so'].send_packet
    send_packet.restype = ssize_t
    send_packet.argtypes = [ctypes.c_int32, POINTER_T(None), size_t, POINTER_T(struct_sockaddr), socklen_t]
