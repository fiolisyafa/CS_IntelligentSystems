       �K"	  @>~��Abrain.Event:2����     �c�	�b>~��A"��

z
input/XPlaceholder*
dtype0*/
_output_shapes
:���������22*$
shape:���������22
�
)Conv2D/W/Initializer/random_uniform/shapeConst*%
valueB"             *
_class
loc:@Conv2D/W*
dtype0*
_output_shapes
:
�
'Conv2D/W/Initializer/random_uniform/minConst*
valueB
 *�\��*
_class
loc:@Conv2D/W*
dtype0*
_output_shapes
: 
�
'Conv2D/W/Initializer/random_uniform/maxConst*
valueB
 *�\�>*
_class
loc:@Conv2D/W*
dtype0*
_output_shapes
: 
�
1Conv2D/W/Initializer/random_uniform/RandomUniformRandomUniform)Conv2D/W/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@Conv2D/W*
seed2 *
dtype0*&
_output_shapes
: 
�
'Conv2D/W/Initializer/random_uniform/subSub'Conv2D/W/Initializer/random_uniform/max'Conv2D/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D/W*
_output_shapes
: 
�
'Conv2D/W/Initializer/random_uniform/mulMul1Conv2D/W/Initializer/random_uniform/RandomUniform'Conv2D/W/Initializer/random_uniform/sub*
T0*
_class
loc:@Conv2D/W*&
_output_shapes
: 
�
#Conv2D/W/Initializer/random_uniformAdd'Conv2D/W/Initializer/random_uniform/mul'Conv2D/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D/W*&
_output_shapes
: 
�
Conv2D/W
VariableV2*
shape: *
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/W*
	container 
�
Conv2D/W/AssignAssignConv2D/W#Conv2D/W/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
q
Conv2D/W/readIdentityConv2D/W*
T0*
_class
loc:@Conv2D/W*&
_output_shapes
: 
�
Conv2D/b/Initializer/ConstConst*
valueB *    *
_class
loc:@Conv2D/b*
dtype0*
_output_shapes
: 
�
Conv2D/b
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/b*
	container 
�
Conv2D/b/AssignAssignConv2D/bConv2D/b/Initializer/Const*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/b
e
Conv2D/b/readIdentityConv2D/b*
T0*
_class
loc:@Conv2D/b*
_output_shapes
: 
�
Conv2D/Conv2DConv2Dinput/XConv2D/W/read*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������22 *
	dilations

�
Conv2D/BiasAddBiasAddConv2D/Conv2DConv2D/b/read*
T0*
data_formatNHWC*/
_output_shapes
:���������22 
]
Conv2D/ReluReluConv2D/BiasAdd*/
_output_shapes
:���������22 *
T0
�
MaxPool2D/MaxPoolMaxPoolConv2D/Relu*/
_output_shapes
:���������

 *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
+Conv2D_1/W/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"          @   *
_class
loc:@Conv2D_1/W
�
)Conv2D_1/W/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *��z�*
_class
loc:@Conv2D_1/W
�
)Conv2D_1/W/Initializer/random_uniform/maxConst*
valueB
 *��z=*
_class
loc:@Conv2D_1/W*
dtype0*
_output_shapes
: 
�
3Conv2D_1/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_1/W/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
: @*

seed *
T0*
_class
loc:@Conv2D_1/W
�
)Conv2D_1/W/Initializer/random_uniform/subSub)Conv2D_1/W/Initializer/random_uniform/max)Conv2D_1/W/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@Conv2D_1/W
�
)Conv2D_1/W/Initializer/random_uniform/mulMul3Conv2D_1/W/Initializer/random_uniform/RandomUniform)Conv2D_1/W/Initializer/random_uniform/sub*
T0*
_class
loc:@Conv2D_1/W*&
_output_shapes
: @
�
%Conv2D_1/W/Initializer/random_uniformAdd)Conv2D_1/W/Initializer/random_uniform/mul)Conv2D_1/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_1/W*&
_output_shapes
: @
�

Conv2D_1/W
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name *
_class
loc:@Conv2D_1/W*
	container *
shape: @
�
Conv2D_1/W/AssignAssign
Conv2D_1/W%Conv2D_1/W/Initializer/random_uniform*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
w
Conv2D_1/W/readIdentity
Conv2D_1/W*
T0*
_class
loc:@Conv2D_1/W*&
_output_shapes
: @
�
Conv2D_1/b/Initializer/ConstConst*
valueB@*    *
_class
loc:@Conv2D_1/b*
dtype0*
_output_shapes
:@
�

Conv2D_1/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_1/b*
	container 
�
Conv2D_1/b/AssignAssign
Conv2D_1/bConv2D_1/b/Initializer/Const*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
k
Conv2D_1/b/readIdentity
Conv2D_1/b*
T0*
_class
loc:@Conv2D_1/b*
_output_shapes
:@
�
Conv2D_1/Conv2DConv2DMaxPool2D/MaxPoolConv2D_1/W/read*
paddingSAME*/
_output_shapes
:���������

@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
�
Conv2D_1/BiasAddBiasAddConv2D_1/Conv2DConv2D_1/b/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

@
a
Conv2D_1/ReluReluConv2D_1/BiasAdd*
T0*/
_output_shapes
:���������

@
�
MaxPool2D_1/MaxPoolMaxPoolConv2D_1/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@*
T0
�
+Conv2D_2/W/Initializer/random_uniform/shapeConst*%
valueB"      @   �   *
_class
loc:@Conv2D_2/W*
dtype0*
_output_shapes
:
�
)Conv2D_2/W/Initializer/random_uniform/minConst*
valueB
 *�\1�*
_class
loc:@Conv2D_2/W*
dtype0*
_output_shapes
: 
�
)Conv2D_2/W/Initializer/random_uniform/maxConst*
valueB
 *�\1=*
_class
loc:@Conv2D_2/W*
dtype0*
_output_shapes
: 
�
3Conv2D_2/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_2/W/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:@�*

seed *
T0*
_class
loc:@Conv2D_2/W*
seed2 
�
)Conv2D_2/W/Initializer/random_uniform/subSub)Conv2D_2/W/Initializer/random_uniform/max)Conv2D_2/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_2/W*
_output_shapes
: 
�
)Conv2D_2/W/Initializer/random_uniform/mulMul3Conv2D_2/W/Initializer/random_uniform/RandomUniform)Conv2D_2/W/Initializer/random_uniform/sub*
T0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�
%Conv2D_2/W/Initializer/random_uniformAdd)Conv2D_2/W/Initializer/random_uniform/mul)Conv2D_2/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�

Conv2D_2/W
VariableV2*
_class
loc:@Conv2D_2/W*
	container *
shape:@�*
dtype0*'
_output_shapes
:@�*
shared_name 
�
Conv2D_2/W/AssignAssign
Conv2D_2/W%Conv2D_2/W/Initializer/random_uniform*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
x
Conv2D_2/W/readIdentity
Conv2D_2/W*'
_output_shapes
:@�*
T0*
_class
loc:@Conv2D_2/W
�
Conv2D_2/b/Initializer/ConstConst*
valueB�*    *
_class
loc:@Conv2D_2/b*
dtype0*
_output_shapes	
:�
�

Conv2D_2/b
VariableV2*
shared_name *
_class
loc:@Conv2D_2/b*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
Conv2D_2/b/AssignAssign
Conv2D_2/bConv2D_2/b/Initializer/Const*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
l
Conv2D_2/b/readIdentity
Conv2D_2/b*
T0*
_class
loc:@Conv2D_2/b*
_output_shapes	
:�
�
Conv2D_2/Conv2DConv2DMaxPool2D_1/MaxPoolConv2D_2/W/read*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
�
Conv2D_2/BiasAddBiasAddConv2D_2/Conv2DConv2D_2/b/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
b
Conv2D_2/ReluReluConv2D_2/BiasAdd*
T0*0
_output_shapes
:����������
�
MaxPool2D_2/MaxPoolMaxPoolConv2D_2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
�
+Conv2D_3/W/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      �   @   *
_class
loc:@Conv2D_3/W
�
)Conv2D_3/W/Initializer/random_uniform/minConst*
valueB
 *����*
_class
loc:@Conv2D_3/W*
dtype0*
_output_shapes
: 
�
)Conv2D_3/W/Initializer/random_uniform/maxConst*
valueB
 *���<*
_class
loc:@Conv2D_3/W*
dtype0*
_output_shapes
: 
�
3Conv2D_3/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_3/W/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@Conv2D_3/W*
seed2 *
dtype0*'
_output_shapes
:�@
�
)Conv2D_3/W/Initializer/random_uniform/subSub)Conv2D_3/W/Initializer/random_uniform/max)Conv2D_3/W/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@Conv2D_3/W
�
)Conv2D_3/W/Initializer/random_uniform/mulMul3Conv2D_3/W/Initializer/random_uniform/RandomUniform)Conv2D_3/W/Initializer/random_uniform/sub*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
%Conv2D_3/W/Initializer/random_uniformAdd)Conv2D_3/W/Initializer/random_uniform/mul)Conv2D_3/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�

Conv2D_3/W
VariableV2*
_class
loc:@Conv2D_3/W*
	container *
shape:�@*
dtype0*'
_output_shapes
:�@*
shared_name 
�
Conv2D_3/W/AssignAssign
Conv2D_3/W%Conv2D_3/W/Initializer/random_uniform*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
x
Conv2D_3/W/readIdentity
Conv2D_3/W*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
Conv2D_3/b/Initializer/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    *
_class
loc:@Conv2D_3/b
�

Conv2D_3/b
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_3/b
�
Conv2D_3/b/AssignAssign
Conv2D_3/bConv2D_3/b/Initializer/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
k
Conv2D_3/b/readIdentity
Conv2D_3/b*
_output_shapes
:@*
T0*
_class
loc:@Conv2D_3/b
�
Conv2D_3/Conv2DConv2DMaxPool2D_2/MaxPoolConv2D_3/W/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������@
�
Conv2D_3/BiasAddBiasAddConv2D_3/Conv2DConv2D_3/b/read*
data_formatNHWC*/
_output_shapes
:���������@*
T0
a
Conv2D_3/ReluReluConv2D_3/BiasAdd*
T0*/
_output_shapes
:���������@
�
MaxPool2D_3/MaxPoolMaxPoolConv2D_3/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@*
T0
�
+Conv2D_4/W/Initializer/random_uniform/shapeConst*%
valueB"      @       *
_class
loc:@Conv2D_4/W*
dtype0*
_output_shapes
:
�
)Conv2D_4/W/Initializer/random_uniform/minConst*
valueB
 *�\1�*
_class
loc:@Conv2D_4/W*
dtype0*
_output_shapes
: 
�
)Conv2D_4/W/Initializer/random_uniform/maxConst*
valueB
 *�\1=*
_class
loc:@Conv2D_4/W*
dtype0*
_output_shapes
: 
�
3Conv2D_4/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_4/W/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@ *

seed *
T0*
_class
loc:@Conv2D_4/W*
seed2 
�
)Conv2D_4/W/Initializer/random_uniform/subSub)Conv2D_4/W/Initializer/random_uniform/max)Conv2D_4/W/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@Conv2D_4/W
�
)Conv2D_4/W/Initializer/random_uniform/mulMul3Conv2D_4/W/Initializer/random_uniform/RandomUniform)Conv2D_4/W/Initializer/random_uniform/sub*&
_output_shapes
:@ *
T0*
_class
loc:@Conv2D_4/W
�
%Conv2D_4/W/Initializer/random_uniformAdd)Conv2D_4/W/Initializer/random_uniform/mul)Conv2D_4/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�

Conv2D_4/W
VariableV2*
	container *
shape:@ *
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@Conv2D_4/W
�
Conv2D_4/W/AssignAssign
Conv2D_4/W%Conv2D_4/W/Initializer/random_uniform*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
w
Conv2D_4/W/readIdentity
Conv2D_4/W*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�
Conv2D_4/b/Initializer/ConstConst*
valueB *    *
_class
loc:@Conv2D_4/b*
dtype0*
_output_shapes
: 
�

Conv2D_4/b
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D_4/b*
	container *
shape: 
�
Conv2D_4/b/AssignAssign
Conv2D_4/bConv2D_4/b/Initializer/Const*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
k
Conv2D_4/b/readIdentity
Conv2D_4/b*
T0*
_class
loc:@Conv2D_4/b*
_output_shapes
: 
�
Conv2D_4/Conv2DConv2DMaxPool2D_3/MaxPoolConv2D_4/W/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*/
_output_shapes
:��������� *
	dilations
*
T0
�
Conv2D_4/BiasAddBiasAddConv2D_4/Conv2DConv2D_4/b/read*
T0*
data_formatNHWC*/
_output_shapes
:��������� 
a
Conv2D_4/ReluReluConv2D_4/BiasAdd*
T0*/
_output_shapes
:��������� 
�
MaxPool2D_4/MaxPoolMaxPoolConv2D_4/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:��������� *
T0
�
3FullyConnected/W/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"       *#
_class
loc:@FullyConnected/W
�
2FullyConnected/W/Initializer/truncated_normal/meanConst*
valueB
 *    *#
_class
loc:@FullyConnected/W*
dtype0*
_output_shapes
: 
�
4FullyConnected/W/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
ף<*#
_class
loc:@FullyConnected/W
�
=FullyConnected/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3FullyConnected/W/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:	 �*

seed *
T0*#
_class
loc:@FullyConnected/W*
seed2 
�
1FullyConnected/W/Initializer/truncated_normal/mulMul=FullyConnected/W/Initializer/truncated_normal/TruncatedNormal4FullyConnected/W/Initializer/truncated_normal/stddev*
_output_shapes
:	 �*
T0*#
_class
loc:@FullyConnected/W
�
-FullyConnected/W/Initializer/truncated_normalAdd1FullyConnected/W/Initializer/truncated_normal/mul2FullyConnected/W/Initializer/truncated_normal/mean*
_output_shapes
:	 �*
T0*#
_class
loc:@FullyConnected/W
�
FullyConnected/W
VariableV2*
shared_name *#
_class
loc:@FullyConnected/W*
	container *
shape:	 �*
dtype0*
_output_shapes
:	 �
�
FullyConnected/W/AssignAssignFullyConnected/W-FullyConnected/W/Initializer/truncated_normal*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
FullyConnected/W/readIdentityFullyConnected/W*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
"FullyConnected/b/Initializer/ConstConst*
valueB�*    *#
_class
loc:@FullyConnected/b*
dtype0*
_output_shapes	
:�
�
FullyConnected/b
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *#
_class
loc:@FullyConnected/b*
	container *
shape:�
�
FullyConnected/b/AssignAssignFullyConnected/b"FullyConnected/b/Initializer/Const*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
~
FullyConnected/b/readIdentityFullyConnected/b*
T0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
m
FullyConnected/Reshape/shapeConst*
valueB"����    *
dtype0*
_output_shapes
:
�
FullyConnected/ReshapeReshapeMaxPool2D_4/MaxPoolFullyConnected/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:��������� 
�
FullyConnected/MatMulMatMulFullyConnected/ReshapeFullyConnected/W/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
FullyConnected/BiasAddBiasAddFullyConnected/MatMulFullyConnected/b/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
f
FullyConnected/ReluReluFullyConnected/BiasAdd*
T0*(
_output_shapes
:����������

is_training/Initializer/ConstConst*
value	B
 Z *
_class
loc:@is_training*
dtype0
*
_output_shapes
: 
�
is_training
VariableV2*
dtype0
*
_output_shapes
: *
shared_name *
_class
loc:@is_training*
	container *
shape: 
�
is_training/AssignAssignis_trainingis_training/Initializer/Const*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
*
_class
loc:@is_training
j
is_training/readIdentityis_training*
_output_shapes
: *
T0
*
_class
loc:@is_training
V
Dropout/Assign/valueConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
Dropout/AssignAssignis_trainingDropout/Assign/value*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: 
X
Dropout/Assign_1/valueConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
�
Dropout/Assign_1Assignis_trainingDropout/Assign_1/value*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: *
use_locking(
_
Dropout/cond/SwitchSwitchis_trainingis_training/read*
T0
*
_output_shapes
: : 
Y
Dropout/cond/switch_tIdentityDropout/cond/Switch:1*
_output_shapes
: *
T0

W
Dropout/cond/switch_fIdentityDropout/cond/Switch*
T0
*
_output_shapes
: 
S
Dropout/cond/pred_idIdentityis_training/read*
_output_shapes
: *
T0

v
Dropout/cond/dropout/rateConst^Dropout/cond/switch_t*
valueB
 *��L>*
dtype0*
_output_shapes
: 
}
Dropout/cond/dropout/ShapeShape#Dropout/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
�
!Dropout/cond/dropout/Shape/SwitchSwitchFullyConnected/ReluDropout/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*&
_class
loc:@FullyConnected/Relu
�
'Dropout/cond/dropout/random_uniform/minConst^Dropout/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'Dropout/cond/dropout/random_uniform/maxConst^Dropout/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
1Dropout/cond/dropout/random_uniform/RandomUniformRandomUniformDropout/cond/dropout/Shape*

seed *
T0*
dtype0*(
_output_shapes
:����������*
seed2 
�
'Dropout/cond/dropout/random_uniform/subSub'Dropout/cond/dropout/random_uniform/max'Dropout/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
'Dropout/cond/dropout/random_uniform/mulMul1Dropout/cond/dropout/random_uniform/RandomUniform'Dropout/cond/dropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
#Dropout/cond/dropout/random_uniformAdd'Dropout/cond/dropout/random_uniform/mul'Dropout/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
w
Dropout/cond/dropout/sub/xConst^Dropout/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
w
Dropout/cond/dropout/subSubDropout/cond/dropout/sub/xDropout/cond/dropout/rate*
_output_shapes
: *
T0
{
Dropout/cond/dropout/truediv/xConst^Dropout/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dropout/cond/dropout/truedivRealDivDropout/cond/dropout/truediv/xDropout/cond/dropout/sub*
_output_shapes
: *
T0
�
!Dropout/cond/dropout/GreaterEqualGreaterEqual#Dropout/cond/dropout/random_uniformDropout/cond/dropout/rate*
T0*(
_output_shapes
:����������
�
Dropout/cond/dropout/mulMul#Dropout/cond/dropout/Shape/Switch:1Dropout/cond/dropout/truediv*
T0*(
_output_shapes
:����������
�
Dropout/cond/dropout/CastCast!Dropout/cond/dropout/GreaterEqual*

SrcT0
*
Truncate( *(
_output_shapes
:����������*

DstT0
�
Dropout/cond/dropout/mul_1MulDropout/cond/dropout/mulDropout/cond/dropout/Cast*
T0*(
_output_shapes
:����������
�
Dropout/cond/Switch_1SwitchFullyConnected/ReluDropout/cond/pred_id*
T0*&
_class
loc:@FullyConnected/Relu*<
_output_shapes*
(:����������:����������
�
Dropout/cond/MergeMergeDropout/cond/Switch_1Dropout/cond/dropout/mul_1*
T0*
N**
_output_shapes
:����������: 
�
5FullyConnected_1/W/Initializer/truncated_normal/shapeConst*
valueB"      *%
_class
loc:@FullyConnected_1/W*
dtype0*
_output_shapes
:
�
4FullyConnected_1/W/Initializer/truncated_normal/meanConst*
valueB
 *    *%
_class
loc:@FullyConnected_1/W*
dtype0*
_output_shapes
: 
�
6FullyConnected_1/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*%
_class
loc:@FullyConnected_1/W*
dtype0*
_output_shapes
: 
�
?FullyConnected_1/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5FullyConnected_1/W/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:	�*

seed *
T0*%
_class
loc:@FullyConnected_1/W*
seed2 
�
3FullyConnected_1/W/Initializer/truncated_normal/mulMul?FullyConnected_1/W/Initializer/truncated_normal/TruncatedNormal6FullyConnected_1/W/Initializer/truncated_normal/stddev*
_output_shapes
:	�*
T0*%
_class
loc:@FullyConnected_1/W
�
/FullyConnected_1/W/Initializer/truncated_normalAdd3FullyConnected_1/W/Initializer/truncated_normal/mul4FullyConnected_1/W/Initializer/truncated_normal/mean*
T0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
�
FullyConnected_1/W
VariableV2*%
_class
loc:@FullyConnected_1/W*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name 
�
FullyConnected_1/W/AssignAssignFullyConnected_1/W/FullyConnected_1/W/Initializer/truncated_normal*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
FullyConnected_1/W/readIdentityFullyConnected_1/W*
T0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
�
$FullyConnected_1/b/Initializer/ConstConst*
valueB*    *%
_class
loc:@FullyConnected_1/b*
dtype0*
_output_shapes
:
�
FullyConnected_1/b
VariableV2*
shared_name *%
_class
loc:@FullyConnected_1/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
FullyConnected_1/b/AssignAssignFullyConnected_1/b$FullyConnected_1/b/Initializer/Const*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
FullyConnected_1/b/readIdentityFullyConnected_1/b*
_output_shapes
:*
T0*%
_class
loc:@FullyConnected_1/b
�
FullyConnected_1/MatMulMatMulDropout/cond/MergeFullyConnected_1/W/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
FullyConnected_1/BiasAddBiasAddFullyConnected_1/MatMulFullyConnected_1/b/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
o
FullyConnected_1/SoftmaxSoftmaxFullyConnected_1/BiasAdd*'
_output_shapes
:���������*
T0
l
	targets/YPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
[
Accuracy/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
�
Accuracy/ArgMaxArgMaxFullyConnected_1/SoftmaxAccuracy/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
]
Accuracy/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
Accuracy/ArgMax_1ArgMax	targets/YAccuracy/ArgMax_1/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
�
Accuracy/EqualEqualAccuracy/ArgMaxAccuracy/ArgMax_1*
T0	*#
_output_shapes
:���������*
incompatible_shape_error(
r
Accuracy/CastCastAccuracy/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
X
Accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
r
Accuracy/MeanMeanAccuracy/CastAccuracy/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
d
"Crossentropy/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
Crossentropy/SumSumFullyConnected_1/Softmax"Crossentropy/Sum/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
}
Crossentropy/truedivRealDivFullyConnected_1/SoftmaxCrossentropy/Sum*
T0*'
_output_shapes
:���������
X
Crossentropy/Cast/xConst*
dtype0*
_output_shapes
: *
valueB
 *���.
Z
Crossentropy/Cast_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
"Crossentropy/clip_by_value/MinimumMinimumCrossentropy/truedivCrossentropy/Cast_1/x*
T0*'
_output_shapes
:���������
�
Crossentropy/clip_by_valueMaximum"Crossentropy/clip_by_value/MinimumCrossentropy/Cast/x*'
_output_shapes
:���������*
T0
e
Crossentropy/LogLogCrossentropy/clip_by_value*
T0*'
_output_shapes
:���������
f
Crossentropy/mulMul	targets/YCrossentropy/Log*
T0*'
_output_shapes
:���������
f
$Crossentropy/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
Crossentropy/Sum_1SumCrossentropy/mul$Crossentropy/Sum_1/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
Y
Crossentropy/NegNegCrossentropy/Sum_1*
T0*#
_output_shapes
:���������
\
Crossentropy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
}
Crossentropy/MeanMeanCrossentropy/NegCrossentropy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
Training_step/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
q
Training_step
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
Training_step/AssignAssignTraining_stepTraining_step/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@Training_step
p
Training_step/readIdentityTraining_step*
T0* 
_class
loc:@Training_step*
_output_shapes
: 
^
Global_Step/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
Global_Step
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
Global_Step/AssignAssignGlobal_StepGlobal_Step/initial_value*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: *
use_locking(
j
Global_Step/readIdentityGlobal_Step*
_output_shapes
: *
T0*
_class
loc:@Global_Step
J
Add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
D
AddAddGlobal_Step/readAdd/y*
T0*
_output_shapes
: 
�
AssignAssignGlobal_StepAdd*
use_locking(*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: 
[
val_loss/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
l
val_loss
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
val_loss/AssignAssignval_lossval_loss/initial_value*
T0*
_class
loc:@val_loss*
validate_shape(*
_output_shapes
: *
use_locking(
a
val_loss/readIdentityval_loss*
T0*
_class
loc:@val_loss*
_output_shapes
: 
Z
val_acc/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
val_acc
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
val_acc/AssignAssignval_accval_acc/initial_value*
use_locking(*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: 
^
val_acc/readIdentityval_acc*
T0*
_class
loc:@val_acc*
_output_shapes
: 
Y
placeholder/val_lossPlaceholder*
dtype0*
_output_shapes
:*
shape:
X
placeholder/val_accPlaceholder*
shape:*
dtype0*
_output_shapes
:
�
assign/val_lossAssignval_lossplaceholder/val_loss*
use_locking(*
T0*
_class
loc:@val_loss*
validate_shape(*
_output_shapes
: 
�
assign/val_accAssignval_accplaceholder/val_acc*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@val_acc
�
*Accuracy/Mean/moving_avg/Initializer/zerosConst*
valueB
 *    *+
_class!
loc:@Accuracy/Mean/moving_avg*
dtype0*
_output_shapes
: 
�
Accuracy/Mean/moving_avg
VariableV2*
shared_name *+
_class!
loc:@Accuracy/Mean/moving_avg*
	container *
shape: *
dtype0*
_output_shapes
: 
�
Accuracy/Mean/moving_avg/AssignAssignAccuracy/Mean/moving_avg*Accuracy/Mean/moving_avg/Initializer/zeros*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: *
use_locking(
�
Accuracy/Mean/moving_avg/readIdentityAccuracy/Mean/moving_avg*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
_output_shapes
: 
U
moving_avg/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
moving_avg/add/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
^
moving_avg/addAddV2moving_avg/add/xTraining_step/read*
T0*
_output_shapes
: 
W
moving_avg/add_1/xConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
b
moving_avg/add_1AddV2moving_avg/add_1/xTraining_step/read*
T0*
_output_shapes
: 
`
moving_avg/truedivRealDivmoving_avg/addmoving_avg/add_1*
T0*
_output_shapes
: 
d
moving_avg/MinimumMinimummoving_avg/decaymoving_avg/truediv*
T0*
_output_shapes
: 
e
 moving_avg/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
|
moving_avg/AssignMovingAvg/subSub moving_avg/AssignMovingAvg/sub/xmoving_avg/Minimum*
T0*
_output_shapes
: 
v
 moving_avg/AssignMovingAvg/sub_1SubAccuracy/Mean/moving_avg/readAccuracy/Mean*
T0*
_output_shapes
: 
�
moving_avg/AssignMovingAvg/mulMul moving_avg/AssignMovingAvg/sub_1moving_avg/AssignMovingAvg/sub*
T0*
_output_shapes
: 
�
moving_avg/AssignMovingAvg	AssignSubAccuracy/Mean/moving_avgmoving_avg/AssignMovingAvg/mul*
use_locking( *
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
_output_shapes
: 
/

moving_avgNoOp^moving_avg/AssignMovingAvg
O
Adam/Total_LossIdentityCrossentropy/Mean*
T0*
_output_shapes
: 
�
.Crossentropy/Mean/moving_avg/Initializer/zerosConst*
valueB
 *    */
_class%
#!loc:@Crossentropy/Mean/moving_avg*
dtype0*
_output_shapes
: 
�
Crossentropy/Mean/moving_avg
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name */
_class%
#!loc:@Crossentropy/Mean/moving_avg
�
#Crossentropy/Mean/moving_avg/AssignAssignCrossentropy/Mean/moving_avg.Crossentropy/Mean/moving_avg/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
�
!Crossentropy/Mean/moving_avg/readIdentityCrossentropy/Mean/moving_avg*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
_output_shapes
: 
Z
Adam/moving_avg/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Z
Adam/moving_avg/add/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
h
Adam/moving_avg/addAddV2Adam/moving_avg/add/xTraining_step/read*
_output_shapes
: *
T0
\
Adam/moving_avg/add_1/xConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
l
Adam/moving_avg/add_1AddV2Adam/moving_avg/add_1/xTraining_step/read*
T0*
_output_shapes
: 
o
Adam/moving_avg/truedivRealDivAdam/moving_avg/addAdam/moving_avg/add_1*
_output_shapes
: *
T0
s
Adam/moving_avg/MinimumMinimumAdam/moving_avg/decayAdam/moving_avg/truediv*
T0*
_output_shapes
: 
j
%Adam/moving_avg/AssignMovingAvg/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
#Adam/moving_avg/AssignMovingAvg/subSub%Adam/moving_avg/AssignMovingAvg/sub/xAdam/moving_avg/Minimum*
T0*
_output_shapes
: 
�
%Adam/moving_avg/AssignMovingAvg/sub_1Sub!Crossentropy/Mean/moving_avg/readCrossentropy/Mean*
_output_shapes
: *
T0
�
#Adam/moving_avg/AssignMovingAvg/mulMul%Adam/moving_avg/AssignMovingAvg/sub_1#Adam/moving_avg/AssignMovingAvg/sub*
_output_shapes
: *
T0
�
Adam/moving_avg/AssignMovingAvg	AssignSubCrossentropy/Mean/moving_avg#Adam/moving_avg/AssignMovingAvg/mul*
use_locking( *
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
_output_shapes
: 
9
Adam/moving_avgNoOp ^Adam/moving_avg/AssignMovingAvg
N
	Loss/tagsConst*
dtype0*
_output_shapes
: *
valueB
 BLoss
d
LossScalarSummary	Loss/tags!Crossentropy/Mean/moving_avg/read*
T0*
_output_shapes
: 
`
Adam/Loss/raw/tagsConst*
valueB BAdam/Loss/raw*
dtype0*
_output_shapes
: 
f
Adam/Loss/rawScalarSummaryAdam/Loss/raw/tagsCrossentropy/Mean*
T0*
_output_shapes
: 
v
Adam/gradients/ShapeConst^Adam/moving_avg^moving_avg*
valueB *
dtype0*
_output_shapes
: 
|
Adam/gradients/grad_ys_0Const^Adam/moving_avg^moving_avg*
valueB
 *  �?*
dtype0*
_output_shapes
: 
~
Adam/gradients/FillFillAdam/gradients/ShapeAdam/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
3Adam/gradients/Crossentropy/Mean_grad/Reshape/shapeConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
:*
valueB:
�
-Adam/gradients/Crossentropy/Mean_grad/ReshapeReshapeAdam/gradients/Fill3Adam/gradients/Crossentropy/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
�
+Adam/gradients/Crossentropy/Mean_grad/ShapeShapeCrossentropy/Neg^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
*Adam/gradients/Crossentropy/Mean_grad/TileTile-Adam/gradients/Crossentropy/Mean_grad/Reshape+Adam/gradients/Crossentropy/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
�
-Adam/gradients/Crossentropy/Mean_grad/Shape_1ShapeCrossentropy/Neg^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
-Adam/gradients/Crossentropy/Mean_grad/Shape_2Const^Adam/moving_avg^moving_avg*
valueB *
dtype0*
_output_shapes
: 
�
+Adam/gradients/Crossentropy/Mean_grad/ConstConst^Adam/moving_avg^moving_avg*
valueB: *
dtype0*
_output_shapes
:
�
*Adam/gradients/Crossentropy/Mean_grad/ProdProd-Adam/gradients/Crossentropy/Mean_grad/Shape_1+Adam/gradients/Crossentropy/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
-Adam/gradients/Crossentropy/Mean_grad/Const_1Const^Adam/moving_avg^moving_avg*
valueB: *
dtype0*
_output_shapes
:
�
,Adam/gradients/Crossentropy/Mean_grad/Prod_1Prod-Adam/gradients/Crossentropy/Mean_grad/Shape_2-Adam/gradients/Crossentropy/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
/Adam/gradients/Crossentropy/Mean_grad/Maximum/yConst^Adam/moving_avg^moving_avg*
value	B :*
dtype0*
_output_shapes
: 
�
-Adam/gradients/Crossentropy/Mean_grad/MaximumMaximum,Adam/gradients/Crossentropy/Mean_grad/Prod_1/Adam/gradients/Crossentropy/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
.Adam/gradients/Crossentropy/Mean_grad/floordivFloorDiv*Adam/gradients/Crossentropy/Mean_grad/Prod-Adam/gradients/Crossentropy/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
*Adam/gradients/Crossentropy/Mean_grad/CastCast.Adam/gradients/Crossentropy/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
-Adam/gradients/Crossentropy/Mean_grad/truedivRealDiv*Adam/gradients/Crossentropy/Mean_grad/Tile*Adam/gradients/Crossentropy/Mean_grad/Cast*#
_output_shapes
:���������*
T0
�
(Adam/gradients/Crossentropy/Neg_grad/NegNeg-Adam/gradients/Crossentropy/Mean_grad/truediv*#
_output_shapes
:���������*
T0
�
,Adam/gradients/Crossentropy/Sum_1_grad/ShapeShapeCrossentropy/mul^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
+Adam/gradients/Crossentropy/Sum_1_grad/SizeConst^Adam/moving_avg^moving_avg*
value	B :*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
*Adam/gradients/Crossentropy/Sum_1_grad/addAddV2$Crossentropy/Sum_1/reduction_indices+Adam/gradients/Crossentropy/Sum_1_grad/Size*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
_output_shapes
: 
�
*Adam/gradients/Crossentropy/Sum_1_grad/modFloorMod*Adam/gradients/Crossentropy/Sum_1_grad/add+Adam/gradients/Crossentropy/Sum_1_grad/Size*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
_output_shapes
: 
�
.Adam/gradients/Crossentropy/Sum_1_grad/Shape_1Const^Adam/moving_avg^moving_avg*
valueB *?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
2Adam/gradients/Crossentropy/Sum_1_grad/range/startConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
value	B : *?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape
�
2Adam/gradients/Crossentropy/Sum_1_grad/range/deltaConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
value	B :*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape
�
,Adam/gradients/Crossentropy/Sum_1_grad/rangeRange2Adam/gradients/Crossentropy/Sum_1_grad/range/start+Adam/gradients/Crossentropy/Sum_1_grad/Size2Adam/gradients/Crossentropy/Sum_1_grad/range/delta*

Tidx0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
_output_shapes
:
�
1Adam/gradients/Crossentropy/Sum_1_grad/Fill/valueConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
value	B :*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape
�
+Adam/gradients/Crossentropy/Sum_1_grad/FillFill.Adam/gradients/Crossentropy/Sum_1_grad/Shape_11Adam/gradients/Crossentropy/Sum_1_grad/Fill/value*
T0*

index_type0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
_output_shapes
: 
�
4Adam/gradients/Crossentropy/Sum_1_grad/DynamicStitchDynamicStitch,Adam/gradients/Crossentropy/Sum_1_grad/range*Adam/gradients/Crossentropy/Sum_1_grad/mod,Adam/gradients/Crossentropy/Sum_1_grad/Shape+Adam/gradients/Crossentropy/Sum_1_grad/Fill*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
N*
_output_shapes
:
�
0Adam/gradients/Crossentropy/Sum_1_grad/Maximum/yConst^Adam/moving_avg^moving_avg*
value	B :*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
.Adam/gradients/Crossentropy/Sum_1_grad/MaximumMaximum4Adam/gradients/Crossentropy/Sum_1_grad/DynamicStitch0Adam/gradients/Crossentropy/Sum_1_grad/Maximum/y*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
_output_shapes
:
�
/Adam/gradients/Crossentropy/Sum_1_grad/floordivFloorDiv,Adam/gradients/Crossentropy/Sum_1_grad/Shape.Adam/gradients/Crossentropy/Sum_1_grad/Maximum*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
_output_shapes
:
�
.Adam/gradients/Crossentropy/Sum_1_grad/ReshapeReshape(Adam/gradients/Crossentropy/Neg_grad/Neg4Adam/gradients/Crossentropy/Sum_1_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
+Adam/gradients/Crossentropy/Sum_1_grad/TileTile.Adam/gradients/Crossentropy/Sum_1_grad/Reshape/Adam/gradients/Crossentropy/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
�
*Adam/gradients/Crossentropy/mul_grad/ShapeShape	targets/Y^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
�
,Adam/gradients/Crossentropy/mul_grad/Shape_1ShapeCrossentropy/Log^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
:Adam/gradients/Crossentropy/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*Adam/gradients/Crossentropy/mul_grad/Shape,Adam/gradients/Crossentropy/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
(Adam/gradients/Crossentropy/mul_grad/MulMul+Adam/gradients/Crossentropy/Sum_1_grad/TileCrossentropy/Log*
T0*'
_output_shapes
:���������
�
(Adam/gradients/Crossentropy/mul_grad/SumSum(Adam/gradients/Crossentropy/mul_grad/Mul:Adam/gradients/Crossentropy/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
,Adam/gradients/Crossentropy/mul_grad/ReshapeReshape(Adam/gradients/Crossentropy/mul_grad/Sum*Adam/gradients/Crossentropy/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
*Adam/gradients/Crossentropy/mul_grad/Mul_1Mul	targets/Y+Adam/gradients/Crossentropy/Sum_1_grad/Tile*
T0*'
_output_shapes
:���������
�
*Adam/gradients/Crossentropy/mul_grad/Sum_1Sum*Adam/gradients/Crossentropy/mul_grad/Mul_1<Adam/gradients/Crossentropy/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
.Adam/gradients/Crossentropy/mul_grad/Reshape_1Reshape*Adam/gradients/Crossentropy/mul_grad/Sum_1,Adam/gradients/Crossentropy/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/Adam/gradients/Crossentropy/Log_grad/Reciprocal
ReciprocalCrossentropy/clip_by_value/^Adam/gradients/Crossentropy/mul_grad/Reshape_1^Adam/moving_avg^moving_avg*'
_output_shapes
:���������*
T0
�
(Adam/gradients/Crossentropy/Log_grad/mulMul.Adam/gradients/Crossentropy/mul_grad/Reshape_1/Adam/gradients/Crossentropy/Log_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
4Adam/gradients/Crossentropy/clip_by_value_grad/ShapeShape"Crossentropy/clip_by_value/Minimum^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_1Const^Adam/moving_avg^moving_avg*
valueB *
dtype0*
_output_shapes
: 
�
6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_2Shape(Adam/gradients/Crossentropy/Log_grad/mul*
_output_shapes
:*
T0*
out_type0
�
:Adam/gradients/Crossentropy/clip_by_value_grad/zeros/ConstConst^Adam/moving_avg^moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
�
4Adam/gradients/Crossentropy/clip_by_value_grad/zerosFill6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_2:Adam/gradients/Crossentropy/clip_by_value_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������
�
;Adam/gradients/Crossentropy/clip_by_value_grad/GreaterEqualGreaterEqual"Crossentropy/clip_by_value/MinimumCrossentropy/Cast/x^Adam/moving_avg^moving_avg*
T0*'
_output_shapes
:���������
�
DAdam/gradients/Crossentropy/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs4Adam/gradients/Crossentropy/clip_by_value_grad/Shape6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5Adam/gradients/Crossentropy/clip_by_value_grad/SelectSelect;Adam/gradients/Crossentropy/clip_by_value_grad/GreaterEqual(Adam/gradients/Crossentropy/Log_grad/mul4Adam/gradients/Crossentropy/clip_by_value_grad/zeros*
T0*'
_output_shapes
:���������
�
2Adam/gradients/Crossentropy/clip_by_value_grad/SumSum5Adam/gradients/Crossentropy/clip_by_value_grad/SelectDAdam/gradients/Crossentropy/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
6Adam/gradients/Crossentropy/clip_by_value_grad/ReshapeReshape2Adam/gradients/Crossentropy/clip_by_value_grad/Sum4Adam/gradients/Crossentropy/clip_by_value_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
7Adam/gradients/Crossentropy/clip_by_value_grad/Select_1Select;Adam/gradients/Crossentropy/clip_by_value_grad/GreaterEqual4Adam/gradients/Crossentropy/clip_by_value_grad/zeros(Adam/gradients/Crossentropy/Log_grad/mul*
T0*'
_output_shapes
:���������
�
4Adam/gradients/Crossentropy/clip_by_value_grad/Sum_1Sum7Adam/gradients/Crossentropy/clip_by_value_grad/Select_1FAdam/gradients/Crossentropy/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
8Adam/gradients/Crossentropy/clip_by_value_grad/Reshape_1Reshape4Adam/gradients/Crossentropy/clip_by_value_grad/Sum_16Adam/gradients/Crossentropy/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/ShapeShapeCrossentropy/truediv^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
�
>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_1Const^Adam/moving_avg^moving_avg*
valueB *
dtype0*
_output_shapes
: 
�
>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_2Shape6Adam/gradients/Crossentropy/clip_by_value_grad/Reshape*
_output_shapes
:*
T0*
out_type0
�
BAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros/ConstConst^Adam/moving_avg^moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/zerosFill>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_2BAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������
�
@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/LessEqual	LessEqualCrossentropy/truedivCrossentropy/Cast_1/x^Adam/moving_avg^moving_avg*
T0*'
_output_shapes
:���������
�
LAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
=Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SelectSelect@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/LessEqual6Adam/gradients/Crossentropy/clip_by_value_grad/Reshape<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros*
T0*'
_output_shapes
:���������
�
:Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SumSum=Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SelectLAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/ReshapeReshape:Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Sum<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
?Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Select_1Select@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/LessEqual<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros6Adam/gradients/Crossentropy/clip_by_value_grad/Reshape*
T0*'
_output_shapes
:���������
�
<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Sum_1Sum?Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Select_1NAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Reshape_1Reshape<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Sum_1>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
.Adam/gradients/Crossentropy/truediv_grad/ShapeShapeFullyConnected_1/Softmax^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
0Adam/gradients/Crossentropy/truediv_grad/Shape_1ShapeCrossentropy/Sum^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
>Adam/gradients/Crossentropy/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs.Adam/gradients/Crossentropy/truediv_grad/Shape0Adam/gradients/Crossentropy/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0Adam/gradients/Crossentropy/truediv_grad/RealDivRealDiv>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/ReshapeCrossentropy/Sum*
T0*'
_output_shapes
:���������
�
,Adam/gradients/Crossentropy/truediv_grad/SumSum0Adam/gradients/Crossentropy/truediv_grad/RealDiv>Adam/gradients/Crossentropy/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
0Adam/gradients/Crossentropy/truediv_grad/ReshapeReshape,Adam/gradients/Crossentropy/truediv_grad/Sum.Adam/gradients/Crossentropy/truediv_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
,Adam/gradients/Crossentropy/truediv_grad/NegNegFullyConnected_1/Softmax^Adam/moving_avg^moving_avg*'
_output_shapes
:���������*
T0
�
2Adam/gradients/Crossentropy/truediv_grad/RealDiv_1RealDiv,Adam/gradients/Crossentropy/truediv_grad/NegCrossentropy/Sum*'
_output_shapes
:���������*
T0
�
2Adam/gradients/Crossentropy/truediv_grad/RealDiv_2RealDiv2Adam/gradients/Crossentropy/truediv_grad/RealDiv_1Crossentropy/Sum*
T0*'
_output_shapes
:���������
�
,Adam/gradients/Crossentropy/truediv_grad/mulMul>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Reshape2Adam/gradients/Crossentropy/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������
�
.Adam/gradients/Crossentropy/truediv_grad/Sum_1Sum,Adam/gradients/Crossentropy/truediv_grad/mul@Adam/gradients/Crossentropy/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
2Adam/gradients/Crossentropy/truediv_grad/Reshape_1Reshape.Adam/gradients/Crossentropy/truediv_grad/Sum_10Adam/gradients/Crossentropy/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
*Adam/gradients/Crossentropy/Sum_grad/ShapeShapeFullyConnected_1/Softmax^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
�
)Adam/gradients/Crossentropy/Sum_grad/SizeConst^Adam/moving_avg^moving_avg*
value	B :*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
(Adam/gradients/Crossentropy/Sum_grad/addAddV2"Crossentropy/Sum/reduction_indices)Adam/gradients/Crossentropy/Sum_grad/Size*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
: 
�
(Adam/gradients/Crossentropy/Sum_grad/modFloorMod(Adam/gradients/Crossentropy/Sum_grad/add)Adam/gradients/Crossentropy/Sum_grad/Size*
_output_shapes
: *
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape
�
,Adam/gradients/Crossentropy/Sum_grad/Shape_1Const^Adam/moving_avg^moving_avg*
valueB *=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
0Adam/gradients/Crossentropy/Sum_grad/range/startConst^Adam/moving_avg^moving_avg*
value	B : *=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
0Adam/gradients/Crossentropy/Sum_grad/range/deltaConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
value	B :*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape
�
*Adam/gradients/Crossentropy/Sum_grad/rangeRange0Adam/gradients/Crossentropy/Sum_grad/range/start)Adam/gradients/Crossentropy/Sum_grad/Size0Adam/gradients/Crossentropy/Sum_grad/range/delta*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
:*

Tidx0
�
/Adam/gradients/Crossentropy/Sum_grad/Fill/valueConst^Adam/moving_avg^moving_avg*
value	B :*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
)Adam/gradients/Crossentropy/Sum_grad/FillFill,Adam/gradients/Crossentropy/Sum_grad/Shape_1/Adam/gradients/Crossentropy/Sum_grad/Fill/value*
T0*

index_type0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
: 
�
2Adam/gradients/Crossentropy/Sum_grad/DynamicStitchDynamicStitch*Adam/gradients/Crossentropy/Sum_grad/range(Adam/gradients/Crossentropy/Sum_grad/mod*Adam/gradients/Crossentropy/Sum_grad/Shape)Adam/gradients/Crossentropy/Sum_grad/Fill*
N*
_output_shapes
:*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape
�
.Adam/gradients/Crossentropy/Sum_grad/Maximum/yConst^Adam/moving_avg^moving_avg*
value	B :*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
,Adam/gradients/Crossentropy/Sum_grad/MaximumMaximum2Adam/gradients/Crossentropy/Sum_grad/DynamicStitch.Adam/gradients/Crossentropy/Sum_grad/Maximum/y*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
:
�
-Adam/gradients/Crossentropy/Sum_grad/floordivFloorDiv*Adam/gradients/Crossentropy/Sum_grad/Shape,Adam/gradients/Crossentropy/Sum_grad/Maximum*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
:
�
,Adam/gradients/Crossentropy/Sum_grad/ReshapeReshape2Adam/gradients/Crossentropy/truediv_grad/Reshape_12Adam/gradients/Crossentropy/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
)Adam/gradients/Crossentropy/Sum_grad/TileTile,Adam/gradients/Crossentropy/Sum_grad/Reshape-Adam/gradients/Crossentropy/Sum_grad/floordiv*'
_output_shapes
:���������*

Tmultiples0*
T0
�
Adam/gradients/AddNAddN0Adam/gradients/Crossentropy/truediv_grad/Reshape)Adam/gradients/Crossentropy/Sum_grad/Tile*
T0*C
_class9
75loc:@Adam/gradients/Crossentropy/truediv_grad/Reshape*
N*'
_output_shapes
:���������
�
0Adam/gradients/FullyConnected_1/Softmax_grad/mulMulAdam/gradients/AddNFullyConnected_1/Softmax*'
_output_shapes
:���������*
T0
�
BAdam/gradients/FullyConnected_1/Softmax_grad/Sum/reduction_indicesConst^Adam/moving_avg^moving_avg*
valueB :
���������*
dtype0*
_output_shapes
: 
�
0Adam/gradients/FullyConnected_1/Softmax_grad/SumSum0Adam/gradients/FullyConnected_1/Softmax_grad/mulBAdam/gradients/FullyConnected_1/Softmax_grad/Sum/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
�
0Adam/gradients/FullyConnected_1/Softmax_grad/subSubAdam/gradients/AddN0Adam/gradients/FullyConnected_1/Softmax_grad/Sum*
T0*'
_output_shapes
:���������
�
2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1Mul0Adam/gradients/FullyConnected_1/Softmax_grad/subFullyConnected_1/Softmax*
T0*'
_output_shapes
:���������
�
8Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGradBiasAddGrad2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1*
T0*
data_formatNHWC*
_output_shapes
:
�
2Adam/gradients/FullyConnected_1/MatMul_grad/MatMulMatMul2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1FullyConnected_1/W/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
4Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1MatMulDropout/cond/Merge2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1*
transpose_b( *
T0*
_output_shapes
:	�*
transpose_a(
�
0Adam/gradients/Dropout/cond/Merge_grad/cond_gradSwitch2Adam/gradients/FullyConnected_1/MatMul_grad/MatMulDropout/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul
�
Adam/gradients/SwitchSwitchFullyConnected/ReluDropout/cond/pred_id^Adam/moving_avg^moving_avg*
T0*<
_output_shapes*
(:����������:����������
o
Adam/gradients/IdentityIdentityAdam/gradients/Switch:1*
T0*(
_output_shapes
:����������
m
Adam/gradients/Shape_1ShapeAdam/gradients/Switch:1*
T0*
out_type0*
_output_shapes
:
�
Adam/gradients/zeros/ConstConst^Adam/gradients/Identity^Adam/moving_avg^moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Adam/gradients/zerosFillAdam/gradients/Shape_1Adam/gradients/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
3Adam/gradients/Dropout/cond/Switch_1_grad/cond_gradMerge0Adam/gradients/Dropout/cond/Merge_grad/cond_gradAdam/gradients/zeros*
T0*
N**
_output_shapes
:����������: 
�
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/ShapeShapeDropout/cond/dropout/mul^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
�
6Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape_1ShapeDropout/cond/dropout/Cast^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
�
DAdam/gradients/Dropout/cond/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape6Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2Adam/gradients/Dropout/cond/dropout/mul_1_grad/MulMul2Adam/gradients/Dropout/cond/Merge_grad/cond_grad:1Dropout/cond/dropout/Cast*
T0*(
_output_shapes
:����������
�
2Adam/gradients/Dropout/cond/dropout/mul_1_grad/SumSum2Adam/gradients/Dropout/cond/dropout/mul_1_grad/MulDAdam/gradients/Dropout/cond/dropout/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6Adam/gradients/Dropout/cond/dropout/mul_1_grad/ReshapeReshape2Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Mul_1MulDropout/cond/dropout/mul2Adam/gradients/Dropout/cond/Merge_grad/cond_grad:1*(
_output_shapes
:����������*
T0
�
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum_1Sum4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Mul_1FAdam/gradients/Dropout/cond/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
8Adam/gradients/Dropout/cond/dropout/mul_1_grad/Reshape_1Reshape4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum_16Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/ShapeShape#Dropout/cond/dropout/Shape/Switch:1^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
4Adam/gradients/Dropout/cond/dropout/mul_grad/Shape_1ShapeDropout/cond/dropout/truediv^Adam/moving_avg^moving_avg*
_output_shapes
: *
T0*
out_type0
�
BAdam/gradients/Dropout/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2Adam/gradients/Dropout/cond/dropout/mul_grad/Shape4Adam/gradients/Dropout/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0Adam/gradients/Dropout/cond/dropout/mul_grad/MulMul6Adam/gradients/Dropout/cond/dropout/mul_1_grad/ReshapeDropout/cond/dropout/truediv*
T0*(
_output_shapes
:����������
�
0Adam/gradients/Dropout/cond/dropout/mul_grad/SumSum0Adam/gradients/Dropout/cond/dropout/mul_grad/MulBAdam/gradients/Dropout/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
4Adam/gradients/Dropout/cond/dropout/mul_grad/ReshapeReshape0Adam/gradients/Dropout/cond/dropout/mul_grad/Sum2Adam/gradients/Dropout/cond/dropout/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/Mul_1Mul#Dropout/cond/dropout/Shape/Switch:16Adam/gradients/Dropout/cond/dropout/mul_1_grad/Reshape*
T0*(
_output_shapes
:����������
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/Sum_1Sum2Adam/gradients/Dropout/cond/dropout/mul_grad/Mul_1DAdam/gradients/Dropout/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
6Adam/gradients/Dropout/cond/dropout/mul_grad/Reshape_1Reshape2Adam/gradients/Dropout/cond/dropout/mul_grad/Sum_14Adam/gradients/Dropout/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Adam/gradients/Switch_1SwitchFullyConnected/ReluDropout/cond/pred_id^Adam/moving_avg^moving_avg*
T0*<
_output_shapes*
(:����������:����������
q
Adam/gradients/Identity_1IdentityAdam/gradients/Switch_1*(
_output_shapes
:����������*
T0
m
Adam/gradients/Shape_2ShapeAdam/gradients/Switch_1*
T0*
out_type0*
_output_shapes
:
�
Adam/gradients/zeros_1/ConstConst^Adam/gradients/Identity_1^Adam/moving_avg^moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Adam/gradients/zeros_1FillAdam/gradients/Shape_2Adam/gradients/zeros_1/Const*(
_output_shapes
:����������*
T0*

index_type0
�
?Adam/gradients/Dropout/cond/dropout/Shape/Switch_grad/cond_gradMergeAdam/gradients/zeros_14Adam/gradients/Dropout/cond/dropout/mul_grad/Reshape*
T0*
N**
_output_shapes
:����������: 
�
Adam/gradients/AddN_1AddN3Adam/gradients/Dropout/cond/Switch_1_grad/cond_grad?Adam/gradients/Dropout/cond/dropout/Shape/Switch_grad/cond_grad*
T0*F
_class<
:8loc:@Adam/gradients/Dropout/cond/Switch_1_grad/cond_grad*
N*(
_output_shapes
:����������
�
0Adam/gradients/FullyConnected/Relu_grad/ReluGradReluGradAdam/gradients/AddN_1FullyConnected/Relu*
T0*(
_output_shapes
:����������
�
6Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGradBiasAddGrad0Adam/gradients/FullyConnected/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
0Adam/gradients/FullyConnected/MatMul_grad/MatMulMatMul0Adam/gradients/FullyConnected/Relu_grad/ReluGradFullyConnected/W/read*
transpose_b(*
T0*'
_output_shapes
:��������� *
transpose_a( 
�
2Adam/gradients/FullyConnected/MatMul_grad/MatMul_1MatMulFullyConnected/Reshape0Adam/gradients/FullyConnected/Relu_grad/ReluGrad*
T0*
_output_shapes
:	 �*
transpose_a(*
transpose_b( 
�
0Adam/gradients/FullyConnected/Reshape_grad/ShapeShapeMaxPool2D_4/MaxPool^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
�
2Adam/gradients/FullyConnected/Reshape_grad/ReshapeReshape0Adam/gradients/FullyConnected/MatMul_grad/MatMul0Adam/gradients/FullyConnected/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
3Adam/gradients/MaxPool2D_4/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_4/ReluMaxPool2D_4/MaxPool2Adam/gradients/FullyConnected/Reshape_grad/Reshape*/
_output_shapes
:��������� *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
*Adam/gradients/Conv2D_4/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_4/MaxPool_grad/MaxPoolGradConv2D_4/Relu*
T0*/
_output_shapes
:��������� 
�
0Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
*Adam/gradients/Conv2D_4/Conv2D_grad/ShapeNShapeNMaxPool2D_3/MaxPoolConv2D_4/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
7Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_4/Conv2D_grad/ShapeNConv2D_4/W/read*Adam/gradients/Conv2D_4/Relu_grad/ReluGrad*
paddingSAME*/
_output_shapes
:���������@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
�
8Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_3/MaxPool,Adam/gradients/Conv2D_4/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_4/Relu_grad/ReluGrad*&
_output_shapes
:@ *
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
�
3Adam/gradients/MaxPool2D_3/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_3/ReluMaxPool2D_3/MaxPool7Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingSAME*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides

�
*Adam/gradients/Conv2D_3/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_3/MaxPool_grad/MaxPoolGradConv2D_3/Relu*
T0*/
_output_shapes
:���������@
�
0Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
*Adam/gradients/Conv2D_3/Conv2D_grad/ShapeNShapeNMaxPool2D_2/MaxPoolConv2D_3/W/read^Adam/moving_avg^moving_avg*
N* 
_output_shapes
::*
T0*
out_type0
�
7Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_3/Conv2D_grad/ShapeNConv2D_3/W/read*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
�
8Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_2/MaxPool,Adam/gradients/Conv2D_3/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*
paddingSAME*'
_output_shapes
:�@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
�
3Adam/gradients/MaxPool2D_2/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_2/ReluMaxPool2D_2/MaxPool7Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides

�
*Adam/gradients/Conv2D_2/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_2/MaxPool_grad/MaxPoolGradConv2D_2/Relu*0
_output_shapes
:����������*
T0
�
0Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
*Adam/gradients/Conv2D_2/Conv2D_grad/ShapeNShapeNMaxPool2D_1/MaxPoolConv2D_2/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
7Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_2/Conv2D_grad/ShapeNConv2D_2/W/read*Adam/gradients/Conv2D_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������@*
	dilations

�
8Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_1/MaxPool,Adam/gradients/Conv2D_2/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_2/Relu_grad/ReluGrad*
paddingSAME*'
_output_shapes
:@�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
�
3Adam/gradients/MaxPool2D_1/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_1/ReluMaxPool2D_1/MaxPool7Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropInput*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������

@
�
*Adam/gradients/Conv2D_1/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_1/MaxPool_grad/MaxPoolGradConv2D_1/Relu*
T0*/
_output_shapes
:���������

@
�
0Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
*Adam/gradients/Conv2D_1/Conv2D_grad/ShapeNShapeNMaxPool2D/MaxPoolConv2D_1/W/read^Adam/moving_avg^moving_avg*
N* 
_output_shapes
::*
T0*
out_type0
�
7Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_1/Conv2D_grad/ShapeNConv2D_1/W/read*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*
paddingSAME*/
_output_shapes
:���������

 *
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
�
8Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D/MaxPool,Adam/gradients/Conv2D_1/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*&
_output_shapes
: @*
	dilations

�
1Adam/gradients/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D/ReluMaxPool2D/MaxPool7Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropInput*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������22 
�
(Adam/gradients/Conv2D/Relu_grad/ReluGradReluGrad1Adam/gradients/MaxPool2D/MaxPool_grad/MaxPoolGradConv2D/Relu*
T0*/
_output_shapes
:���������22 
�
.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGradBiasAddGrad(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
�
(Adam/gradients/Conv2D/Conv2D_grad/ShapeNShapeNinput/XConv2D/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
5Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(Adam/gradients/Conv2D/Conv2D_grad/ShapeNConv2D/W/read(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*/
_output_shapes
:���������22
�
6Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput/X*Adam/gradients/Conv2D/Conv2D_grad/ShapeN:1(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*&
_output_shapes
: *
	dilations

�
Adam/global_norm/L2LossL2Loss6Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*
T0*I
_class?
=;loc:@Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_1L2Loss.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad*
T0*A
_class7
53loc:@Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_2L2Loss8Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_3L2Loss0Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_4L2Loss8Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_5L2Loss0Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_6L2Loss8Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_7L2Loss0Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_8L2Loss8Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_9L2Loss0Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_10L2Loss2Adam/gradients/FullyConnected/MatMul_grad/MatMul_1*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1*
_output_shapes
: 
�
Adam/global_norm/L2Loss_11L2Loss6Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad*
T0*I
_class?
=;loc:@Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_12L2Loss4Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1*
T0*G
_class=
;9loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1*
_output_shapes
: 
�
Adam/global_norm/L2Loss_13L2Loss8Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
T0*K
_classA
?=loc:@Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/stackPackAdam/global_norm/L2LossAdam/global_norm/L2Loss_1Adam/global_norm/L2Loss_2Adam/global_norm/L2Loss_3Adam/global_norm/L2Loss_4Adam/global_norm/L2Loss_5Adam/global_norm/L2Loss_6Adam/global_norm/L2Loss_7Adam/global_norm/L2Loss_8Adam/global_norm/L2Loss_9Adam/global_norm/L2Loss_10Adam/global_norm/L2Loss_11Adam/global_norm/L2Loss_12Adam/global_norm/L2Loss_13*
T0*

axis *
N*
_output_shapes
:

Adam/global_norm/ConstConst^Adam/moving_avg^moving_avg*
valueB: *
dtype0*
_output_shapes
:
�
Adam/global_norm/SumSumAdam/global_norm/stackAdam/global_norm/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
|
Adam/global_norm/Const_1Const^Adam/moving_avg^moving_avg*
valueB
 *   @*
dtype0*
_output_shapes
: 
l
Adam/global_norm/mulMulAdam/global_norm/SumAdam/global_norm/Const_1*
T0*
_output_shapes
: 
[
Adam/global_norm/global_normSqrtAdam/global_norm/mul*
T0*
_output_shapes
: 
�
"Adam/clip_by_global_norm/truediv/xConst^Adam/moving_avg^moving_avg*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
 Adam/clip_by_global_norm/truedivRealDiv"Adam/clip_by_global_norm/truediv/xAdam/global_norm/global_norm*
_output_shapes
: *
T0
�
Adam/clip_by_global_norm/ConstConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$Adam/clip_by_global_norm/truediv_1/yConst^Adam/moving_avg^moving_avg*
valueB
 *  �@*
dtype0*
_output_shapes
: 
�
"Adam/clip_by_global_norm/truediv_1RealDivAdam/clip_by_global_norm/Const$Adam/clip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 
�
 Adam/clip_by_global_norm/MinimumMinimum Adam/clip_by_global_norm/truediv"Adam/clip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
�
Adam/clip_by_global_norm/mul/xConst^Adam/moving_avg^moving_avg*
valueB
 *  �@*
dtype0*
_output_shapes
: 
�
Adam/clip_by_global_norm/mulMulAdam/clip_by_global_norm/mul/x Adam/clip_by_global_norm/Minimum*
T0*
_output_shapes
: 
l
!Adam/clip_by_global_norm/IsFiniteIsFiniteAdam/global_norm/global_norm*
T0*
_output_shapes
: 
�
 Adam/clip_by_global_norm/Const_1Const^Adam/moving_avg^moving_avg*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
Adam/clip_by_global_norm/SelectSelect!Adam/clip_by_global_norm/IsFiniteAdam/clip_by_global_norm/mul Adam/clip_by_global_norm/Const_1*
_output_shapes
: *
T0
�
Adam/clip_by_global_norm/mul_1Mul6Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*
T0*I
_class?
=;loc:@Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_0IdentityAdam/clip_by_global_norm/mul_1*&
_output_shapes
: *
T0*I
_class?
=;loc:@Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter
�
Adam/clip_by_global_norm/mul_2Mul.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*A
_class7
53loc:@Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_1IdentityAdam/clip_by_global_norm/mul_2*
T0*A
_class7
53loc:@Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/clip_by_global_norm/mul_3Mul8Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_2IdentityAdam/clip_by_global_norm/mul_3*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
Adam/clip_by_global_norm/mul_4Mul0Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_3IdentityAdam/clip_by_global_norm/mul_4*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
Adam/clip_by_global_norm/mul_5Mul8Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_4IdentityAdam/clip_by_global_norm/mul_5*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
Adam/clip_by_global_norm/mul_6Mul0Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_5IdentityAdam/clip_by_global_norm/mul_6*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Adam/clip_by_global_norm/mul_7Mul8Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*'
_output_shapes
:�@*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_6IdentityAdam/clip_by_global_norm/mul_7*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�@
�
Adam/clip_by_global_norm/mul_8Mul0Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_7IdentityAdam/clip_by_global_norm/mul_8*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
Adam/clip_by_global_norm/mul_9Mul8Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@ 
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_8IdentityAdam/clip_by_global_norm/mul_9*&
_output_shapes
:@ *
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter
�
Adam/clip_by_global_norm/mul_10Mul0Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
_output_shapes
: *
T0*C
_class9
75loc:@Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGrad
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_9IdentityAdam/clip_by_global_norm/mul_10*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/clip_by_global_norm/mul_11Mul2Adam/gradients/FullyConnected/MatMul_grad/MatMul_1Adam/clip_by_global_norm/Select*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1*
_output_shapes
:	 �
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_10IdentityAdam/clip_by_global_norm/mul_11*
_output_shapes
:	 �*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1
�
Adam/clip_by_global_norm/mul_12Mul6Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
_output_shapes	
:�*
T0*I
_class?
=;loc:@Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_11IdentityAdam/clip_by_global_norm/mul_12*
_output_shapes	
:�*
T0*I
_class?
=;loc:@Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad
�
Adam/clip_by_global_norm/mul_13Mul4Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1Adam/clip_by_global_norm/Select*
T0*G
_class=
;9loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_12IdentityAdam/clip_by_global_norm/mul_13*
T0*G
_class=
;9loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
Adam/clip_by_global_norm/mul_14Mul8Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*K
_classA
?=loc:@Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_13IdentityAdam/clip_by_global_norm/mul_14*
T0*K
_classA
?=loc:@Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
Adam/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@Conv2D/W
�
Adam/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/W*
	container *
shape: 
�
Adam/beta1_power/AssignAssignAdam/beta1_powerAdam/beta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
q
Adam/beta1_power/readIdentityAdam/beta1_power*
T0*
_class
loc:@Conv2D/W*
_output_shapes
: 
�
Adam/beta2_power/initial_valueConst*
valueB
 *w�?*
_class
loc:@Conv2D/W*
dtype0*
_output_shapes
: 
�
Adam/beta2_power
VariableV2*
shared_name *
_class
loc:@Conv2D/W*
	container *
shape: *
dtype0*
_output_shapes
: 
�
Adam/beta2_power/AssignAssignAdam/beta2_powerAdam/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
q
Adam/beta2_power/readIdentityAdam/beta2_power*
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
Conv2D/W/Adam/Initializer/zerosConst*
_class
loc:@Conv2D/W*%
valueB *    *
dtype0*&
_output_shapes
: 
�
Conv2D/W/Adam
VariableV2*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/W
�
Conv2D/W/Adam/AssignAssignConv2D/W/AdamConv2D/W/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
{
Conv2D/W/Adam/readIdentityConv2D/W/Adam*
T0*
_class
loc:@Conv2D/W*&
_output_shapes
: 
�
!Conv2D/W/Adam_1/Initializer/zerosConst*
_class
loc:@Conv2D/W*%
valueB *    *
dtype0*&
_output_shapes
: 
�
Conv2D/W/Adam_1
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/W*
	container *
shape: 
�
Conv2D/W/Adam_1/AssignAssignConv2D/W/Adam_1!Conv2D/W/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(

Conv2D/W/Adam_1/readIdentityConv2D/W/Adam_1*&
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
Conv2D/b/Adam/Initializer/zerosConst*
_class
loc:@Conv2D/b*
valueB *    *
dtype0*
_output_shapes
: 
�
Conv2D/b/Adam
VariableV2*
shared_name *
_class
loc:@Conv2D/b*
	container *
shape: *
dtype0*
_output_shapes
: 
�
Conv2D/b/Adam/AssignAssignConv2D/b/AdamConv2D/b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
o
Conv2D/b/Adam/readIdentityConv2D/b/Adam*
T0*
_class
loc:@Conv2D/b*
_output_shapes
: 
�
!Conv2D/b/Adam_1/Initializer/zerosConst*
_class
loc:@Conv2D/b*
valueB *    *
dtype0*
_output_shapes
: 
�
Conv2D/b/Adam_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/b
�
Conv2D/b/Adam_1/AssignAssignConv2D/b/Adam_1!Conv2D/b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
s
Conv2D/b/Adam_1/readIdentityConv2D/b/Adam_1*
T0*
_class
loc:@Conv2D/b*
_output_shapes
: 
�
1Conv2D_1/W/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Conv2D_1/W*%
valueB"          @   *
dtype0*
_output_shapes
:
�
'Conv2D_1/W/Adam/Initializer/zeros/ConstConst*
_class
loc:@Conv2D_1/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!Conv2D_1/W/Adam/Initializer/zerosFill1Conv2D_1/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_1/W/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Conv2D_1/W*

index_type0*&
_output_shapes
: @
�
Conv2D_1/W/Adam
VariableV2*
shape: @*
dtype0*&
_output_shapes
: @*
shared_name *
_class
loc:@Conv2D_1/W*
	container 
�
Conv2D_1/W/Adam/AssignAssignConv2D_1/W/Adam!Conv2D_1/W/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
Conv2D_1/W/Adam/readIdentityConv2D_1/W/Adam*
T0*
_class
loc:@Conv2D_1/W*&
_output_shapes
: @
�
3Conv2D_1/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Conv2D_1/W*%
valueB"          @   *
dtype0*
_output_shapes
:
�
)Conv2D_1/W/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_1/W*
valueB
 *    
�
#Conv2D_1/W/Adam_1/Initializer/zerosFill3Conv2D_1/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_1/W/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Conv2D_1/W*

index_type0*&
_output_shapes
: @
�
Conv2D_1/W/Adam_1
VariableV2*
shared_name *
_class
loc:@Conv2D_1/W*
	container *
shape: @*
dtype0*&
_output_shapes
: @
�
Conv2D_1/W/Adam_1/AssignAssignConv2D_1/W/Adam_1#Conv2D_1/W/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
Conv2D_1/W/Adam_1/readIdentityConv2D_1/W/Adam_1*
T0*
_class
loc:@Conv2D_1/W*&
_output_shapes
: @
�
!Conv2D_1/b/Adam/Initializer/zerosConst*
_class
loc:@Conv2D_1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
Conv2D_1/b/Adam
VariableV2*
shared_name *
_class
loc:@Conv2D_1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
Conv2D_1/b/Adam/AssignAssignConv2D_1/b/Adam!Conv2D_1/b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
u
Conv2D_1/b/Adam/readIdentityConv2D_1/b/Adam*
_output_shapes
:@*
T0*
_class
loc:@Conv2D_1/b
�
#Conv2D_1/b/Adam_1/Initializer/zerosConst*
_class
loc:@Conv2D_1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
Conv2D_1/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_1/b*
	container *
shape:@
�
Conv2D_1/b/Adam_1/AssignAssignConv2D_1/b/Adam_1#Conv2D_1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
y
Conv2D_1/b/Adam_1/readIdentityConv2D_1/b/Adam_1*
T0*
_class
loc:@Conv2D_1/b*
_output_shapes
:@
�
1Conv2D_2/W/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Conv2D_2/W*%
valueB"      @   �   *
dtype0*
_output_shapes
:
�
'Conv2D_2/W/Adam/Initializer/zeros/ConstConst*
_class
loc:@Conv2D_2/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!Conv2D_2/W/Adam/Initializer/zerosFill1Conv2D_2/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_2/W/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Conv2D_2/W*

index_type0*'
_output_shapes
:@�
�
Conv2D_2/W/Adam
VariableV2*
dtype0*'
_output_shapes
:@�*
shared_name *
_class
loc:@Conv2D_2/W*
	container *
shape:@�
�
Conv2D_2/W/Adam/AssignAssignConv2D_2/W/Adam!Conv2D_2/W/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
Conv2D_2/W/Adam/readIdentityConv2D_2/W/Adam*
T0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�
3Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@Conv2D_2/W*%
valueB"      @   �   
�
)Conv2D_2/W/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Conv2D_2/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#Conv2D_2/W/Adam_1/Initializer/zerosFill3Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_2/W/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Conv2D_2/W*

index_type0*'
_output_shapes
:@�
�
Conv2D_2/W/Adam_1
VariableV2*
shared_name *
_class
loc:@Conv2D_2/W*
	container *
shape:@�*
dtype0*'
_output_shapes
:@�
�
Conv2D_2/W/Adam_1/AssignAssignConv2D_2/W/Adam_1#Conv2D_2/W/Adam_1/Initializer/zeros*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0*
_class
loc:@Conv2D_2/W
�
Conv2D_2/W/Adam_1/readIdentityConv2D_2/W/Adam_1*
T0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�
!Conv2D_2/b/Adam/Initializer/zerosConst*
_class
loc:@Conv2D_2/b*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Conv2D_2/b/Adam
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@Conv2D_2/b*
	container 
�
Conv2D_2/b/Adam/AssignAssignConv2D_2/b/Adam!Conv2D_2/b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
v
Conv2D_2/b/Adam/readIdentityConv2D_2/b/Adam*
_output_shapes	
:�*
T0*
_class
loc:@Conv2D_2/b
�
#Conv2D_2/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
_class
loc:@Conv2D_2/b*
valueB�*    
�
Conv2D_2/b/Adam_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@Conv2D_2/b
�
Conv2D_2/b/Adam_1/AssignAssignConv2D_2/b/Adam_1#Conv2D_2/b/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
z
Conv2D_2/b/Adam_1/readIdentityConv2D_2/b/Adam_1*
T0*
_class
loc:@Conv2D_2/b*
_output_shapes	
:�
�
1Conv2D_3/W/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Conv2D_3/W*%
valueB"      �   @   *
dtype0*
_output_shapes
:
�
'Conv2D_3/W/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_3/W*
valueB
 *    
�
!Conv2D_3/W/Adam/Initializer/zerosFill1Conv2D_3/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_3/W/Adam/Initializer/zeros/Const*'
_output_shapes
:�@*
T0*
_class
loc:@Conv2D_3/W*

index_type0
�
Conv2D_3/W/Adam
VariableV2*
	container *
shape:�@*
dtype0*'
_output_shapes
:�@*
shared_name *
_class
loc:@Conv2D_3/W
�
Conv2D_3/W/Adam/AssignAssignConv2D_3/W/Adam!Conv2D_3/W/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
Conv2D_3/W/Adam/readIdentityConv2D_3/W/Adam*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
3Conv2D_3/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Conv2D_3/W*%
valueB"      �   @   *
dtype0*
_output_shapes
:
�
)Conv2D_3/W/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Conv2D_3/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#Conv2D_3/W/Adam_1/Initializer/zerosFill3Conv2D_3/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_3/W/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Conv2D_3/W*

index_type0*'
_output_shapes
:�@
�
Conv2D_3/W/Adam_1
VariableV2*
dtype0*'
_output_shapes
:�@*
shared_name *
_class
loc:@Conv2D_3/W*
	container *
shape:�@
�
Conv2D_3/W/Adam_1/AssignAssignConv2D_3/W/Adam_1#Conv2D_3/W/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
Conv2D_3/W/Adam_1/readIdentityConv2D_3/W/Adam_1*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
!Conv2D_3/b/Adam/Initializer/zerosConst*
_class
loc:@Conv2D_3/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
Conv2D_3/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_3/b*
	container *
shape:@
�
Conv2D_3/b/Adam/AssignAssignConv2D_3/b/Adam!Conv2D_3/b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
u
Conv2D_3/b/Adam/readIdentityConv2D_3/b/Adam*
T0*
_class
loc:@Conv2D_3/b*
_output_shapes
:@
�
#Conv2D_3/b/Adam_1/Initializer/zerosConst*
_class
loc:@Conv2D_3/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
Conv2D_3/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_3/b*
	container *
shape:@
�
Conv2D_3/b/Adam_1/AssignAssignConv2D_3/b/Adam_1#Conv2D_3/b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
y
Conv2D_3/b/Adam_1/readIdentityConv2D_3/b/Adam_1*
_output_shapes
:@*
T0*
_class
loc:@Conv2D_3/b
�
1Conv2D_4/W/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@Conv2D_4/W*%
valueB"      @       
�
'Conv2D_4/W/Adam/Initializer/zeros/ConstConst*
_class
loc:@Conv2D_4/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!Conv2D_4/W/Adam/Initializer/zerosFill1Conv2D_4/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_4/W/Adam/Initializer/zeros/Const*&
_output_shapes
:@ *
T0*
_class
loc:@Conv2D_4/W*

index_type0
�
Conv2D_4/W/Adam
VariableV2*
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@Conv2D_4/W*
	container *
shape:@ 
�
Conv2D_4/W/Adam/AssignAssignConv2D_4/W/Adam!Conv2D_4/W/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
Conv2D_4/W/Adam/readIdentityConv2D_4/W/Adam*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�
3Conv2D_4/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@Conv2D_4/W*%
valueB"      @       
�
)Conv2D_4/W/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Conv2D_4/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#Conv2D_4/W/Adam_1/Initializer/zerosFill3Conv2D_4/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_4/W/Adam_1/Initializer/zeros/Const*&
_output_shapes
:@ *
T0*
_class
loc:@Conv2D_4/W*

index_type0
�
Conv2D_4/W/Adam_1
VariableV2*
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@Conv2D_4/W*
	container *
shape:@ 
�
Conv2D_4/W/Adam_1/AssignAssignConv2D_4/W/Adam_1#Conv2D_4/W/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
Conv2D_4/W/Adam_1/readIdentityConv2D_4/W/Adam_1*&
_output_shapes
:@ *
T0*
_class
loc:@Conv2D_4/W
�
!Conv2D_4/b/Adam/Initializer/zerosConst*
_class
loc:@Conv2D_4/b*
valueB *    *
dtype0*
_output_shapes
: 
�
Conv2D_4/b/Adam
VariableV2*
shared_name *
_class
loc:@Conv2D_4/b*
	container *
shape: *
dtype0*
_output_shapes
: 
�
Conv2D_4/b/Adam/AssignAssignConv2D_4/b/Adam!Conv2D_4/b/Adam/Initializer/zeros*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
u
Conv2D_4/b/Adam/readIdentityConv2D_4/b/Adam*
_output_shapes
: *
T0*
_class
loc:@Conv2D_4/b
�
#Conv2D_4/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_4/b*
valueB *    
�
Conv2D_4/b/Adam_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D_4/b*
	container *
shape: 
�
Conv2D_4/b/Adam_1/AssignAssignConv2D_4/b/Adam_1#Conv2D_4/b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
y
Conv2D_4/b/Adam_1/readIdentityConv2D_4/b/Adam_1*
T0*
_class
loc:@Conv2D_4/b*
_output_shapes
: 
�
7FullyConnected/W/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@FullyConnected/W*
valueB"       *
dtype0*
_output_shapes
:
�
-FullyConnected/W/Adam/Initializer/zeros/ConstConst*#
_class
loc:@FullyConnected/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'FullyConnected/W/Adam/Initializer/zerosFill7FullyConnected/W/Adam/Initializer/zeros/shape_as_tensor-FullyConnected/W/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@FullyConnected/W*

index_type0*
_output_shapes
:	 �
�
FullyConnected/W/Adam
VariableV2*#
_class
loc:@FullyConnected/W*
	container *
shape:	 �*
dtype0*
_output_shapes
:	 �*
shared_name 
�
FullyConnected/W/Adam/AssignAssignFullyConnected/W/Adam'FullyConnected/W/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
FullyConnected/W/Adam/readIdentityFullyConnected/W/Adam*
_output_shapes
:	 �*
T0*#
_class
loc:@FullyConnected/W
�
9FullyConnected/W/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@FullyConnected/W*
valueB"       *
dtype0*
_output_shapes
:
�
/FullyConnected/W/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@FullyConnected/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)FullyConnected/W/Adam_1/Initializer/zerosFill9FullyConnected/W/Adam_1/Initializer/zeros/shape_as_tensor/FullyConnected/W/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@FullyConnected/W*

index_type0*
_output_shapes
:	 �
�
FullyConnected/W/Adam_1
VariableV2*
shared_name *#
_class
loc:@FullyConnected/W*
	container *
shape:	 �*
dtype0*
_output_shapes
:	 �
�
FullyConnected/W/Adam_1/AssignAssignFullyConnected/W/Adam_1)FullyConnected/W/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
FullyConnected/W/Adam_1/readIdentityFullyConnected/W/Adam_1*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
7FullyConnected/b/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@FullyConnected/b*
valueB:�*
dtype0*
_output_shapes
:
�
-FullyConnected/b/Adam/Initializer/zeros/ConstConst*#
_class
loc:@FullyConnected/b*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'FullyConnected/b/Adam/Initializer/zerosFill7FullyConnected/b/Adam/Initializer/zeros/shape_as_tensor-FullyConnected/b/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@FullyConnected/b*

index_type0*
_output_shapes	
:�
�
FullyConnected/b/Adam
VariableV2*#
_class
loc:@FullyConnected/b*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
FullyConnected/b/Adam/AssignAssignFullyConnected/b/Adam'FullyConnected/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
FullyConnected/b/Adam/readIdentityFullyConnected/b/Adam*
_output_shapes	
:�*
T0*#
_class
loc:@FullyConnected/b
�
9FullyConnected/b/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@FullyConnected/b*
valueB:�*
dtype0*
_output_shapes
:
�
/FullyConnected/b/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@FullyConnected/b*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)FullyConnected/b/Adam_1/Initializer/zerosFill9FullyConnected/b/Adam_1/Initializer/zeros/shape_as_tensor/FullyConnected/b/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@FullyConnected/b*

index_type0*
_output_shapes	
:�
�
FullyConnected/b/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *#
_class
loc:@FullyConnected/b*
	container *
shape:�
�
FullyConnected/b/Adam_1/AssignAssignFullyConnected/b/Adam_1)FullyConnected/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
FullyConnected/b/Adam_1/readIdentityFullyConnected/b/Adam_1*
T0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
�
9FullyConnected_1/W/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
_class
loc:@FullyConnected_1/W*
valueB"      
�
/FullyConnected_1/W/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@FullyConnected_1/W*
valueB
 *    
�
)FullyConnected_1/W/Adam/Initializer/zerosFill9FullyConnected_1/W/Adam/Initializer/zeros/shape_as_tensor/FullyConnected_1/W/Adam/Initializer/zeros/Const*
T0*%
_class
loc:@FullyConnected_1/W*

index_type0*
_output_shapes
:	�
�
FullyConnected_1/W/Adam
VariableV2*%
_class
loc:@FullyConnected_1/W*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name 
�
FullyConnected_1/W/Adam/AssignAssignFullyConnected_1/W/Adam)FullyConnected_1/W/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
FullyConnected_1/W/Adam/readIdentityFullyConnected_1/W/Adam*
T0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
�
;FullyConnected_1/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
_class
loc:@FullyConnected_1/W*
valueB"      
�
1FullyConnected_1/W/Adam_1/Initializer/zeros/ConstConst*%
_class
loc:@FullyConnected_1/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
+FullyConnected_1/W/Adam_1/Initializer/zerosFill;FullyConnected_1/W/Adam_1/Initializer/zeros/shape_as_tensor1FullyConnected_1/W/Adam_1/Initializer/zeros/Const*
T0*%
_class
loc:@FullyConnected_1/W*

index_type0*
_output_shapes
:	�
�
FullyConnected_1/W/Adam_1
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *%
_class
loc:@FullyConnected_1/W*
	container *
shape:	�
�
 FullyConnected_1/W/Adam_1/AssignAssignFullyConnected_1/W/Adam_1+FullyConnected_1/W/Adam_1/Initializer/zeros*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
FullyConnected_1/W/Adam_1/readIdentityFullyConnected_1/W/Adam_1*
_output_shapes
:	�*
T0*%
_class
loc:@FullyConnected_1/W
�
)FullyConnected_1/b/Adam/Initializer/zerosConst*%
_class
loc:@FullyConnected_1/b*
valueB*    *
dtype0*
_output_shapes
:
�
FullyConnected_1/b/Adam
VariableV2*
shared_name *%
_class
loc:@FullyConnected_1/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
FullyConnected_1/b/Adam/AssignAssignFullyConnected_1/b/Adam)FullyConnected_1/b/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
FullyConnected_1/b/Adam/readIdentityFullyConnected_1/b/Adam*
_output_shapes
:*
T0*%
_class
loc:@FullyConnected_1/b
�
+FullyConnected_1/b/Adam_1/Initializer/zerosConst*%
_class
loc:@FullyConnected_1/b*
valueB*    *
dtype0*
_output_shapes
:
�
FullyConnected_1/b/Adam_1
VariableV2*
shared_name *%
_class
loc:@FullyConnected_1/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
 FullyConnected_1/b/Adam_1/AssignAssignFullyConnected_1/b/Adam_1+FullyConnected_1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
FullyConnected_1/b/Adam_1/readIdentityFullyConnected_1/b/Adam_1*
T0*%
_class
loc:@FullyConnected_1/b*
_output_shapes
:
g
"Adam/apply_grad_op_0/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
_
Adam/apply_grad_op_0/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
_
Adam/apply_grad_op_0/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
a
Adam/apply_grad_op_0/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
.Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam	ApplyAdamConv2D/WConv2D/W/AdamConv2D/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_0*
T0*
_class
loc:@Conv2D/W*
use_nesterov( *&
_output_shapes
: *
use_locking( 
�
.Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam	ApplyAdamConv2D/bConv2D/b/AdamConv2D/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_1*
use_locking( *
T0*
_class
loc:@Conv2D/b*
use_nesterov( *
_output_shapes
: 
�
0Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam	ApplyAdam
Conv2D_1/WConv2D_1/W/AdamConv2D_1/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_2*
T0*
_class
loc:@Conv2D_1/W*
use_nesterov( *&
_output_shapes
: @*
use_locking( 
�
0Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam	ApplyAdam
Conv2D_1/bConv2D_1/b/AdamConv2D_1/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_3*
use_locking( *
T0*
_class
loc:@Conv2D_1/b*
use_nesterov( *
_output_shapes
:@
�
0Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam	ApplyAdam
Conv2D_2/WConv2D_2/W/AdamConv2D_2/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_4*
use_locking( *
T0*
_class
loc:@Conv2D_2/W*
use_nesterov( *'
_output_shapes
:@�
�
0Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam	ApplyAdam
Conv2D_2/bConv2D_2/b/AdamConv2D_2/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_5*
T0*
_class
loc:@Conv2D_2/b*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
0Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam	ApplyAdam
Conv2D_3/WConv2D_3/W/AdamConv2D_3/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_6*
T0*
_class
loc:@Conv2D_3/W*
use_nesterov( *'
_output_shapes
:�@*
use_locking( 
�
0Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam	ApplyAdam
Conv2D_3/bConv2D_3/b/AdamConv2D_3/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_7*
use_locking( *
T0*
_class
loc:@Conv2D_3/b*
use_nesterov( *
_output_shapes
:@
�
0Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam	ApplyAdam
Conv2D_4/WConv2D_4/W/AdamConv2D_4/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_8*
T0*
_class
loc:@Conv2D_4/W*
use_nesterov( *&
_output_shapes
:@ *
use_locking( 
�
0Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam	ApplyAdam
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_9*
T0*
_class
loc:@Conv2D_4/b*
use_nesterov( *
_output_shapes
: *
use_locking( 
�
6Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam	ApplyAdamFullyConnected/WFullyConnected/W/AdamFullyConnected/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_10*
T0*#
_class
loc:@FullyConnected/W*
use_nesterov( *
_output_shapes
:	 �*
use_locking( 
�
6Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam	ApplyAdamFullyConnected/bFullyConnected/b/AdamFullyConnected/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_11*
use_locking( *
T0*#
_class
loc:@FullyConnected/b*
use_nesterov( *
_output_shapes	
:�
�
8Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam	ApplyAdamFullyConnected_1/WFullyConnected_1/W/AdamFullyConnected_1/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_12*
use_nesterov( *
_output_shapes
:	�*
use_locking( *
T0*%
_class
loc:@FullyConnected_1/W
�
8Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam	ApplyAdamFullyConnected_1/bFullyConnected_1/b/AdamFullyConnected_1/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_13*
use_locking( *
T0*%
_class
loc:@FullyConnected_1/b*
use_nesterov( *
_output_shapes
:
�
Adam/apply_grad_op_0/mulMulAdam/beta1_power/readAdam/apply_grad_op_0/beta1/^Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam/^Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
Adam/apply_grad_op_0/AssignAssignAdam/beta1_powerAdam/apply_grad_op_0/mul*
use_locking( *
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
�
Adam/apply_grad_op_0/mul_1MulAdam/beta2_power/readAdam/apply_grad_op_0/beta2/^Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam/^Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
Adam/apply_grad_op_0/Assign_1AssignAdam/beta2_powerAdam/apply_grad_op_0/mul_1*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: *
use_locking( 
�
Adam/apply_grad_op_0/updateNoOp^Adam/apply_grad_op_0/Assign^Adam/apply_grad_op_0/Assign_1/^Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam/^Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam
�
Adam/apply_grad_op_0/valueConst^Adam/apply_grad_op_0/update*
valueB
 *  �?* 
_class
loc:@Training_step*
dtype0*
_output_shapes
: 
�
Adam/apply_grad_op_0	AssignAddTraining_stepAdam/apply_grad_op_0/value*
T0* 
_class
loc:@Training_step*
_output_shapes
: *
use_locking( 
]
Adam/Merge/MergeSummaryMergeSummaryLossAdam/Loss/raw*
N*
_output_shapes
: 
.
Adam/train_op_0NoOp^Adam/apply_grad_op_0
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:3*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss
�
save/SaveV2/shape_and_slicesConst*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesAccuracy/Mean/moving_avgAdam/beta1_powerAdam/beta2_powerConv2D/WConv2D/W/AdamConv2D/W/Adam_1Conv2D/bConv2D/b/AdamConv2D/b/Adam_1
Conv2D_1/WConv2D_1/W/AdamConv2D_1/W/Adam_1
Conv2D_1/bConv2D_1/b/AdamConv2D_1/b/Adam_1
Conv2D_2/WConv2D_2/W/AdamConv2D_2/W/Adam_1
Conv2D_2/bConv2D_2/b/AdamConv2D_2/b/Adam_1
Conv2D_3/WConv2D_3/W/AdamConv2D_3/W/Adam_1
Conv2D_3/bConv2D_3/b/AdamConv2D_3/b/Adam_1
Conv2D_4/WConv2D_4/W/AdamConv2D_4/W/Adam_1
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Crossentropy/Mean/moving_avgFullyConnected/WFullyConnected/W/AdamFullyConnected/W/Adam_1FullyConnected/bFullyConnected/b/AdamFullyConnected/b/Adam_1FullyConnected_1/WFullyConnected_1/W/AdamFullyConnected_1/W/Adam_1FullyConnected_1/bFullyConnected_1/b/AdamFullyConnected_1/b/Adam_1Global_StepTraining_stepis_trainingval_accval_loss*A
dtypes7
523

}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:3*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*A
dtypes7
523
*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::
�
save/AssignAssignAccuracy/Mean/moving_avgsave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg
�
save/Assign_1AssignAdam/beta1_powersave/RestoreV2:1*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_2AssignAdam/beta2_powersave/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save/Assign_3AssignConv2D/Wsave/RestoreV2:3*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save/Assign_4AssignConv2D/W/Adamsave/RestoreV2:4*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save/Assign_5AssignConv2D/W/Adam_1save/RestoreV2:5*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save/Assign_6AssignConv2D/bsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save/Assign_7AssignConv2D/b/Adamsave/RestoreV2:7*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_8AssignConv2D/b/Adam_1save/RestoreV2:8*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_9Assign
Conv2D_1/Wsave/RestoreV2:9*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save/Assign_10AssignConv2D_1/W/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save/Assign_11AssignConv2D_1/W/Adam_1save/RestoreV2:11*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
�
save/Assign_12Assign
Conv2D_1/bsave/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save/Assign_13AssignConv2D_1/b/Adamsave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
�
save/Assign_14AssignConv2D_1/b/Adam_1save/RestoreV2:14*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save/Assign_15Assign
Conv2D_2/Wsave/RestoreV2:15*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save/Assign_16AssignConv2D_2/W/Adamsave/RestoreV2:16*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save/Assign_17AssignConv2D_2/W/Adam_1save/RestoreV2:17*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save/Assign_18Assign
Conv2D_2/bsave/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save/Assign_19AssignConv2D_2/b/Adamsave/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save/Assign_20AssignConv2D_2/b/Adam_1save/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save/Assign_21Assign
Conv2D_3/Wsave/RestoreV2:21*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save/Assign_22AssignConv2D_3/W/Adamsave/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save/Assign_23AssignConv2D_3/W/Adam_1save/RestoreV2:23*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
save/Assign_24Assign
Conv2D_3/bsave/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save/Assign_25AssignConv2D_3/b/Adamsave/RestoreV2:25*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save/Assign_26AssignConv2D_3/b/Adam_1save/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save/Assign_27Assign
Conv2D_4/Wsave/RestoreV2:27*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ *
use_locking(
�
save/Assign_28AssignConv2D_4/W/Adamsave/RestoreV2:28*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save/Assign_29AssignConv2D_4/W/Adam_1save/RestoreV2:29*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
save/Assign_30Assign
Conv2D_4/bsave/RestoreV2:30*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_31AssignConv2D_4/b/Adamsave/RestoreV2:31*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save/Assign_32AssignConv2D_4/b/Adam_1save/RestoreV2:32*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save/Assign_33AssignCrossentropy/Mean/moving_avgsave/RestoreV2:33*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_34AssignFullyConnected/Wsave/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save/Assign_35AssignFullyConnected/W/Adamsave/RestoreV2:35*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save/Assign_36AssignFullyConnected/W/Adam_1save/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save/Assign_37AssignFullyConnected/bsave/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save/Assign_38AssignFullyConnected/b/Adamsave/RestoreV2:38*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save/Assign_39AssignFullyConnected/b/Adam_1save/RestoreV2:39*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
save/Assign_40AssignFullyConnected_1/Wsave/RestoreV2:40*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save/Assign_41AssignFullyConnected_1/W/Adamsave/RestoreV2:41*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W
�
save/Assign_42AssignFullyConnected_1/W/Adam_1save/RestoreV2:42*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save/Assign_43AssignFullyConnected_1/bsave/RestoreV2:43*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
�
save/Assign_44AssignFullyConnected_1/b/Adamsave/RestoreV2:44*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_45AssignFullyConnected_1/b/Adam_1save/RestoreV2:45*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save/Assign_46AssignGlobal_Stepsave/RestoreV2:46*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_47AssignTraining_stepsave/RestoreV2:47*
T0* 
_class
loc:@Training_step*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_48Assignis_trainingsave/RestoreV2:48*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
*
_class
loc:@is_training
�
save/Assign_49Assignval_accsave/RestoreV2:49*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_50Assignval_losssave/RestoreV2:50*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@val_loss
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
save_1/SaveV2/shape_and_slicesConst*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesAccuracy/Mean/moving_avgAdam/beta1_powerAdam/beta2_powerConv2D/WConv2D/W/AdamConv2D/W/Adam_1Conv2D/bConv2D/b/AdamConv2D/b/Adam_1
Conv2D_1/WConv2D_1/W/AdamConv2D_1/W/Adam_1
Conv2D_1/bConv2D_1/b/AdamConv2D_1/b/Adam_1
Conv2D_2/WConv2D_2/W/AdamConv2D_2/W/Adam_1
Conv2D_2/bConv2D_2/b/AdamConv2D_2/b/Adam_1
Conv2D_3/WConv2D_3/W/AdamConv2D_3/W/Adam_1
Conv2D_3/bConv2D_3/b/AdamConv2D_3/b/Adam_1
Conv2D_4/WConv2D_4/W/AdamConv2D_4/W/Adam_1
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Crossentropy/Mean/moving_avgFullyConnected/WFullyConnected/W/AdamFullyConnected/W/Adam_1FullyConnected/bFullyConnected/b/AdamFullyConnected/b/Adam_1FullyConnected_1/WFullyConnected_1/W/AdamFullyConnected_1/W/Adam_1FullyConnected_1/bFullyConnected_1/b/AdamFullyConnected_1/b/Adam_1Global_StepTraining_stepis_trainingval_accval_loss*A
dtypes7
523

�
save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
T0*
_class
loc:@save_1/Const*
_output_shapes
: 
�
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523

�
save_1/AssignAssignAccuracy/Mean/moving_avgsave_1/RestoreV2*
use_locking(*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_1AssignAdam/beta1_powersave_1/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_2AssignAdam/beta2_powersave_1/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_3AssignConv2D/Wsave_1/RestoreV2:3*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save_1/Assign_4AssignConv2D/W/Adamsave_1/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
�
save_1/Assign_5AssignConv2D/W/Adam_1save_1/RestoreV2:5*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save_1/Assign_6AssignConv2D/bsave_1/RestoreV2:6*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_7AssignConv2D/b/Adamsave_1/RestoreV2:7*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_8AssignConv2D/b/Adam_1save_1/RestoreV2:8*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_9Assign
Conv2D_1/Wsave_1/RestoreV2:9*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save_1/Assign_10AssignConv2D_1/W/Adamsave_1/RestoreV2:10*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
�
save_1/Assign_11AssignConv2D_1/W/Adam_1save_1/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save_1/Assign_12Assign
Conv2D_1/bsave_1/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save_1/Assign_13AssignConv2D_1/b/Adamsave_1/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_14AssignConv2D_1/b/Adam_1save_1/RestoreV2:14*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_15Assign
Conv2D_2/Wsave_1/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
save_1/Assign_16AssignConv2D_2/W/Adamsave_1/RestoreV2:16*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
save_1/Assign_17AssignConv2D_2/W/Adam_1save_1/RestoreV2:17*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0*
_class
loc:@Conv2D_2/W
�
save_1/Assign_18Assign
Conv2D_2/bsave_1/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_19AssignConv2D_2/b/Adamsave_1/RestoreV2:19*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
�
save_1/Assign_20AssignConv2D_2/b/Adam_1save_1/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_21Assign
Conv2D_3/Wsave_1/RestoreV2:21*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save_1/Assign_22AssignConv2D_3/W/Adamsave_1/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save_1/Assign_23AssignConv2D_3/W/Adam_1save_1/RestoreV2:23*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
save_1/Assign_24Assign
Conv2D_3/bsave_1/RestoreV2:24*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_25AssignConv2D_3/b/Adamsave_1/RestoreV2:25*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_26AssignConv2D_3/b/Adam_1save_1/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_27Assign
Conv2D_4/Wsave_1/RestoreV2:27*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
save_1/Assign_28AssignConv2D_4/W/Adamsave_1/RestoreV2:28*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save_1/Assign_29AssignConv2D_4/W/Adam_1save_1/RestoreV2:29*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
save_1/Assign_30Assign
Conv2D_4/bsave_1/RestoreV2:30*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_31AssignConv2D_4/b/Adamsave_1/RestoreV2:31*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_32AssignConv2D_4/b/Adam_1save_1/RestoreV2:32*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_33AssignCrossentropy/Mean/moving_avgsave_1/RestoreV2:33*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg
�
save_1/Assign_34AssignFullyConnected/Wsave_1/RestoreV2:34*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save_1/Assign_35AssignFullyConnected/W/Adamsave_1/RestoreV2:35*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save_1/Assign_36AssignFullyConnected/W/Adam_1save_1/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save_1/Assign_37AssignFullyConnected/bsave_1/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_38AssignFullyConnected/b/Adamsave_1/RestoreV2:38*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
save_1/Assign_39AssignFullyConnected/b/Adam_1save_1/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_40AssignFullyConnected_1/Wsave_1/RestoreV2:40*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_1/Assign_41AssignFullyConnected_1/W/Adamsave_1/RestoreV2:41*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_1/Assign_42AssignFullyConnected_1/W/Adam_1save_1/RestoreV2:42*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
save_1/Assign_43AssignFullyConnected_1/bsave_1/RestoreV2:43*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_44AssignFullyConnected_1/b/Adamsave_1/RestoreV2:44*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_45AssignFullyConnected_1/b/Adam_1save_1/RestoreV2:45*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_1/Assign_46AssignGlobal_Stepsave_1/RestoreV2:46*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Global_Step
�
save_1/Assign_47AssignTraining_stepsave_1/RestoreV2:47*
use_locking(*
T0* 
_class
loc:@Training_step*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_48Assignis_trainingsave_1/RestoreV2:48*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_49Assignval_accsave_1/RestoreV2:49*
use_locking(*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_50Assignval_losssave_1/RestoreV2:50*
T0*
_class
loc:@val_loss*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
[
save_2/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_2/SaveV2/tensor_namesConst*�
value�B�BConv2D/WBConv2D/bB
Conv2D_1/WB
Conv2D_1/bB
Conv2D_2/WB
Conv2D_2/bB
Conv2D_3/WB
Conv2D_3/bB
Conv2D_4/WB
Conv2D_4/bBFullyConnected/WBFullyConnected/bBFullyConnected_1/WBFullyConnected_1/b*
dtype0*
_output_shapes
:
�
save_2/SaveV2/shape_and_slicesConst*/
value&B$B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save_2/SaveV2SaveV2save_2/Constsave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesConv2D/WConv2D/b
Conv2D_1/W
Conv2D_1/b
Conv2D_2/W
Conv2D_2/b
Conv2D_3/W
Conv2D_3/b
Conv2D_4/W
Conv2D_4/bFullyConnected/WFullyConnected/bFullyConnected_1/WFullyConnected_1/b*
dtypes
2
�
save_2/control_dependencyIdentitysave_2/Const^save_2/SaveV2*
T0*
_class
loc:@save_2/Const*
_output_shapes
: 
�
save_2/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�BConv2D/WBConv2D/bB
Conv2D_1/WB
Conv2D_1/bB
Conv2D_2/WB
Conv2D_2/bB
Conv2D_3/WB
Conv2D_3/bB
Conv2D_4/WB
Conv2D_4/bBFullyConnected/WBFullyConnected/bBFullyConnected_1/WBFullyConnected_1/b*
dtype0*
_output_shapes
:
�
!save_2/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*/
value&B$B B B B B B B B B B B B B B 
�
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2
�
save_2/AssignAssignConv2D/Wsave_2/RestoreV2*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save_2/Assign_1AssignConv2D/bsave_2/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_2Assign
Conv2D_1/Wsave_2/RestoreV2:2*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save_2/Assign_3Assign
Conv2D_1/bsave_2/RestoreV2:3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save_2/Assign_4Assign
Conv2D_2/Wsave_2/RestoreV2:4*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save_2/Assign_5Assign
Conv2D_2/bsave_2/RestoreV2:5*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
�
save_2/Assign_6Assign
Conv2D_3/Wsave_2/RestoreV2:6*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
save_2/Assign_7Assign
Conv2D_3/bsave_2/RestoreV2:7*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_2/Assign_8Assign
Conv2D_4/Wsave_2/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save_2/Assign_9Assign
Conv2D_4/bsave_2/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_10AssignFullyConnected/Wsave_2/RestoreV2:10*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save_2/Assign_11AssignFullyConnected/bsave_2/RestoreV2:11*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save_2/Assign_12AssignFullyConnected_1/Wsave_2/RestoreV2:12*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
save_2/Assign_13AssignFullyConnected_1/bsave_2/RestoreV2:13*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save_2/restore_allNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_2^save_2/Assign_3^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
�

initNoOp ^Accuracy/Mean/moving_avg/Assign^Adam/beta1_power/Assign^Adam/beta2_power/Assign^Conv2D/W/Adam/Assign^Conv2D/W/Adam_1/Assign^Conv2D/W/Assign^Conv2D/b/Adam/Assign^Conv2D/b/Adam_1/Assign^Conv2D/b/Assign^Conv2D_1/W/Adam/Assign^Conv2D_1/W/Adam_1/Assign^Conv2D_1/W/Assign^Conv2D_1/b/Adam/Assign^Conv2D_1/b/Adam_1/Assign^Conv2D_1/b/Assign^Conv2D_2/W/Adam/Assign^Conv2D_2/W/Adam_1/Assign^Conv2D_2/W/Assign^Conv2D_2/b/Adam/Assign^Conv2D_2/b/Adam_1/Assign^Conv2D_2/b/Assign^Conv2D_3/W/Adam/Assign^Conv2D_3/W/Adam_1/Assign^Conv2D_3/W/Assign^Conv2D_3/b/Adam/Assign^Conv2D_3/b/Adam_1/Assign^Conv2D_3/b/Assign^Conv2D_4/W/Adam/Assign^Conv2D_4/W/Adam_1/Assign^Conv2D_4/W/Assign^Conv2D_4/b/Adam/Assign^Conv2D_4/b/Adam_1/Assign^Conv2D_4/b/Assign$^Crossentropy/Mean/moving_avg/Assign^FullyConnected/W/Adam/Assign^FullyConnected/W/Adam_1/Assign^FullyConnected/W/Assign^FullyConnected/b/Adam/Assign^FullyConnected/b/Adam_1/Assign^FullyConnected/b/Assign^FullyConnected_1/W/Adam/Assign!^FullyConnected_1/W/Adam_1/Assign^FullyConnected_1/W/Assign^FullyConnected_1/b/Adam/Assign!^FullyConnected_1/b/Adam_1/Assign^FullyConnected_1/b/Assign^Global_Step/Assign^Training_step/Assign^is_training/Assign^val_acc/Assign^val_loss/Assign

init_1NoOp
"

group_depsNoOp^init^init_1
#
init_2NoOp^is_training/Assign
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_3/SaveV2/tensor_namesConst*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
save_3/SaveV2/shape_and_slicesConst*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save_3/SaveV2SaveV2save_3/Constsave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesAccuracy/Mean/moving_avgAdam/beta1_powerAdam/beta2_powerConv2D/WConv2D/W/AdamConv2D/W/Adam_1Conv2D/bConv2D/b/AdamConv2D/b/Adam_1
Conv2D_1/WConv2D_1/W/AdamConv2D_1/W/Adam_1
Conv2D_1/bConv2D_1/b/AdamConv2D_1/b/Adam_1
Conv2D_2/WConv2D_2/W/AdamConv2D_2/W/Adam_1
Conv2D_2/bConv2D_2/b/AdamConv2D_2/b/Adam_1
Conv2D_3/WConv2D_3/W/AdamConv2D_3/W/Adam_1
Conv2D_3/bConv2D_3/b/AdamConv2D_3/b/Adam_1
Conv2D_4/WConv2D_4/W/AdamConv2D_4/W/Adam_1
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Crossentropy/Mean/moving_avgFullyConnected/WFullyConnected/W/AdamFullyConnected/W/Adam_1FullyConnected/bFullyConnected/b/AdamFullyConnected/b/Adam_1FullyConnected_1/WFullyConnected_1/W/AdamFullyConnected_1/W/Adam_1FullyConnected_1/bFullyConnected_1/b/AdamFullyConnected_1/b/Adam_1Global_StepTraining_stepis_trainingval_accval_loss*A
dtypes7
523

�
save_3/control_dependencyIdentitysave_3/Const^save_3/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save_3/Const
�
save_3/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
!save_3/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:3*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523

�
save_3/AssignAssignAccuracy/Mean/moving_avgsave_3/RestoreV2*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_1AssignAdam/beta1_powersave_3/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save_3/Assign_2AssignAdam/beta2_powersave_3/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_3AssignConv2D/Wsave_3/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
�
save_3/Assign_4AssignConv2D/W/Adamsave_3/RestoreV2:4*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save_3/Assign_5AssignConv2D/W/Adam_1save_3/RestoreV2:5*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save_3/Assign_6AssignConv2D/bsave_3/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_7AssignConv2D/b/Adamsave_3/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_8AssignConv2D/b/Adam_1save_3/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_9Assign
Conv2D_1/Wsave_3/RestoreV2:9*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
�
save_3/Assign_10AssignConv2D_1/W/Adamsave_3/RestoreV2:10*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save_3/Assign_11AssignConv2D_1/W/Adam_1save_3/RestoreV2:11*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save_3/Assign_12Assign
Conv2D_1/bsave_3/RestoreV2:12*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_3/Assign_13AssignConv2D_1/b/Adamsave_3/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_14AssignConv2D_1/b/Adam_1save_3/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_15Assign
Conv2D_2/Wsave_3/RestoreV2:15*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0*
_class
loc:@Conv2D_2/W
�
save_3/Assign_16AssignConv2D_2/W/Adamsave_3/RestoreV2:16*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
save_3/Assign_17AssignConv2D_2/W/Adam_1save_3/RestoreV2:17*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save_3/Assign_18Assign
Conv2D_2/bsave_3/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save_3/Assign_19AssignConv2D_2/b/Adamsave_3/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save_3/Assign_20AssignConv2D_2/b/Adam_1save_3/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save_3/Assign_21Assign
Conv2D_3/Wsave_3/RestoreV2:21*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save_3/Assign_22AssignConv2D_3/W/Adamsave_3/RestoreV2:22*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save_3/Assign_23AssignConv2D_3/W/Adam_1save_3/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save_3/Assign_24Assign
Conv2D_3/bsave_3/RestoreV2:24*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
�
save_3/Assign_25AssignConv2D_3/b/Adamsave_3/RestoreV2:25*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_26AssignConv2D_3/b/Adam_1save_3/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_27Assign
Conv2D_4/Wsave_3/RestoreV2:27*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
save_3/Assign_28AssignConv2D_4/W/Adamsave_3/RestoreV2:28*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save_3/Assign_29AssignConv2D_4/W/Adam_1save_3/RestoreV2:29*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
save_3/Assign_30Assign
Conv2D_4/bsave_3/RestoreV2:30*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_31AssignConv2D_4/b/Adamsave_3/RestoreV2:31*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_32AssignConv2D_4/b/Adam_1save_3/RestoreV2:32*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_33AssignCrossentropy/Mean/moving_avgsave_3/RestoreV2:33*
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_34AssignFullyConnected/Wsave_3/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save_3/Assign_35AssignFullyConnected/W/Adamsave_3/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save_3/Assign_36AssignFullyConnected/W/Adam_1save_3/RestoreV2:36*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save_3/Assign_37AssignFullyConnected/bsave_3/RestoreV2:37*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_3/Assign_38AssignFullyConnected/b/Adamsave_3/RestoreV2:38*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_3/Assign_39AssignFullyConnected/b/Adam_1save_3/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save_3/Assign_40AssignFullyConnected_1/Wsave_3/RestoreV2:40*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
save_3/Assign_41AssignFullyConnected_1/W/Adamsave_3/RestoreV2:41*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_3/Assign_42AssignFullyConnected_1/W/Adam_1save_3/RestoreV2:42*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_3/Assign_43AssignFullyConnected_1/bsave_3/RestoreV2:43*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_44AssignFullyConnected_1/b/Adamsave_3/RestoreV2:44*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
�
save_3/Assign_45AssignFullyConnected_1/b/Adam_1save_3/RestoreV2:45*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
�
save_3/Assign_46AssignGlobal_Stepsave_3/RestoreV2:46*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Global_Step
�
save_3/Assign_47AssignTraining_stepsave_3/RestoreV2:47*
T0* 
_class
loc:@Training_step*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_48Assignis_trainingsave_3/RestoreV2:48*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
*
_class
loc:@is_training
�
save_3/Assign_49Assignval_accsave_3/RestoreV2:49*
use_locking(*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_50Assignval_losssave_3/RestoreV2:50*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@val_loss
�
save_3/restore_allNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9"�6(�Eq�     ��^	ef>~��AJ��
�2�2
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
A
AddV2
x"T
y"T
z"T"
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignAdd
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
s
	AssignSub
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
.
IsFinite
x"T
y
"
Ttype:
2
2
L2Loss
t"T
output"T"
Ttype:
2
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
�
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.15.02v1.15.0-rc3-22-g590d6eef7e��

z
input/XPlaceholder*
dtype0*/
_output_shapes
:���������22*$
shape:���������22
�
)Conv2D/W/Initializer/random_uniform/shapeConst*
_class
loc:@Conv2D/W*%
valueB"             *
dtype0*
_output_shapes
:
�
'Conv2D/W/Initializer/random_uniform/minConst*
_class
loc:@Conv2D/W*
valueB
 *�\��*
dtype0*
_output_shapes
: 
�
'Conv2D/W/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@Conv2D/W*
valueB
 *�\�>
�
1Conv2D/W/Initializer/random_uniform/RandomUniformRandomUniform)Conv2D/W/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
: *

seed *
T0*
_class
loc:@Conv2D/W
�
'Conv2D/W/Initializer/random_uniform/subSub'Conv2D/W/Initializer/random_uniform/max'Conv2D/W/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
'Conv2D/W/Initializer/random_uniform/mulMul1Conv2D/W/Initializer/random_uniform/RandomUniform'Conv2D/W/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
#Conv2D/W/Initializer/random_uniformAdd'Conv2D/W/Initializer/random_uniform/mul'Conv2D/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D/W*&
_output_shapes
: 
�
Conv2D/W
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/W*
	container *
shape: 
�
Conv2D/W/AssignAssignConv2D/W#Conv2D/W/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
q
Conv2D/W/readIdentityConv2D/W*
T0*
_class
loc:@Conv2D/W*&
_output_shapes
: 
�
Conv2D/b/Initializer/ConstConst*
_class
loc:@Conv2D/b*
valueB *    *
dtype0*
_output_shapes
: 
�
Conv2D/b
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/b*
	container *
shape: 
�
Conv2D/b/AssignAssignConv2D/bConv2D/b/Initializer/Const*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
e
Conv2D/b/readIdentityConv2D/b*
T0*
_class
loc:@Conv2D/b*
_output_shapes
: 
�
Conv2D/Conv2DConv2Dinput/XConv2D/W/read*/
_output_shapes
:���������22 *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
�
Conv2D/BiasAddBiasAddConv2D/Conv2DConv2D/b/read*
T0*
data_formatNHWC*/
_output_shapes
:���������22 
]
Conv2D/ReluReluConv2D/BiasAdd*
T0*/
_output_shapes
:���������22 
�
MaxPool2D/MaxPoolMaxPoolConv2D/Relu*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������

 *
T0
�
+Conv2D_1/W/Initializer/random_uniform/shapeConst*
_class
loc:@Conv2D_1/W*%
valueB"          @   *
dtype0*
_output_shapes
:
�
)Conv2D_1/W/Initializer/random_uniform/minConst*
_class
loc:@Conv2D_1/W*
valueB
 *��z�*
dtype0*
_output_shapes
: 
�
)Conv2D_1/W/Initializer/random_uniform/maxConst*
_class
loc:@Conv2D_1/W*
valueB
 *��z=*
dtype0*
_output_shapes
: 
�
3Conv2D_1/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_1/W/Initializer/random_uniform/shape*
T0*
_class
loc:@Conv2D_1/W*
seed2 *
dtype0*&
_output_shapes
: @*

seed 
�
)Conv2D_1/W/Initializer/random_uniform/subSub)Conv2D_1/W/Initializer/random_uniform/max)Conv2D_1/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_1/W*
_output_shapes
: 
�
)Conv2D_1/W/Initializer/random_uniform/mulMul3Conv2D_1/W/Initializer/random_uniform/RandomUniform)Conv2D_1/W/Initializer/random_uniform/sub*&
_output_shapes
: @*
T0*
_class
loc:@Conv2D_1/W
�
%Conv2D_1/W/Initializer/random_uniformAdd)Conv2D_1/W/Initializer/random_uniform/mul)Conv2D_1/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_1/W*&
_output_shapes
: @
�

Conv2D_1/W
VariableV2*
shared_name *
_class
loc:@Conv2D_1/W*
	container *
shape: @*
dtype0*&
_output_shapes
: @
�
Conv2D_1/W/AssignAssign
Conv2D_1/W%Conv2D_1/W/Initializer/random_uniform*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
w
Conv2D_1/W/readIdentity
Conv2D_1/W*&
_output_shapes
: @*
T0*
_class
loc:@Conv2D_1/W
�
Conv2D_1/b/Initializer/ConstConst*
dtype0*
_output_shapes
:@*
_class
loc:@Conv2D_1/b*
valueB@*    
�

Conv2D_1/b
VariableV2*
shared_name *
_class
loc:@Conv2D_1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
Conv2D_1/b/AssignAssign
Conv2D_1/bConv2D_1/b/Initializer/Const*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
k
Conv2D_1/b/readIdentity
Conv2D_1/b*
T0*
_class
loc:@Conv2D_1/b*
_output_shapes
:@
�
Conv2D_1/Conv2DConv2DMaxPool2D/MaxPoolConv2D_1/W/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*/
_output_shapes
:���������

@
�
Conv2D_1/BiasAddBiasAddConv2D_1/Conv2DConv2D_1/b/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

@
a
Conv2D_1/ReluReluConv2D_1/BiasAdd*
T0*/
_output_shapes
:���������

@
�
MaxPool2D_1/MaxPoolMaxPoolConv2D_1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
�
+Conv2D_2/W/Initializer/random_uniform/shapeConst*
_class
loc:@Conv2D_2/W*%
valueB"      @   �   *
dtype0*
_output_shapes
:
�
)Conv2D_2/W/Initializer/random_uniform/minConst*
_class
loc:@Conv2D_2/W*
valueB
 *�\1�*
dtype0*
_output_shapes
: 
�
)Conv2D_2/W/Initializer/random_uniform/maxConst*
_class
loc:@Conv2D_2/W*
valueB
 *�\1=*
dtype0*
_output_shapes
: 
�
3Conv2D_2/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_2/W/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:@�*

seed *
T0*
_class
loc:@Conv2D_2/W*
seed2 
�
)Conv2D_2/W/Initializer/random_uniform/subSub)Conv2D_2/W/Initializer/random_uniform/max)Conv2D_2/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_2/W*
_output_shapes
: 
�
)Conv2D_2/W/Initializer/random_uniform/mulMul3Conv2D_2/W/Initializer/random_uniform/RandomUniform)Conv2D_2/W/Initializer/random_uniform/sub*'
_output_shapes
:@�*
T0*
_class
loc:@Conv2D_2/W
�
%Conv2D_2/W/Initializer/random_uniformAdd)Conv2D_2/W/Initializer/random_uniform/mul)Conv2D_2/W/Initializer/random_uniform/min*'
_output_shapes
:@�*
T0*
_class
loc:@Conv2D_2/W
�

Conv2D_2/W
VariableV2*
dtype0*'
_output_shapes
:@�*
shared_name *
_class
loc:@Conv2D_2/W*
	container *
shape:@�
�
Conv2D_2/W/AssignAssign
Conv2D_2/W%Conv2D_2/W/Initializer/random_uniform*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
x
Conv2D_2/W/readIdentity
Conv2D_2/W*
T0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�
Conv2D_2/b/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*
_class
loc:@Conv2D_2/b*
valueB�*    
�

Conv2D_2/b
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@Conv2D_2/b*
	container *
shape:�
�
Conv2D_2/b/AssignAssign
Conv2D_2/bConv2D_2/b/Initializer/Const*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
l
Conv2D_2/b/readIdentity
Conv2D_2/b*
T0*
_class
loc:@Conv2D_2/b*
_output_shapes	
:�
�
Conv2D_2/Conv2DConv2DMaxPool2D_1/MaxPoolConv2D_2/W/read*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
�
Conv2D_2/BiasAddBiasAddConv2D_2/Conv2DConv2D_2/b/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
b
Conv2D_2/ReluReluConv2D_2/BiasAdd*
T0*0
_output_shapes
:����������
�
MaxPool2D_2/MaxPoolMaxPoolConv2D_2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
�
+Conv2D_3/W/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@Conv2D_3/W*%
valueB"      �   @   
�
)Conv2D_3/W/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_3/W*
valueB
 *����
�
)Conv2D_3/W/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_3/W*
valueB
 *���<
�
3Conv2D_3/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_3/W/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:�@*

seed *
T0*
_class
loc:@Conv2D_3/W*
seed2 
�
)Conv2D_3/W/Initializer/random_uniform/subSub)Conv2D_3/W/Initializer/random_uniform/max)Conv2D_3/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_3/W*
_output_shapes
: 
�
)Conv2D_3/W/Initializer/random_uniform/mulMul3Conv2D_3/W/Initializer/random_uniform/RandomUniform)Conv2D_3/W/Initializer/random_uniform/sub*'
_output_shapes
:�@*
T0*
_class
loc:@Conv2D_3/W
�
%Conv2D_3/W/Initializer/random_uniformAdd)Conv2D_3/W/Initializer/random_uniform/mul)Conv2D_3/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�

Conv2D_3/W
VariableV2*
	container *
shape:�@*
dtype0*'
_output_shapes
:�@*
shared_name *
_class
loc:@Conv2D_3/W
�
Conv2D_3/W/AssignAssign
Conv2D_3/W%Conv2D_3/W/Initializer/random_uniform*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
x
Conv2D_3/W/readIdentity
Conv2D_3/W*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
Conv2D_3/b/Initializer/ConstConst*
dtype0*
_output_shapes
:@*
_class
loc:@Conv2D_3/b*
valueB@*    
�

Conv2D_3/b
VariableV2*
shared_name *
_class
loc:@Conv2D_3/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
Conv2D_3/b/AssignAssign
Conv2D_3/bConv2D_3/b/Initializer/Const*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
k
Conv2D_3/b/readIdentity
Conv2D_3/b*
T0*
_class
loc:@Conv2D_3/b*
_output_shapes
:@
�
Conv2D_3/Conv2DConv2DMaxPool2D_2/MaxPoolConv2D_3/W/read*/
_output_shapes
:���������@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
�
Conv2D_3/BiasAddBiasAddConv2D_3/Conv2DConv2D_3/b/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
a
Conv2D_3/ReluReluConv2D_3/BiasAdd*
T0*/
_output_shapes
:���������@
�
MaxPool2D_3/MaxPoolMaxPoolConv2D_3/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@*
T0
�
+Conv2D_4/W/Initializer/random_uniform/shapeConst*
_class
loc:@Conv2D_4/W*%
valueB"      @       *
dtype0*
_output_shapes
:
�
)Conv2D_4/W/Initializer/random_uniform/minConst*
_class
loc:@Conv2D_4/W*
valueB
 *�\1�*
dtype0*
_output_shapes
: 
�
)Conv2D_4/W/Initializer/random_uniform/maxConst*
_class
loc:@Conv2D_4/W*
valueB
 *�\1=*
dtype0*
_output_shapes
: 
�
3Conv2D_4/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_4/W/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
:@ *

seed *
T0*
_class
loc:@Conv2D_4/W
�
)Conv2D_4/W/Initializer/random_uniform/subSub)Conv2D_4/W/Initializer/random_uniform/max)Conv2D_4/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_4/W*
_output_shapes
: 
�
)Conv2D_4/W/Initializer/random_uniform/mulMul3Conv2D_4/W/Initializer/random_uniform/RandomUniform)Conv2D_4/W/Initializer/random_uniform/sub*&
_output_shapes
:@ *
T0*
_class
loc:@Conv2D_4/W
�
%Conv2D_4/W/Initializer/random_uniformAdd)Conv2D_4/W/Initializer/random_uniform/mul)Conv2D_4/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�

Conv2D_4/W
VariableV2*
_class
loc:@Conv2D_4/W*
	container *
shape:@ *
dtype0*&
_output_shapes
:@ *
shared_name 
�
Conv2D_4/W/AssignAssign
Conv2D_4/W%Conv2D_4/W/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
w
Conv2D_4/W/readIdentity
Conv2D_4/W*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�
Conv2D_4/b/Initializer/ConstConst*
_class
loc:@Conv2D_4/b*
valueB *    *
dtype0*
_output_shapes
: 
�

Conv2D_4/b
VariableV2*
shared_name *
_class
loc:@Conv2D_4/b*
	container *
shape: *
dtype0*
_output_shapes
: 
�
Conv2D_4/b/AssignAssign
Conv2D_4/bConv2D_4/b/Initializer/Const*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
k
Conv2D_4/b/readIdentity
Conv2D_4/b*
_output_shapes
: *
T0*
_class
loc:@Conv2D_4/b
�
Conv2D_4/Conv2DConv2DMaxPool2D_3/MaxPoolConv2D_4/W/read*/
_output_shapes
:��������� *
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
�
Conv2D_4/BiasAddBiasAddConv2D_4/Conv2DConv2D_4/b/read*
data_formatNHWC*/
_output_shapes
:��������� *
T0
a
Conv2D_4/ReluReluConv2D_4/BiasAdd*
T0*/
_output_shapes
:��������� 
�
MaxPool2D_4/MaxPoolMaxPoolConv2D_4/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:��������� 
�
3FullyConnected/W/Initializer/truncated_normal/shapeConst*#
_class
loc:@FullyConnected/W*
valueB"       *
dtype0*
_output_shapes
:
�
2FullyConnected/W/Initializer/truncated_normal/meanConst*#
_class
loc:@FullyConnected/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
4FullyConnected/W/Initializer/truncated_normal/stddevConst*#
_class
loc:@FullyConnected/W*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
=FullyConnected/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3FullyConnected/W/Initializer/truncated_normal/shape*

seed *
T0*#
_class
loc:@FullyConnected/W*
seed2 *
dtype0*
_output_shapes
:	 �
�
1FullyConnected/W/Initializer/truncated_normal/mulMul=FullyConnected/W/Initializer/truncated_normal/TruncatedNormal4FullyConnected/W/Initializer/truncated_normal/stddev*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
-FullyConnected/W/Initializer/truncated_normalAdd1FullyConnected/W/Initializer/truncated_normal/mul2FullyConnected/W/Initializer/truncated_normal/mean*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
FullyConnected/W
VariableV2*
dtype0*
_output_shapes
:	 �*
shared_name *#
_class
loc:@FullyConnected/W*
	container *
shape:	 �
�
FullyConnected/W/AssignAssignFullyConnected/W-FullyConnected/W/Initializer/truncated_normal*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
�
FullyConnected/W/readIdentityFullyConnected/W*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
"FullyConnected/b/Initializer/ConstConst*#
_class
loc:@FullyConnected/b*
valueB�*    *
dtype0*
_output_shapes	
:�
�
FullyConnected/b
VariableV2*#
_class
loc:@FullyConnected/b*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
FullyConnected/b/AssignAssignFullyConnected/b"FullyConnected/b/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
~
FullyConnected/b/readIdentityFullyConnected/b*
T0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
m
FullyConnected/Reshape/shapeConst*
valueB"����    *
dtype0*
_output_shapes
:
�
FullyConnected/ReshapeReshapeMaxPool2D_4/MaxPoolFullyConnected/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:��������� 
�
FullyConnected/MatMulMatMulFullyConnected/ReshapeFullyConnected/W/read*
T0*
transpose_a( *(
_output_shapes
:����������*
transpose_b( 
�
FullyConnected/BiasAddBiasAddFullyConnected/MatMulFullyConnected/b/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
f
FullyConnected/ReluReluFullyConnected/BiasAdd*
T0*(
_output_shapes
:����������

is_training/Initializer/ConstConst*
_class
loc:@is_training*
value	B
 Z *
dtype0
*
_output_shapes
: 
�
is_training
VariableV2*
_class
loc:@is_training*
	container *
shape: *
dtype0
*
_output_shapes
: *
shared_name 
�
is_training/AssignAssignis_trainingis_training/Initializer/Const*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: 
j
is_training/readIdentityis_training*
T0
*
_class
loc:@is_training*
_output_shapes
: 
V
Dropout/Assign/valueConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
Dropout/AssignAssignis_trainingDropout/Assign/value*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: 
X
Dropout/Assign_1/valueConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
�
Dropout/Assign_1Assignis_trainingDropout/Assign_1/value*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: *
use_locking(
_
Dropout/cond/SwitchSwitchis_trainingis_training/read*
T0
*
_output_shapes
: : 
Y
Dropout/cond/switch_tIdentityDropout/cond/Switch:1*
T0
*
_output_shapes
: 
W
Dropout/cond/switch_fIdentityDropout/cond/Switch*
T0
*
_output_shapes
: 
S
Dropout/cond/pred_idIdentityis_training/read*
_output_shapes
: *
T0

v
Dropout/cond/dropout/rateConst^Dropout/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *��L>
}
Dropout/cond/dropout/ShapeShape#Dropout/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
�
!Dropout/cond/dropout/Shape/SwitchSwitchFullyConnected/ReluDropout/cond/pred_id*
T0*&
_class
loc:@FullyConnected/Relu*<
_output_shapes*
(:����������:����������
�
'Dropout/cond/dropout/random_uniform/minConst^Dropout/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'Dropout/cond/dropout/random_uniform/maxConst^Dropout/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
1Dropout/cond/dropout/random_uniform/RandomUniformRandomUniformDropout/cond/dropout/Shape*
T0*
dtype0*
seed2 *(
_output_shapes
:����������*

seed 
�
'Dropout/cond/dropout/random_uniform/subSub'Dropout/cond/dropout/random_uniform/max'Dropout/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
'Dropout/cond/dropout/random_uniform/mulMul1Dropout/cond/dropout/random_uniform/RandomUniform'Dropout/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
#Dropout/cond/dropout/random_uniformAdd'Dropout/cond/dropout/random_uniform/mul'Dropout/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
w
Dropout/cond/dropout/sub/xConst^Dropout/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
w
Dropout/cond/dropout/subSubDropout/cond/dropout/sub/xDropout/cond/dropout/rate*
T0*
_output_shapes
: 
{
Dropout/cond/dropout/truediv/xConst^Dropout/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dropout/cond/dropout/truedivRealDivDropout/cond/dropout/truediv/xDropout/cond/dropout/sub*
T0*
_output_shapes
: 
�
!Dropout/cond/dropout/GreaterEqualGreaterEqual#Dropout/cond/dropout/random_uniformDropout/cond/dropout/rate*
T0*(
_output_shapes
:����������
�
Dropout/cond/dropout/mulMul#Dropout/cond/dropout/Shape/Switch:1Dropout/cond/dropout/truediv*
T0*(
_output_shapes
:����������
�
Dropout/cond/dropout/CastCast!Dropout/cond/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*(
_output_shapes
:����������
�
Dropout/cond/dropout/mul_1MulDropout/cond/dropout/mulDropout/cond/dropout/Cast*
T0*(
_output_shapes
:����������
�
Dropout/cond/Switch_1SwitchFullyConnected/ReluDropout/cond/pred_id*
T0*&
_class
loc:@FullyConnected/Relu*<
_output_shapes*
(:����������:����������
�
Dropout/cond/MergeMergeDropout/cond/Switch_1Dropout/cond/dropout/mul_1*
T0*
N**
_output_shapes
:����������: 
�
5FullyConnected_1/W/Initializer/truncated_normal/shapeConst*%
_class
loc:@FullyConnected_1/W*
valueB"      *
dtype0*
_output_shapes
:
�
4FullyConnected_1/W/Initializer/truncated_normal/meanConst*%
_class
loc:@FullyConnected_1/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6FullyConnected_1/W/Initializer/truncated_normal/stddevConst*%
_class
loc:@FullyConnected_1/W*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
?FullyConnected_1/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5FullyConnected_1/W/Initializer/truncated_normal/shape*

seed *
T0*%
_class
loc:@FullyConnected_1/W*
seed2 *
dtype0*
_output_shapes
:	�
�
3FullyConnected_1/W/Initializer/truncated_normal/mulMul?FullyConnected_1/W/Initializer/truncated_normal/TruncatedNormal6FullyConnected_1/W/Initializer/truncated_normal/stddev*
T0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
�
/FullyConnected_1/W/Initializer/truncated_normalAdd3FullyConnected_1/W/Initializer/truncated_normal/mul4FullyConnected_1/W/Initializer/truncated_normal/mean*
T0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
�
FullyConnected_1/W
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *%
_class
loc:@FullyConnected_1/W*
	container *
shape:	�
�
FullyConnected_1/W/AssignAssignFullyConnected_1/W/FullyConnected_1/W/Initializer/truncated_normal*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
FullyConnected_1/W/readIdentityFullyConnected_1/W*
T0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
�
$FullyConnected_1/b/Initializer/ConstConst*%
_class
loc:@FullyConnected_1/b*
valueB*    *
dtype0*
_output_shapes
:
�
FullyConnected_1/b
VariableV2*
shared_name *%
_class
loc:@FullyConnected_1/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
FullyConnected_1/b/AssignAssignFullyConnected_1/b$FullyConnected_1/b/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
�
FullyConnected_1/b/readIdentityFullyConnected_1/b*
T0*%
_class
loc:@FullyConnected_1/b*
_output_shapes
:
�
FullyConnected_1/MatMulMatMulDropout/cond/MergeFullyConnected_1/W/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
FullyConnected_1/BiasAddBiasAddFullyConnected_1/MatMulFullyConnected_1/b/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
o
FullyConnected_1/SoftmaxSoftmaxFullyConnected_1/BiasAdd*
T0*'
_output_shapes
:���������
l
	targets/YPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
[
Accuracy/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
Accuracy/ArgMaxArgMaxFullyConnected_1/SoftmaxAccuracy/ArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
]
Accuracy/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
Accuracy/ArgMax_1ArgMax	targets/YAccuracy/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
�
Accuracy/EqualEqualAccuracy/ArgMaxAccuracy/ArgMax_1*
T0	*#
_output_shapes
:���������*
incompatible_shape_error(
r
Accuracy/CastCastAccuracy/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:���������
X
Accuracy/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
r
Accuracy/MeanMeanAccuracy/CastAccuracy/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
d
"Crossentropy/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
Crossentropy/SumSumFullyConnected_1/Softmax"Crossentropy/Sum/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
}
Crossentropy/truedivRealDivFullyConnected_1/SoftmaxCrossentropy/Sum*
T0*'
_output_shapes
:���������
X
Crossentropy/Cast/xConst*
dtype0*
_output_shapes
: *
valueB
 *���.
Z
Crossentropy/Cast_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
"Crossentropy/clip_by_value/MinimumMinimumCrossentropy/truedivCrossentropy/Cast_1/x*
T0*'
_output_shapes
:���������
�
Crossentropy/clip_by_valueMaximum"Crossentropy/clip_by_value/MinimumCrossentropy/Cast/x*
T0*'
_output_shapes
:���������
e
Crossentropy/LogLogCrossentropy/clip_by_value*
T0*'
_output_shapes
:���������
f
Crossentropy/mulMul	targets/YCrossentropy/Log*
T0*'
_output_shapes
:���������
f
$Crossentropy/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
Crossentropy/Sum_1SumCrossentropy/mul$Crossentropy/Sum_1/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
Y
Crossentropy/NegNegCrossentropy/Sum_1*
T0*#
_output_shapes
:���������
\
Crossentropy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
}
Crossentropy/MeanMeanCrossentropy/NegCrossentropy/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
`
Training_step/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
Training_step
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
Training_step/AssignAssignTraining_stepTraining_step/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@Training_step
p
Training_step/readIdentityTraining_step*
_output_shapes
: *
T0* 
_class
loc:@Training_step
^
Global_Step/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
o
Global_Step
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
Global_Step/AssignAssignGlobal_StepGlobal_Step/initial_value*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: *
use_locking(
j
Global_Step/readIdentityGlobal_Step*
T0*
_class
loc:@Global_Step*
_output_shapes
: 
J
Add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
D
AddAddGlobal_Step/readAdd/y*
T0*
_output_shapes
: 
�
AssignAssignGlobal_StepAdd*
use_locking(*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: 
[
val_loss/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
l
val_loss
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
val_loss/AssignAssignval_lossval_loss/initial_value*
use_locking(*
T0*
_class
loc:@val_loss*
validate_shape(*
_output_shapes
: 
a
val_loss/readIdentityval_loss*
_output_shapes
: *
T0*
_class
loc:@val_loss
Z
val_acc/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
val_acc
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
val_acc/AssignAssignval_accval_acc/initial_value*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: *
use_locking(
^
val_acc/readIdentityval_acc*
_output_shapes
: *
T0*
_class
loc:@val_acc
Y
placeholder/val_lossPlaceholder*
shape:*
dtype0*
_output_shapes
:
X
placeholder/val_accPlaceholder*
dtype0*
_output_shapes
:*
shape:
�
assign/val_lossAssignval_lossplaceholder/val_loss*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@val_loss
�
assign/val_accAssignval_accplaceholder/val_acc*
use_locking(*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: 
�
*Accuracy/Mean/moving_avg/Initializer/zerosConst*
dtype0*
_output_shapes
: *+
_class!
loc:@Accuracy/Mean/moving_avg*
valueB
 *    
�
Accuracy/Mean/moving_avg
VariableV2*
dtype0*
_output_shapes
: *
shared_name *+
_class!
loc:@Accuracy/Mean/moving_avg*
	container *
shape: 
�
Accuracy/Mean/moving_avg/AssignAssignAccuracy/Mean/moving_avg*Accuracy/Mean/moving_avg/Initializer/zeros*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: *
use_locking(
�
Accuracy/Mean/moving_avg/readIdentityAccuracy/Mean/moving_avg*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
_output_shapes
: 
U
moving_avg/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
moving_avg/add/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
^
moving_avg/addAddV2moving_avg/add/xTraining_step/read*
T0*
_output_shapes
: 
W
moving_avg/add_1/xConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
b
moving_avg/add_1AddV2moving_avg/add_1/xTraining_step/read*
T0*
_output_shapes
: 
`
moving_avg/truedivRealDivmoving_avg/addmoving_avg/add_1*
T0*
_output_shapes
: 
d
moving_avg/MinimumMinimummoving_avg/decaymoving_avg/truediv*
_output_shapes
: *
T0
e
 moving_avg/AssignMovingAvg/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
|
moving_avg/AssignMovingAvg/subSub moving_avg/AssignMovingAvg/sub/xmoving_avg/Minimum*
T0*
_output_shapes
: 
v
 moving_avg/AssignMovingAvg/sub_1SubAccuracy/Mean/moving_avg/readAccuracy/Mean*
T0*
_output_shapes
: 
�
moving_avg/AssignMovingAvg/mulMul moving_avg/AssignMovingAvg/sub_1moving_avg/AssignMovingAvg/sub*
T0*
_output_shapes
: 
�
moving_avg/AssignMovingAvg	AssignSubAccuracy/Mean/moving_avgmoving_avg/AssignMovingAvg/mul*
use_locking( *
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
_output_shapes
: 
/

moving_avgNoOp^moving_avg/AssignMovingAvg
O
Adam/Total_LossIdentityCrossentropy/Mean*
T0*
_output_shapes
: 
�
.Crossentropy/Mean/moving_avg/Initializer/zerosConst*
dtype0*
_output_shapes
: */
_class%
#!loc:@Crossentropy/Mean/moving_avg*
valueB
 *    
�
Crossentropy/Mean/moving_avg
VariableV2*
shared_name */
_class%
#!loc:@Crossentropy/Mean/moving_avg*
	container *
shape: *
dtype0*
_output_shapes
: 
�
#Crossentropy/Mean/moving_avg/AssignAssignCrossentropy/Mean/moving_avg.Crossentropy/Mean/moving_avg/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
�
!Crossentropy/Mean/moving_avg/readIdentityCrossentropy/Mean/moving_avg*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
_output_shapes
: 
Z
Adam/moving_avg/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Z
Adam/moving_avg/add/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
h
Adam/moving_avg/addAddV2Adam/moving_avg/add/xTraining_step/read*
_output_shapes
: *
T0
\
Adam/moving_avg/add_1/xConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
l
Adam/moving_avg/add_1AddV2Adam/moving_avg/add_1/xTraining_step/read*
T0*
_output_shapes
: 
o
Adam/moving_avg/truedivRealDivAdam/moving_avg/addAdam/moving_avg/add_1*
T0*
_output_shapes
: 
s
Adam/moving_avg/MinimumMinimumAdam/moving_avg/decayAdam/moving_avg/truediv*
T0*
_output_shapes
: 
j
%Adam/moving_avg/AssignMovingAvg/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
#Adam/moving_avg/AssignMovingAvg/subSub%Adam/moving_avg/AssignMovingAvg/sub/xAdam/moving_avg/Minimum*
_output_shapes
: *
T0
�
%Adam/moving_avg/AssignMovingAvg/sub_1Sub!Crossentropy/Mean/moving_avg/readCrossentropy/Mean*
_output_shapes
: *
T0
�
#Adam/moving_avg/AssignMovingAvg/mulMul%Adam/moving_avg/AssignMovingAvg/sub_1#Adam/moving_avg/AssignMovingAvg/sub*
T0*
_output_shapes
: 
�
Adam/moving_avg/AssignMovingAvg	AssignSubCrossentropy/Mean/moving_avg#Adam/moving_avg/AssignMovingAvg/mul*
_output_shapes
: *
use_locking( *
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg
9
Adam/moving_avgNoOp ^Adam/moving_avg/AssignMovingAvg
N
	Loss/tagsConst*
valueB
 BLoss*
dtype0*
_output_shapes
: 
d
LossScalarSummary	Loss/tags!Crossentropy/Mean/moving_avg/read*
T0*
_output_shapes
: 
`
Adam/Loss/raw/tagsConst*
valueB BAdam/Loss/raw*
dtype0*
_output_shapes
: 
f
Adam/Loss/rawScalarSummaryAdam/Loss/raw/tagsCrossentropy/Mean*
T0*
_output_shapes
: 
v
Adam/gradients/ShapeConst^Adam/moving_avg^moving_avg*
valueB *
dtype0*
_output_shapes
: 
|
Adam/gradients/grad_ys_0Const^Adam/moving_avg^moving_avg*
valueB
 *  �?*
dtype0*
_output_shapes
: 
~
Adam/gradients/FillFillAdam/gradients/ShapeAdam/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
3Adam/gradients/Crossentropy/Mean_grad/Reshape/shapeConst^Adam/moving_avg^moving_avg*
valueB:*
dtype0*
_output_shapes
:
�
-Adam/gradients/Crossentropy/Mean_grad/ReshapeReshapeAdam/gradients/Fill3Adam/gradients/Crossentropy/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
�
+Adam/gradients/Crossentropy/Mean_grad/ShapeShapeCrossentropy/Neg^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
*Adam/gradients/Crossentropy/Mean_grad/TileTile-Adam/gradients/Crossentropy/Mean_grad/Reshape+Adam/gradients/Crossentropy/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
�
-Adam/gradients/Crossentropy/Mean_grad/Shape_1ShapeCrossentropy/Neg^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
-Adam/gradients/Crossentropy/Mean_grad/Shape_2Const^Adam/moving_avg^moving_avg*
valueB *
dtype0*
_output_shapes
: 
�
+Adam/gradients/Crossentropy/Mean_grad/ConstConst^Adam/moving_avg^moving_avg*
valueB: *
dtype0*
_output_shapes
:
�
*Adam/gradients/Crossentropy/Mean_grad/ProdProd-Adam/gradients/Crossentropy/Mean_grad/Shape_1+Adam/gradients/Crossentropy/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
-Adam/gradients/Crossentropy/Mean_grad/Const_1Const^Adam/moving_avg^moving_avg*
valueB: *
dtype0*
_output_shapes
:
�
,Adam/gradients/Crossentropy/Mean_grad/Prod_1Prod-Adam/gradients/Crossentropy/Mean_grad/Shape_2-Adam/gradients/Crossentropy/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
/Adam/gradients/Crossentropy/Mean_grad/Maximum/yConst^Adam/moving_avg^moving_avg*
value	B :*
dtype0*
_output_shapes
: 
�
-Adam/gradients/Crossentropy/Mean_grad/MaximumMaximum,Adam/gradients/Crossentropy/Mean_grad/Prod_1/Adam/gradients/Crossentropy/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
.Adam/gradients/Crossentropy/Mean_grad/floordivFloorDiv*Adam/gradients/Crossentropy/Mean_grad/Prod-Adam/gradients/Crossentropy/Mean_grad/Maximum*
_output_shapes
: *
T0
�
*Adam/gradients/Crossentropy/Mean_grad/CastCast.Adam/gradients/Crossentropy/Mean_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
�
-Adam/gradients/Crossentropy/Mean_grad/truedivRealDiv*Adam/gradients/Crossentropy/Mean_grad/Tile*Adam/gradients/Crossentropy/Mean_grad/Cast*
T0*#
_output_shapes
:���������
�
(Adam/gradients/Crossentropy/Neg_grad/NegNeg-Adam/gradients/Crossentropy/Mean_grad/truediv*
T0*#
_output_shapes
:���������
�
,Adam/gradients/Crossentropy/Sum_1_grad/ShapeShapeCrossentropy/mul^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
+Adam/gradients/Crossentropy/Sum_1_grad/SizeConst^Adam/moving_avg^moving_avg*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
*Adam/gradients/Crossentropy/Sum_1_grad/addAddV2$Crossentropy/Sum_1/reduction_indices+Adam/gradients/Crossentropy/Sum_1_grad/Size*
_output_shapes
: *
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape
�
*Adam/gradients/Crossentropy/Sum_1_grad/modFloorMod*Adam/gradients/Crossentropy/Sum_1_grad/add+Adam/gradients/Crossentropy/Sum_1_grad/Size*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
_output_shapes
: 
�
.Adam/gradients/Crossentropy/Sum_1_grad/Shape_1Const^Adam/moving_avg^moving_avg*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
2Adam/gradients/Crossentropy/Sum_1_grad/range/startConst^Adam/moving_avg^moving_avg*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
2Adam/gradients/Crossentropy/Sum_1_grad/range/deltaConst^Adam/moving_avg^moving_avg*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
,Adam/gradients/Crossentropy/Sum_1_grad/rangeRange2Adam/gradients/Crossentropy/Sum_1_grad/range/start+Adam/gradients/Crossentropy/Sum_1_grad/Size2Adam/gradients/Crossentropy/Sum_1_grad/range/delta*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
_output_shapes
:*

Tidx0
�
1Adam/gradients/Crossentropy/Sum_1_grad/Fill/valueConst^Adam/moving_avg^moving_avg*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
+Adam/gradients/Crossentropy/Sum_1_grad/FillFill.Adam/gradients/Crossentropy/Sum_1_grad/Shape_11Adam/gradients/Crossentropy/Sum_1_grad/Fill/value*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*

index_type0*
_output_shapes
: 
�
4Adam/gradients/Crossentropy/Sum_1_grad/DynamicStitchDynamicStitch,Adam/gradients/Crossentropy/Sum_1_grad/range*Adam/gradients/Crossentropy/Sum_1_grad/mod,Adam/gradients/Crossentropy/Sum_1_grad/Shape+Adam/gradients/Crossentropy/Sum_1_grad/Fill*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
N*
_output_shapes
:
�
0Adam/gradients/Crossentropy/Sum_1_grad/Maximum/yConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
value	B :
�
.Adam/gradients/Crossentropy/Sum_1_grad/MaximumMaximum4Adam/gradients/Crossentropy/Sum_1_grad/DynamicStitch0Adam/gradients/Crossentropy/Sum_1_grad/Maximum/y*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
_output_shapes
:
�
/Adam/gradients/Crossentropy/Sum_1_grad/floordivFloorDiv,Adam/gradients/Crossentropy/Sum_1_grad/Shape.Adam/gradients/Crossentropy/Sum_1_grad/Maximum*
_output_shapes
:*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape
�
.Adam/gradients/Crossentropy/Sum_1_grad/ReshapeReshape(Adam/gradients/Crossentropy/Neg_grad/Neg4Adam/gradients/Crossentropy/Sum_1_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
+Adam/gradients/Crossentropy/Sum_1_grad/TileTile.Adam/gradients/Crossentropy/Sum_1_grad/Reshape/Adam/gradients/Crossentropy/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
�
*Adam/gradients/Crossentropy/mul_grad/ShapeShape	targets/Y^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
,Adam/gradients/Crossentropy/mul_grad/Shape_1ShapeCrossentropy/Log^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
:Adam/gradients/Crossentropy/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*Adam/gradients/Crossentropy/mul_grad/Shape,Adam/gradients/Crossentropy/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
(Adam/gradients/Crossentropy/mul_grad/MulMul+Adam/gradients/Crossentropy/Sum_1_grad/TileCrossentropy/Log*
T0*'
_output_shapes
:���������
�
(Adam/gradients/Crossentropy/mul_grad/SumSum(Adam/gradients/Crossentropy/mul_grad/Mul:Adam/gradients/Crossentropy/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
,Adam/gradients/Crossentropy/mul_grad/ReshapeReshape(Adam/gradients/Crossentropy/mul_grad/Sum*Adam/gradients/Crossentropy/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
*Adam/gradients/Crossentropy/mul_grad/Mul_1Mul	targets/Y+Adam/gradients/Crossentropy/Sum_1_grad/Tile*
T0*'
_output_shapes
:���������
�
*Adam/gradients/Crossentropy/mul_grad/Sum_1Sum*Adam/gradients/Crossentropy/mul_grad/Mul_1<Adam/gradients/Crossentropy/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
.Adam/gradients/Crossentropy/mul_grad/Reshape_1Reshape*Adam/gradients/Crossentropy/mul_grad/Sum_1,Adam/gradients/Crossentropy/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/Adam/gradients/Crossentropy/Log_grad/Reciprocal
ReciprocalCrossentropy/clip_by_value/^Adam/gradients/Crossentropy/mul_grad/Reshape_1^Adam/moving_avg^moving_avg*
T0*'
_output_shapes
:���������
�
(Adam/gradients/Crossentropy/Log_grad/mulMul.Adam/gradients/Crossentropy/mul_grad/Reshape_1/Adam/gradients/Crossentropy/Log_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
4Adam/gradients/Crossentropy/clip_by_value_grad/ShapeShape"Crossentropy/clip_by_value/Minimum^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
�
6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_1Const^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB 
�
6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_2Shape(Adam/gradients/Crossentropy/Log_grad/mul*
T0*
out_type0*
_output_shapes
:
�
:Adam/gradients/Crossentropy/clip_by_value_grad/zeros/ConstConst^Adam/moving_avg^moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
�
4Adam/gradients/Crossentropy/clip_by_value_grad/zerosFill6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_2:Adam/gradients/Crossentropy/clip_by_value_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������
�
;Adam/gradients/Crossentropy/clip_by_value_grad/GreaterEqualGreaterEqual"Crossentropy/clip_by_value/MinimumCrossentropy/Cast/x^Adam/moving_avg^moving_avg*
T0*'
_output_shapes
:���������
�
DAdam/gradients/Crossentropy/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs4Adam/gradients/Crossentropy/clip_by_value_grad/Shape6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5Adam/gradients/Crossentropy/clip_by_value_grad/SelectSelect;Adam/gradients/Crossentropy/clip_by_value_grad/GreaterEqual(Adam/gradients/Crossentropy/Log_grad/mul4Adam/gradients/Crossentropy/clip_by_value_grad/zeros*
T0*'
_output_shapes
:���������
�
2Adam/gradients/Crossentropy/clip_by_value_grad/SumSum5Adam/gradients/Crossentropy/clip_by_value_grad/SelectDAdam/gradients/Crossentropy/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
6Adam/gradients/Crossentropy/clip_by_value_grad/ReshapeReshape2Adam/gradients/Crossentropy/clip_by_value_grad/Sum4Adam/gradients/Crossentropy/clip_by_value_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
7Adam/gradients/Crossentropy/clip_by_value_grad/Select_1Select;Adam/gradients/Crossentropy/clip_by_value_grad/GreaterEqual4Adam/gradients/Crossentropy/clip_by_value_grad/zeros(Adam/gradients/Crossentropy/Log_grad/mul*
T0*'
_output_shapes
:���������
�
4Adam/gradients/Crossentropy/clip_by_value_grad/Sum_1Sum7Adam/gradients/Crossentropy/clip_by_value_grad/Select_1FAdam/gradients/Crossentropy/clip_by_value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
8Adam/gradients/Crossentropy/clip_by_value_grad/Reshape_1Reshape4Adam/gradients/Crossentropy/clip_by_value_grad/Sum_16Adam/gradients/Crossentropy/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/ShapeShapeCrossentropy/truediv^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_1Const^Adam/moving_avg^moving_avg*
valueB *
dtype0*
_output_shapes
: 
�
>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_2Shape6Adam/gradients/Crossentropy/clip_by_value_grad/Reshape*
T0*
out_type0*
_output_shapes
:
�
BAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros/ConstConst^Adam/moving_avg^moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/zerosFill>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_2BAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������
�
@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/LessEqual	LessEqualCrossentropy/truedivCrossentropy/Cast_1/x^Adam/moving_avg^moving_avg*
T0*'
_output_shapes
:���������
�
LAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
=Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SelectSelect@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/LessEqual6Adam/gradients/Crossentropy/clip_by_value_grad/Reshape<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:���������*
T0
�
:Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SumSum=Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SelectLAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/ReshapeReshape:Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Sum<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
?Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Select_1Select@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/LessEqual<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros6Adam/gradients/Crossentropy/clip_by_value_grad/Reshape*
T0*'
_output_shapes
:���������
�
<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Sum_1Sum?Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Select_1NAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Reshape_1Reshape<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Sum_1>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
.Adam/gradients/Crossentropy/truediv_grad/ShapeShapeFullyConnected_1/Softmax^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
0Adam/gradients/Crossentropy/truediv_grad/Shape_1ShapeCrossentropy/Sum^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
>Adam/gradients/Crossentropy/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs.Adam/gradients/Crossentropy/truediv_grad/Shape0Adam/gradients/Crossentropy/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0Adam/gradients/Crossentropy/truediv_grad/RealDivRealDiv>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/ReshapeCrossentropy/Sum*
T0*'
_output_shapes
:���������
�
,Adam/gradients/Crossentropy/truediv_grad/SumSum0Adam/gradients/Crossentropy/truediv_grad/RealDiv>Adam/gradients/Crossentropy/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
0Adam/gradients/Crossentropy/truediv_grad/ReshapeReshape,Adam/gradients/Crossentropy/truediv_grad/Sum.Adam/gradients/Crossentropy/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
,Adam/gradients/Crossentropy/truediv_grad/NegNegFullyConnected_1/Softmax^Adam/moving_avg^moving_avg*
T0*'
_output_shapes
:���������
�
2Adam/gradients/Crossentropy/truediv_grad/RealDiv_1RealDiv,Adam/gradients/Crossentropy/truediv_grad/NegCrossentropy/Sum*
T0*'
_output_shapes
:���������
�
2Adam/gradients/Crossentropy/truediv_grad/RealDiv_2RealDiv2Adam/gradients/Crossentropy/truediv_grad/RealDiv_1Crossentropy/Sum*
T0*'
_output_shapes
:���������
�
,Adam/gradients/Crossentropy/truediv_grad/mulMul>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Reshape2Adam/gradients/Crossentropy/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������
�
.Adam/gradients/Crossentropy/truediv_grad/Sum_1Sum,Adam/gradients/Crossentropy/truediv_grad/mul@Adam/gradients/Crossentropy/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
2Adam/gradients/Crossentropy/truediv_grad/Reshape_1Reshape.Adam/gradients/Crossentropy/truediv_grad/Sum_10Adam/gradients/Crossentropy/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
*Adam/gradients/Crossentropy/Sum_grad/ShapeShapeFullyConnected_1/Softmax^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
)Adam/gradients/Crossentropy/Sum_grad/SizeConst^Adam/moving_avg^moving_avg*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
(Adam/gradients/Crossentropy/Sum_grad/addAddV2"Crossentropy/Sum/reduction_indices)Adam/gradients/Crossentropy/Sum_grad/Size*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
: 
�
(Adam/gradients/Crossentropy/Sum_grad/modFloorMod(Adam/gradients/Crossentropy/Sum_grad/add)Adam/gradients/Crossentropy/Sum_grad/Size*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
: 
�
,Adam/gradients/Crossentropy/Sum_grad/Shape_1Const^Adam/moving_avg^moving_avg*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
0Adam/gradients/Crossentropy/Sum_grad/range/startConst^Adam/moving_avg^moving_avg*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
0Adam/gradients/Crossentropy/Sum_grad/range/deltaConst^Adam/moving_avg^moving_avg*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
*Adam/gradients/Crossentropy/Sum_grad/rangeRange0Adam/gradients/Crossentropy/Sum_grad/range/start)Adam/gradients/Crossentropy/Sum_grad/Size0Adam/gradients/Crossentropy/Sum_grad/range/delta*

Tidx0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
:
�
/Adam/gradients/Crossentropy/Sum_grad/Fill/valueConst^Adam/moving_avg^moving_avg*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
)Adam/gradients/Crossentropy/Sum_grad/FillFill,Adam/gradients/Crossentropy/Sum_grad/Shape_1/Adam/gradients/Crossentropy/Sum_grad/Fill/value*
_output_shapes
: *
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*

index_type0
�
2Adam/gradients/Crossentropy/Sum_grad/DynamicStitchDynamicStitch*Adam/gradients/Crossentropy/Sum_grad/range(Adam/gradients/Crossentropy/Sum_grad/mod*Adam/gradients/Crossentropy/Sum_grad/Shape)Adam/gradients/Crossentropy/Sum_grad/Fill*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
N*
_output_shapes
:
�
.Adam/gradients/Crossentropy/Sum_grad/Maximum/yConst^Adam/moving_avg^moving_avg*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
,Adam/gradients/Crossentropy/Sum_grad/MaximumMaximum2Adam/gradients/Crossentropy/Sum_grad/DynamicStitch.Adam/gradients/Crossentropy/Sum_grad/Maximum/y*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
:
�
-Adam/gradients/Crossentropy/Sum_grad/floordivFloorDiv*Adam/gradients/Crossentropy/Sum_grad/Shape,Adam/gradients/Crossentropy/Sum_grad/Maximum*
_output_shapes
:*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape
�
,Adam/gradients/Crossentropy/Sum_grad/ReshapeReshape2Adam/gradients/Crossentropy/truediv_grad/Reshape_12Adam/gradients/Crossentropy/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
)Adam/gradients/Crossentropy/Sum_grad/TileTile,Adam/gradients/Crossentropy/Sum_grad/Reshape-Adam/gradients/Crossentropy/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
�
Adam/gradients/AddNAddN0Adam/gradients/Crossentropy/truediv_grad/Reshape)Adam/gradients/Crossentropy/Sum_grad/Tile*
T0*C
_class9
75loc:@Adam/gradients/Crossentropy/truediv_grad/Reshape*
N*'
_output_shapes
:���������
�
0Adam/gradients/FullyConnected_1/Softmax_grad/mulMulAdam/gradients/AddNFullyConnected_1/Softmax*
T0*'
_output_shapes
:���������
�
BAdam/gradients/FullyConnected_1/Softmax_grad/Sum/reduction_indicesConst^Adam/moving_avg^moving_avg*
valueB :
���������*
dtype0*
_output_shapes
: 
�
0Adam/gradients/FullyConnected_1/Softmax_grad/SumSum0Adam/gradients/FullyConnected_1/Softmax_grad/mulBAdam/gradients/FullyConnected_1/Softmax_grad/Sum/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
0Adam/gradients/FullyConnected_1/Softmax_grad/subSubAdam/gradients/AddN0Adam/gradients/FullyConnected_1/Softmax_grad/Sum*
T0*'
_output_shapes
:���������
�
2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1Mul0Adam/gradients/FullyConnected_1/Softmax_grad/subFullyConnected_1/Softmax*'
_output_shapes
:���������*
T0
�
8Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGradBiasAddGrad2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1*
data_formatNHWC*
_output_shapes
:*
T0
�
2Adam/gradients/FullyConnected_1/MatMul_grad/MatMulMatMul2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1FullyConnected_1/W/read*
T0*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
4Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1MatMulDropout/cond/Merge2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1*
transpose_a(*
_output_shapes
:	�*
transpose_b( *
T0
�
0Adam/gradients/Dropout/cond/Merge_grad/cond_gradSwitch2Adam/gradients/FullyConnected_1/MatMul_grad/MatMulDropout/cond/pred_id*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul*<
_output_shapes*
(:����������:����������
�
Adam/gradients/SwitchSwitchFullyConnected/ReluDropout/cond/pred_id^Adam/moving_avg^moving_avg*
T0*<
_output_shapes*
(:����������:����������
o
Adam/gradients/IdentityIdentityAdam/gradients/Switch:1*
T0*(
_output_shapes
:����������
m
Adam/gradients/Shape_1ShapeAdam/gradients/Switch:1*
T0*
out_type0*
_output_shapes
:
�
Adam/gradients/zeros/ConstConst^Adam/gradients/Identity^Adam/moving_avg^moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Adam/gradients/zerosFillAdam/gradients/Shape_1Adam/gradients/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
3Adam/gradients/Dropout/cond/Switch_1_grad/cond_gradMerge0Adam/gradients/Dropout/cond/Merge_grad/cond_gradAdam/gradients/zeros*
N**
_output_shapes
:����������: *
T0
�
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/ShapeShapeDropout/cond/dropout/mul^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
6Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape_1ShapeDropout/cond/dropout/Cast^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
DAdam/gradients/Dropout/cond/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape6Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2Adam/gradients/Dropout/cond/dropout/mul_1_grad/MulMul2Adam/gradients/Dropout/cond/Merge_grad/cond_grad:1Dropout/cond/dropout/Cast*
T0*(
_output_shapes
:����������
�
2Adam/gradients/Dropout/cond/dropout/mul_1_grad/SumSum2Adam/gradients/Dropout/cond/dropout/mul_1_grad/MulDAdam/gradients/Dropout/cond/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
6Adam/gradients/Dropout/cond/dropout/mul_1_grad/ReshapeReshape2Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Mul_1MulDropout/cond/dropout/mul2Adam/gradients/Dropout/cond/Merge_grad/cond_grad:1*
T0*(
_output_shapes
:����������
�
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum_1Sum4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Mul_1FAdam/gradients/Dropout/cond/dropout/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
8Adam/gradients/Dropout/cond/dropout/mul_1_grad/Reshape_1Reshape4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum_16Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/ShapeShape#Dropout/cond/dropout/Shape/Switch:1^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
4Adam/gradients/Dropout/cond/dropout/mul_grad/Shape_1ShapeDropout/cond/dropout/truediv^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
: 
�
BAdam/gradients/Dropout/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2Adam/gradients/Dropout/cond/dropout/mul_grad/Shape4Adam/gradients/Dropout/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0Adam/gradients/Dropout/cond/dropout/mul_grad/MulMul6Adam/gradients/Dropout/cond/dropout/mul_1_grad/ReshapeDropout/cond/dropout/truediv*
T0*(
_output_shapes
:����������
�
0Adam/gradients/Dropout/cond/dropout/mul_grad/SumSum0Adam/gradients/Dropout/cond/dropout/mul_grad/MulBAdam/gradients/Dropout/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
4Adam/gradients/Dropout/cond/dropout/mul_grad/ReshapeReshape0Adam/gradients/Dropout/cond/dropout/mul_grad/Sum2Adam/gradients/Dropout/cond/dropout/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/Mul_1Mul#Dropout/cond/dropout/Shape/Switch:16Adam/gradients/Dropout/cond/dropout/mul_1_grad/Reshape*
T0*(
_output_shapes
:����������
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/Sum_1Sum2Adam/gradients/Dropout/cond/dropout/mul_grad/Mul_1DAdam/gradients/Dropout/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
6Adam/gradients/Dropout/cond/dropout/mul_grad/Reshape_1Reshape2Adam/gradients/Dropout/cond/dropout/mul_grad/Sum_14Adam/gradients/Dropout/cond/dropout/mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Adam/gradients/Switch_1SwitchFullyConnected/ReluDropout/cond/pred_id^Adam/moving_avg^moving_avg*
T0*<
_output_shapes*
(:����������:����������
q
Adam/gradients/Identity_1IdentityAdam/gradients/Switch_1*
T0*(
_output_shapes
:����������
m
Adam/gradients/Shape_2ShapeAdam/gradients/Switch_1*
_output_shapes
:*
T0*
out_type0
�
Adam/gradients/zeros_1/ConstConst^Adam/gradients/Identity_1^Adam/moving_avg^moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Adam/gradients/zeros_1FillAdam/gradients/Shape_2Adam/gradients/zeros_1/Const*
T0*

index_type0*(
_output_shapes
:����������
�
?Adam/gradients/Dropout/cond/dropout/Shape/Switch_grad/cond_gradMergeAdam/gradients/zeros_14Adam/gradients/Dropout/cond/dropout/mul_grad/Reshape*
T0*
N**
_output_shapes
:����������: 
�
Adam/gradients/AddN_1AddN3Adam/gradients/Dropout/cond/Switch_1_grad/cond_grad?Adam/gradients/Dropout/cond/dropout/Shape/Switch_grad/cond_grad*
T0*F
_class<
:8loc:@Adam/gradients/Dropout/cond/Switch_1_grad/cond_grad*
N*(
_output_shapes
:����������
�
0Adam/gradients/FullyConnected/Relu_grad/ReluGradReluGradAdam/gradients/AddN_1FullyConnected/Relu*(
_output_shapes
:����������*
T0
�
6Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGradBiasAddGrad0Adam/gradients/FullyConnected/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
�
0Adam/gradients/FullyConnected/MatMul_grad/MatMulMatMul0Adam/gradients/FullyConnected/Relu_grad/ReluGradFullyConnected/W/read*
T0*
transpose_a( *'
_output_shapes
:��������� *
transpose_b(
�
2Adam/gradients/FullyConnected/MatMul_grad/MatMul_1MatMulFullyConnected/Reshape0Adam/gradients/FullyConnected/Relu_grad/ReluGrad*
T0*
transpose_a(*
_output_shapes
:	 �*
transpose_b( 
�
0Adam/gradients/FullyConnected/Reshape_grad/ShapeShapeMaxPool2D_4/MaxPool^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
�
2Adam/gradients/FullyConnected/Reshape_grad/ReshapeReshape0Adam/gradients/FullyConnected/MatMul_grad/MatMul0Adam/gradients/FullyConnected/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
3Adam/gradients/MaxPool2D_4/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_4/ReluMaxPool2D_4/MaxPool2Adam/gradients/FullyConnected/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:��������� *
T0*
data_formatNHWC*
strides

�
*Adam/gradients/Conv2D_4/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_4/MaxPool_grad/MaxPoolGradConv2D_4/Relu*
T0*/
_output_shapes
:��������� 
�
0Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
*Adam/gradients/Conv2D_4/Conv2D_grad/ShapeNShapeNMaxPool2D_3/MaxPoolConv2D_4/W/read^Adam/moving_avg^moving_avg*
N* 
_output_shapes
::*
T0*
out_type0
�
7Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_4/Conv2D_grad/ShapeNConv2D_4/W/read*Adam/gradients/Conv2D_4/Relu_grad/ReluGrad*
paddingSAME*/
_output_shapes
:���������@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
�
8Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_3/MaxPool,Adam/gradients/Conv2D_4/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_4/Relu_grad/ReluGrad*&
_output_shapes
:@ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
�
3Adam/gradients/MaxPool2D_3/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_3/ReluMaxPool2D_3/MaxPool7Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingSAME*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides

�
*Adam/gradients/Conv2D_3/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_3/MaxPool_grad/MaxPoolGradConv2D_3/Relu*/
_output_shapes
:���������@*
T0
�
0Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
*Adam/gradients/Conv2D_3/Conv2D_grad/ShapeNShapeNMaxPool2D_2/MaxPoolConv2D_3/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
7Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_3/Conv2D_grad/ShapeNConv2D_3/W/read*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*0
_output_shapes
:����������*
	dilations

�
8Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_2/MaxPool,Adam/gradients/Conv2D_3/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*
paddingSAME*'
_output_shapes
:�@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
�
3Adam/gradients/MaxPool2D_2/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_2/ReluMaxPool2D_2/MaxPool7Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
*Adam/gradients/Conv2D_2/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_2/MaxPool_grad/MaxPoolGradConv2D_2/Relu*0
_output_shapes
:����������*
T0
�
0Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
*Adam/gradients/Conv2D_2/Conv2D_grad/ShapeNShapeNMaxPool2D_1/MaxPoolConv2D_2/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
7Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_2/Conv2D_grad/ShapeNConv2D_2/W/read*Adam/gradients/Conv2D_2/Relu_grad/ReluGrad*/
_output_shapes
:���������@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
�
8Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_1/MaxPool,Adam/gradients/Conv2D_2/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_2/Relu_grad/ReluGrad*'
_output_shapes
:@�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
�
3Adam/gradients/MaxPool2D_1/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_1/ReluMaxPool2D_1/MaxPool7Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropInput*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������

@
�
*Adam/gradients/Conv2D_1/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_1/MaxPool_grad/MaxPoolGradConv2D_1/Relu*
T0*/
_output_shapes
:���������

@
�
0Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
*Adam/gradients/Conv2D_1/Conv2D_grad/ShapeNShapeNMaxPool2D/MaxPoolConv2D_1/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
7Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_1/Conv2D_grad/ShapeNConv2D_1/W/read*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*/
_output_shapes
:���������

 *
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
�
8Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D/MaxPool,Adam/gradients/Conv2D_1/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
�
1Adam/gradients/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D/ReluMaxPool2D/MaxPool7Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������22 *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
(Adam/gradients/Conv2D/Relu_grad/ReluGradReluGrad1Adam/gradients/MaxPool2D/MaxPool_grad/MaxPoolGradConv2D/Relu*
T0*/
_output_shapes
:���������22 
�
.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGradBiasAddGrad(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
�
(Adam/gradients/Conv2D/Conv2D_grad/ShapeNShapeNinput/XConv2D/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
5Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(Adam/gradients/Conv2D/Conv2D_grad/ShapeNConv2D/W/read(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*/
_output_shapes
:���������22*
	dilations

�
6Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput/X*Adam/gradients/Conv2D/Conv2D_grad/ShapeN:1(Adam/gradients/Conv2D/Relu_grad/ReluGrad*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
�
Adam/global_norm/L2LossL2Loss6Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*
T0*I
_class?
=;loc:@Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_1L2Loss.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad*
T0*A
_class7
53loc:@Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_2L2Loss8Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_3L2Loss0Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_4L2Loss8Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_5L2Loss0Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_6L2Loss8Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_7L2Loss0Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_8L2Loss8Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_9L2Loss0Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0*C
_class9
75loc:@Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGrad
�
Adam/global_norm/L2Loss_10L2Loss2Adam/gradients/FullyConnected/MatMul_grad/MatMul_1*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1*
_output_shapes
: 
�
Adam/global_norm/L2Loss_11L2Loss6Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad*
T0*I
_class?
=;loc:@Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/L2Loss_12L2Loss4Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1*
T0*G
_class=
;9loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1*
_output_shapes
: 
�
Adam/global_norm/L2Loss_13L2Loss8Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
T0*K
_classA
?=loc:@Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/global_norm/stackPackAdam/global_norm/L2LossAdam/global_norm/L2Loss_1Adam/global_norm/L2Loss_2Adam/global_norm/L2Loss_3Adam/global_norm/L2Loss_4Adam/global_norm/L2Loss_5Adam/global_norm/L2Loss_6Adam/global_norm/L2Loss_7Adam/global_norm/L2Loss_8Adam/global_norm/L2Loss_9Adam/global_norm/L2Loss_10Adam/global_norm/L2Loss_11Adam/global_norm/L2Loss_12Adam/global_norm/L2Loss_13*
T0*

axis *
N*
_output_shapes
:

Adam/global_norm/ConstConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
:*
valueB: 
�
Adam/global_norm/SumSumAdam/global_norm/stackAdam/global_norm/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
|
Adam/global_norm/Const_1Const^Adam/moving_avg^moving_avg*
valueB
 *   @*
dtype0*
_output_shapes
: 
l
Adam/global_norm/mulMulAdam/global_norm/SumAdam/global_norm/Const_1*
T0*
_output_shapes
: 
[
Adam/global_norm/global_normSqrtAdam/global_norm/mul*
T0*
_output_shapes
: 
�
"Adam/clip_by_global_norm/truediv/xConst^Adam/moving_avg^moving_avg*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
 Adam/clip_by_global_norm/truedivRealDiv"Adam/clip_by_global_norm/truediv/xAdam/global_norm/global_norm*
T0*
_output_shapes
: 
�
Adam/clip_by_global_norm/ConstConst^Adam/moving_avg^moving_avg*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
$Adam/clip_by_global_norm/truediv_1/yConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB
 *  �@
�
"Adam/clip_by_global_norm/truediv_1RealDivAdam/clip_by_global_norm/Const$Adam/clip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 
�
 Adam/clip_by_global_norm/MinimumMinimum Adam/clip_by_global_norm/truediv"Adam/clip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
�
Adam/clip_by_global_norm/mul/xConst^Adam/moving_avg^moving_avg*
valueB
 *  �@*
dtype0*
_output_shapes
: 
�
Adam/clip_by_global_norm/mulMulAdam/clip_by_global_norm/mul/x Adam/clip_by_global_norm/Minimum*
T0*
_output_shapes
: 
l
!Adam/clip_by_global_norm/IsFiniteIsFiniteAdam/global_norm/global_norm*
_output_shapes
: *
T0
�
 Adam/clip_by_global_norm/Const_1Const^Adam/moving_avg^moving_avg*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
Adam/clip_by_global_norm/SelectSelect!Adam/clip_by_global_norm/IsFiniteAdam/clip_by_global_norm/mul Adam/clip_by_global_norm/Const_1*
T0*
_output_shapes
: 
�
Adam/clip_by_global_norm/mul_1Mul6Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*
T0*I
_class?
=;loc:@Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_0IdentityAdam/clip_by_global_norm/mul_1*
T0*I
_class?
=;loc:@Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
Adam/clip_by_global_norm/mul_2Mul.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
_output_shapes
: *
T0*A
_class7
53loc:@Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_1IdentityAdam/clip_by_global_norm/mul_2*
_output_shapes
: *
T0*A
_class7
53loc:@Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad
�
Adam/clip_by_global_norm/mul_3Mul8Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_2IdentityAdam/clip_by_global_norm/mul_3*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
Adam/clip_by_global_norm/mul_4Mul0Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
_output_shapes
:@*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_3IdentityAdam/clip_by_global_norm/mul_4*
_output_shapes
:@*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad
�
Adam/clip_by_global_norm/mul_5Mul8Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*'
_output_shapes
:@�*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_4IdentityAdam/clip_by_global_norm/mul_5*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
Adam/clip_by_global_norm/mul_6Mul0Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
_output_shapes	
:�*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGrad
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_5IdentityAdam/clip_by_global_norm/mul_6*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Adam/clip_by_global_norm/mul_7Mul8Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�@
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_6IdentityAdam/clip_by_global_norm/mul_7*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�@
�
Adam/clip_by_global_norm/mul_8Mul0Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_7IdentityAdam/clip_by_global_norm/mul_8*
_output_shapes
:@*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad
�
Adam/clip_by_global_norm/mul_9Mul8Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@ 
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_8IdentityAdam/clip_by_global_norm/mul_9*&
_output_shapes
:@ *
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter
�
Adam/clip_by_global_norm/mul_10Mul0Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_9IdentityAdam/clip_by_global_norm/mul_10*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
Adam/clip_by_global_norm/mul_11Mul2Adam/gradients/FullyConnected/MatMul_grad/MatMul_1Adam/clip_by_global_norm/Select*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1*
_output_shapes
:	 �
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_10IdentityAdam/clip_by_global_norm/mul_11*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1*
_output_shapes
:	 �
�
Adam/clip_by_global_norm/mul_12Mul6Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
_output_shapes	
:�*
T0*I
_class?
=;loc:@Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_11IdentityAdam/clip_by_global_norm/mul_12*
T0*I
_class?
=;loc:@Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Adam/clip_by_global_norm/mul_13Mul4Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1Adam/clip_by_global_norm/Select*
_output_shapes
:	�*
T0*G
_class=
;9loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_12IdentityAdam/clip_by_global_norm/mul_13*
T0*G
_class=
;9loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
Adam/clip_by_global_norm/mul_14Mul8Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
_output_shapes
:*
T0*K
_classA
?=loc:@Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_13IdentityAdam/clip_by_global_norm/mul_14*
T0*K
_classA
?=loc:@Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
Adam/beta1_power/initial_valueConst*
_class
loc:@Conv2D/W*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
Adam/beta1_power
VariableV2*
_class
loc:@Conv2D/W*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
Adam/beta1_power/AssignAssignAdam/beta1_powerAdam/beta1_power/initial_value*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: *
use_locking(
q
Adam/beta1_power/readIdentityAdam/beta1_power*
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
Adam/beta2_power/initial_valueConst*
_class
loc:@Conv2D/W*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
Adam/beta2_power
VariableV2*
_class
loc:@Conv2D/W*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
Adam/beta2_power/AssignAssignAdam/beta2_powerAdam/beta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
q
Adam/beta2_power/readIdentityAdam/beta2_power*
T0*
_class
loc:@Conv2D/W*
_output_shapes
: 
�
Conv2D/W/Adam/Initializer/zerosConst*%
valueB *    *
_class
loc:@Conv2D/W*
dtype0*&
_output_shapes
: 
�
Conv2D/W/Adam
VariableV2*
shape: *
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/W*
	container 
�
Conv2D/W/Adam/AssignAssignConv2D/W/AdamConv2D/W/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
{
Conv2D/W/Adam/readIdentityConv2D/W/Adam*&
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
!Conv2D/W/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
: *%
valueB *    *
_class
loc:@Conv2D/W
�
Conv2D/W/Adam_1
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/W*
	container *
shape: 
�
Conv2D/W/Adam_1/AssignAssignConv2D/W/Adam_1!Conv2D/W/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(

Conv2D/W/Adam_1/readIdentityConv2D/W/Adam_1*
T0*
_class
loc:@Conv2D/W*&
_output_shapes
: 
�
Conv2D/b/Adam/Initializer/zerosConst*
valueB *    *
_class
loc:@Conv2D/b*
dtype0*
_output_shapes
: 
�
Conv2D/b/Adam
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D/b*
	container *
shape: 
�
Conv2D/b/Adam/AssignAssignConv2D/b/AdamConv2D/b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
o
Conv2D/b/Adam/readIdentityConv2D/b/Adam*
_output_shapes
: *
T0*
_class
loc:@Conv2D/b
�
!Conv2D/b/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@Conv2D/b*
dtype0*
_output_shapes
: 
�
Conv2D/b/Adam_1
VariableV2*
shared_name *
_class
loc:@Conv2D/b*
	container *
shape: *
dtype0*
_output_shapes
: 
�
Conv2D/b/Adam_1/AssignAssignConv2D/b/Adam_1!Conv2D/b/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
s
Conv2D/b/Adam_1/readIdentityConv2D/b/Adam_1*
T0*
_class
loc:@Conv2D/b*
_output_shapes
: 
�
1Conv2D_1/W/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *
_class
loc:@Conv2D_1/W*
dtype0*
_output_shapes
:
�
'Conv2D_1/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Conv2D_1/W*
dtype0*
_output_shapes
: 
�
!Conv2D_1/W/Adam/Initializer/zerosFill1Conv2D_1/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_1/W/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Conv2D_1/W*&
_output_shapes
: @
�
Conv2D_1/W/Adam
VariableV2*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name *
_class
loc:@Conv2D_1/W
�
Conv2D_1/W/Adam/AssignAssignConv2D_1/W/Adam!Conv2D_1/W/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
Conv2D_1/W/Adam/readIdentityConv2D_1/W/Adam*
T0*
_class
loc:@Conv2D_1/W*&
_output_shapes
: @
�
3Conv2D_1/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"          @   *
_class
loc:@Conv2D_1/W
�
)Conv2D_1/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Conv2D_1/W*
dtype0*
_output_shapes
: 
�
#Conv2D_1/W/Adam_1/Initializer/zerosFill3Conv2D_1/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_1/W/Adam_1/Initializer/zeros/Const*&
_output_shapes
: @*
T0*

index_type0*
_class
loc:@Conv2D_1/W
�
Conv2D_1/W/Adam_1
VariableV2*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name *
_class
loc:@Conv2D_1/W
�
Conv2D_1/W/Adam_1/AssignAssignConv2D_1/W/Adam_1#Conv2D_1/W/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
Conv2D_1/W/Adam_1/readIdentityConv2D_1/W/Adam_1*
T0*
_class
loc:@Conv2D_1/W*&
_output_shapes
: @
�
!Conv2D_1/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *
_class
loc:@Conv2D_1/b
�
Conv2D_1/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_1/b*
	container *
shape:@
�
Conv2D_1/b/Adam/AssignAssignConv2D_1/b/Adam!Conv2D_1/b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
u
Conv2D_1/b/Adam/readIdentityConv2D_1/b/Adam*
T0*
_class
loc:@Conv2D_1/b*
_output_shapes
:@
�
#Conv2D_1/b/Adam_1/Initializer/zerosConst*
valueB@*    *
_class
loc:@Conv2D_1/b*
dtype0*
_output_shapes
:@
�
Conv2D_1/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_1/b*
	container *
shape:@
�
Conv2D_1/b/Adam_1/AssignAssignConv2D_1/b/Adam_1#Conv2D_1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
y
Conv2D_1/b/Adam_1/readIdentityConv2D_1/b/Adam_1*
_output_shapes
:@*
T0*
_class
loc:@Conv2D_1/b
�
1Conv2D_2/W/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   �   *
_class
loc:@Conv2D_2/W
�
'Conv2D_2/W/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Conv2D_2/W
�
!Conv2D_2/W/Adam/Initializer/zerosFill1Conv2D_2/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_2/W/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�
Conv2D_2/W/Adam
VariableV2*
dtype0*'
_output_shapes
:@�*
shared_name *
_class
loc:@Conv2D_2/W*
	container *
shape:@�
�
Conv2D_2/W/Adam/AssignAssignConv2D_2/W/Adam!Conv2D_2/W/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
Conv2D_2/W/Adam/readIdentityConv2D_2/W/Adam*'
_output_shapes
:@�*
T0*
_class
loc:@Conv2D_2/W
�
3Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   �   *
_class
loc:@Conv2D_2/W*
dtype0*
_output_shapes
:
�
)Conv2D_2/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Conv2D_2/W*
dtype0*
_output_shapes
: 
�
#Conv2D_2/W/Adam_1/Initializer/zerosFill3Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_2/W/Adam_1/Initializer/zeros/Const*'
_output_shapes
:@�*
T0*

index_type0*
_class
loc:@Conv2D_2/W
�
Conv2D_2/W/Adam_1
VariableV2*
shared_name *
_class
loc:@Conv2D_2/W*
	container *
shape:@�*
dtype0*'
_output_shapes
:@�
�
Conv2D_2/W/Adam_1/AssignAssignConv2D_2/W/Adam_1#Conv2D_2/W/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
Conv2D_2/W/Adam_1/readIdentityConv2D_2/W/Adam_1*
T0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�
!Conv2D_2/b/Adam/Initializer/zerosConst*
valueB�*    *
_class
loc:@Conv2D_2/b*
dtype0*
_output_shapes	
:�
�
Conv2D_2/b/Adam
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@Conv2D_2/b*
	container 
�
Conv2D_2/b/Adam/AssignAssignConv2D_2/b/Adam!Conv2D_2/b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
v
Conv2D_2/b/Adam/readIdentityConv2D_2/b/Adam*
_output_shapes	
:�*
T0*
_class
loc:@Conv2D_2/b
�
#Conv2D_2/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Conv2D_2/b
�
Conv2D_2/b/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@Conv2D_2/b*
	container *
shape:�
�
Conv2D_2/b/Adam_1/AssignAssignConv2D_2/b/Adam_1#Conv2D_2/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
z
Conv2D_2/b/Adam_1/readIdentityConv2D_2/b/Adam_1*
T0*
_class
loc:@Conv2D_2/b*
_output_shapes	
:�
�
1Conv2D_3/W/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      �   @   *
_class
loc:@Conv2D_3/W*
dtype0*
_output_shapes
:
�
'Conv2D_3/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Conv2D_3/W*
dtype0*
_output_shapes
: 
�
!Conv2D_3/W/Adam/Initializer/zerosFill1Conv2D_3/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_3/W/Adam/Initializer/zeros/Const*'
_output_shapes
:�@*
T0*

index_type0*
_class
loc:@Conv2D_3/W
�
Conv2D_3/W/Adam
VariableV2*
shape:�@*
dtype0*'
_output_shapes
:�@*
shared_name *
_class
loc:@Conv2D_3/W*
	container 
�
Conv2D_3/W/Adam/AssignAssignConv2D_3/W/Adam!Conv2D_3/W/Adam/Initializer/zeros*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
Conv2D_3/W/Adam/readIdentityConv2D_3/W/Adam*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
3Conv2D_3/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      �   @   *
_class
loc:@Conv2D_3/W
�
)Conv2D_3/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Conv2D_3/W*
dtype0*
_output_shapes
: 
�
#Conv2D_3/W/Adam_1/Initializer/zerosFill3Conv2D_3/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_3/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
Conv2D_3/W/Adam_1
VariableV2*
_class
loc:@Conv2D_3/W*
	container *
shape:�@*
dtype0*'
_output_shapes
:�@*
shared_name 
�
Conv2D_3/W/Adam_1/AssignAssignConv2D_3/W/Adam_1#Conv2D_3/W/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
Conv2D_3/W/Adam_1/readIdentityConv2D_3/W/Adam_1*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
!Conv2D_3/b/Adam/Initializer/zerosConst*
valueB@*    *
_class
loc:@Conv2D_3/b*
dtype0*
_output_shapes
:@
�
Conv2D_3/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_3/b*
	container *
shape:@
�
Conv2D_3/b/Adam/AssignAssignConv2D_3/b/Adam!Conv2D_3/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
u
Conv2D_3/b/Adam/readIdentityConv2D_3/b/Adam*
T0*
_class
loc:@Conv2D_3/b*
_output_shapes
:@
�
#Conv2D_3/b/Adam_1/Initializer/zerosConst*
valueB@*    *
_class
loc:@Conv2D_3/b*
dtype0*
_output_shapes
:@
�
Conv2D_3/b/Adam_1
VariableV2*
shared_name *
_class
loc:@Conv2D_3/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
Conv2D_3/b/Adam_1/AssignAssignConv2D_3/b/Adam_1#Conv2D_3/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
y
Conv2D_3/b/Adam_1/readIdentityConv2D_3/b/Adam_1*
_output_shapes
:@*
T0*
_class
loc:@Conv2D_3/b
�
1Conv2D_4/W/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @       *
_class
loc:@Conv2D_4/W
�
'Conv2D_4/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Conv2D_4/W*
dtype0*
_output_shapes
: 
�
!Conv2D_4/W/Adam/Initializer/zerosFill1Conv2D_4/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_4/W/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�
Conv2D_4/W/Adam
VariableV2*
shared_name *
_class
loc:@Conv2D_4/W*
	container *
shape:@ *
dtype0*&
_output_shapes
:@ 
�
Conv2D_4/W/Adam/AssignAssignConv2D_4/W/Adam!Conv2D_4/W/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
Conv2D_4/W/Adam/readIdentityConv2D_4/W/Adam*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�
3Conv2D_4/W/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      @       *
_class
loc:@Conv2D_4/W*
dtype0*
_output_shapes
:
�
)Conv2D_4/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Conv2D_4/W*
dtype0*
_output_shapes
: 
�
#Conv2D_4/W/Adam_1/Initializer/zerosFill3Conv2D_4/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_4/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�
Conv2D_4/W/Adam_1
VariableV2*
shared_name *
_class
loc:@Conv2D_4/W*
	container *
shape:@ *
dtype0*&
_output_shapes
:@ 
�
Conv2D_4/W/Adam_1/AssignAssignConv2D_4/W/Adam_1#Conv2D_4/W/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
Conv2D_4/W/Adam_1/readIdentityConv2D_4/W/Adam_1*&
_output_shapes
:@ *
T0*
_class
loc:@Conv2D_4/W
�
!Conv2D_4/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *
_class
loc:@Conv2D_4/b
�
Conv2D_4/b/Adam
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D_4/b*
	container *
shape: 
�
Conv2D_4/b/Adam/AssignAssignConv2D_4/b/Adam!Conv2D_4/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D_4/b
u
Conv2D_4/b/Adam/readIdentityConv2D_4/b/Adam*
T0*
_class
loc:@Conv2D_4/b*
_output_shapes
: 
�
#Conv2D_4/b/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@Conv2D_4/b*
dtype0*
_output_shapes
: 
�
Conv2D_4/b/Adam_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D_4/b*
	container *
shape: 
�
Conv2D_4/b/Adam_1/AssignAssignConv2D_4/b/Adam_1#Conv2D_4/b/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
y
Conv2D_4/b/Adam_1/readIdentityConv2D_4/b/Adam_1*
T0*
_class
loc:@Conv2D_4/b*
_output_shapes
: 
�
7FullyConnected/W/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"       *#
_class
loc:@FullyConnected/W*
dtype0*
_output_shapes
:
�
-FullyConnected/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@FullyConnected/W*
dtype0*
_output_shapes
: 
�
'FullyConnected/W/Adam/Initializer/zerosFill7FullyConnected/W/Adam/Initializer/zeros/shape_as_tensor-FullyConnected/W/Adam/Initializer/zeros/Const*
_output_shapes
:	 �*
T0*

index_type0*#
_class
loc:@FullyConnected/W
�
FullyConnected/W/Adam
VariableV2*
shared_name *#
_class
loc:@FullyConnected/W*
	container *
shape:	 �*
dtype0*
_output_shapes
:	 �
�
FullyConnected/W/Adam/AssignAssignFullyConnected/W/Adam'FullyConnected/W/Adam/Initializer/zeros*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
�
FullyConnected/W/Adam/readIdentityFullyConnected/W/Adam*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
9FullyConnected/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"       *#
_class
loc:@FullyConnected/W
�
/FullyConnected/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@FullyConnected/W*
dtype0*
_output_shapes
: 
�
)FullyConnected/W/Adam_1/Initializer/zerosFill9FullyConnected/W/Adam_1/Initializer/zeros/shape_as_tensor/FullyConnected/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
FullyConnected/W/Adam_1
VariableV2*
dtype0*
_output_shapes
:	 �*
shared_name *#
_class
loc:@FullyConnected/W*
	container *
shape:	 �
�
FullyConnected/W/Adam_1/AssignAssignFullyConnected/W/Adam_1)FullyConnected/W/Adam_1/Initializer/zeros*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
�
FullyConnected/W/Adam_1/readIdentityFullyConnected/W/Adam_1*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
7FullyConnected/b/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:�*#
_class
loc:@FullyConnected/b
�
-FullyConnected/b/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *#
_class
loc:@FullyConnected/b
�
'FullyConnected/b/Adam/Initializer/zerosFill7FullyConnected/b/Adam/Initializer/zeros/shape_as_tensor-FullyConnected/b/Adam/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
�
FullyConnected/b/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *#
_class
loc:@FullyConnected/b*
	container *
shape:�
�
FullyConnected/b/Adam/AssignAssignFullyConnected/b/Adam'FullyConnected/b/Adam/Initializer/zeros*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
FullyConnected/b/Adam/readIdentityFullyConnected/b/Adam*
T0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
�
9FullyConnected/b/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:�*#
_class
loc:@FullyConnected/b*
dtype0*
_output_shapes
:
�
/FullyConnected/b/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *#
_class
loc:@FullyConnected/b
�
)FullyConnected/b/Adam_1/Initializer/zerosFill9FullyConnected/b/Adam_1/Initializer/zeros/shape_as_tensor/FullyConnected/b/Adam_1/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
�
FullyConnected/b/Adam_1
VariableV2*
shared_name *#
_class
loc:@FullyConnected/b*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
FullyConnected/b/Adam_1/AssignAssignFullyConnected/b/Adam_1)FullyConnected/b/Adam_1/Initializer/zeros*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
FullyConnected/b/Adam_1/readIdentityFullyConnected/b/Adam_1*
_output_shapes	
:�*
T0*#
_class
loc:@FullyConnected/b
�
9FullyConnected_1/W/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *%
_class
loc:@FullyConnected_1/W*
dtype0*
_output_shapes
:
�
/FullyConnected_1/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@FullyConnected_1/W*
dtype0*
_output_shapes
: 
�
)FullyConnected_1/W/Adam/Initializer/zerosFill9FullyConnected_1/W/Adam/Initializer/zeros/shape_as_tensor/FullyConnected_1/W/Adam/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
�
FullyConnected_1/W/Adam
VariableV2*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *%
_class
loc:@FullyConnected_1/W
�
FullyConnected_1/W/Adam/AssignAssignFullyConnected_1/W/Adam)FullyConnected_1/W/Adam/Initializer/zeros*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
FullyConnected_1/W/Adam/readIdentityFullyConnected_1/W/Adam*
_output_shapes
:	�*
T0*%
_class
loc:@FullyConnected_1/W
�
;FullyConnected_1/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      *%
_class
loc:@FullyConnected_1/W
�
1FullyConnected_1/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@FullyConnected_1/W*
dtype0*
_output_shapes
: 
�
+FullyConnected_1/W/Adam_1/Initializer/zerosFill;FullyConnected_1/W/Adam_1/Initializer/zeros/shape_as_tensor1FullyConnected_1/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
�
FullyConnected_1/W/Adam_1
VariableV2*
shared_name *%
_class
loc:@FullyConnected_1/W*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
 FullyConnected_1/W/Adam_1/AssignAssignFullyConnected_1/W/Adam_1+FullyConnected_1/W/Adam_1/Initializer/zeros*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
FullyConnected_1/W/Adam_1/readIdentityFullyConnected_1/W/Adam_1*
T0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
�
)FullyConnected_1/b/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@FullyConnected_1/b*
dtype0*
_output_shapes
:
�
FullyConnected_1/b/Adam
VariableV2*
shared_name *%
_class
loc:@FullyConnected_1/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
FullyConnected_1/b/Adam/AssignAssignFullyConnected_1/b/Adam)FullyConnected_1/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
�
FullyConnected_1/b/Adam/readIdentityFullyConnected_1/b/Adam*
T0*%
_class
loc:@FullyConnected_1/b*
_output_shapes
:
�
+FullyConnected_1/b/Adam_1/Initializer/zerosConst*
valueB*    *%
_class
loc:@FullyConnected_1/b*
dtype0*
_output_shapes
:
�
FullyConnected_1/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@FullyConnected_1/b*
	container *
shape:
�
 FullyConnected_1/b/Adam_1/AssignAssignFullyConnected_1/b/Adam_1+FullyConnected_1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
FullyConnected_1/b/Adam_1/readIdentityFullyConnected_1/b/Adam_1*
T0*%
_class
loc:@FullyConnected_1/b*
_output_shapes
:
g
"Adam/apply_grad_op_0/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
_
Adam/apply_grad_op_0/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
_
Adam/apply_grad_op_0/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
a
Adam/apply_grad_op_0/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
.Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam	ApplyAdamConv2D/WConv2D/W/AdamConv2D/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_0*
T0*
_class
loc:@Conv2D/W*
use_nesterov( *&
_output_shapes
: *
use_locking( 
�
.Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam	ApplyAdamConv2D/bConv2D/b/AdamConv2D/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_1*
use_locking( *
T0*
_class
loc:@Conv2D/b*
use_nesterov( *
_output_shapes
: 
�
0Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam	ApplyAdam
Conv2D_1/WConv2D_1/W/AdamConv2D_1/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_2*
use_locking( *
T0*
_class
loc:@Conv2D_1/W*
use_nesterov( *&
_output_shapes
: @
�
0Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam	ApplyAdam
Conv2D_1/bConv2D_1/b/AdamConv2D_1/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_3*
use_locking( *
T0*
_class
loc:@Conv2D_1/b*
use_nesterov( *
_output_shapes
:@
�
0Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam	ApplyAdam
Conv2D_2/WConv2D_2/W/AdamConv2D_2/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_4*
use_locking( *
T0*
_class
loc:@Conv2D_2/W*
use_nesterov( *'
_output_shapes
:@�
�
0Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam	ApplyAdam
Conv2D_2/bConv2D_2/b/AdamConv2D_2/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_5*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*
_class
loc:@Conv2D_2/b
�
0Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam	ApplyAdam
Conv2D_3/WConv2D_3/W/AdamConv2D_3/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_6*
use_locking( *
T0*
_class
loc:@Conv2D_3/W*
use_nesterov( *'
_output_shapes
:�@
�
0Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam	ApplyAdam
Conv2D_3/bConv2D_3/b/AdamConv2D_3/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_7*
T0*
_class
loc:@Conv2D_3/b*
use_nesterov( *
_output_shapes
:@*
use_locking( 
�
0Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam	ApplyAdam
Conv2D_4/WConv2D_4/W/AdamConv2D_4/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_8*
use_nesterov( *&
_output_shapes
:@ *
use_locking( *
T0*
_class
loc:@Conv2D_4/W
�
0Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam	ApplyAdam
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_9*
use_nesterov( *
_output_shapes
: *
use_locking( *
T0*
_class
loc:@Conv2D_4/b
�
6Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam	ApplyAdamFullyConnected/WFullyConnected/W/AdamFullyConnected/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_10*
use_nesterov( *
_output_shapes
:	 �*
use_locking( *
T0*#
_class
loc:@FullyConnected/W
�
6Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam	ApplyAdamFullyConnected/bFullyConnected/b/AdamFullyConnected/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_11*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*#
_class
loc:@FullyConnected/b
�
8Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam	ApplyAdamFullyConnected_1/WFullyConnected_1/W/AdamFullyConnected_1/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_12*
T0*%
_class
loc:@FullyConnected_1/W*
use_nesterov( *
_output_shapes
:	�*
use_locking( 
�
8Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam	ApplyAdamFullyConnected_1/bFullyConnected_1/b/AdamFullyConnected_1/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_13*
use_locking( *
T0*%
_class
loc:@FullyConnected_1/b*
use_nesterov( *
_output_shapes
:
�
Adam/apply_grad_op_0/mulMulAdam/beta1_power/readAdam/apply_grad_op_0/beta1/^Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam/^Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam*
T0*
_class
loc:@Conv2D/W*
_output_shapes
: 
�
Adam/apply_grad_op_0/AssignAssignAdam/beta1_powerAdam/apply_grad_op_0/mul*
use_locking( *
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
�
Adam/apply_grad_op_0/mul_1MulAdam/beta2_power/readAdam/apply_grad_op_0/beta2/^Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam/^Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
Adam/apply_grad_op_0/Assign_1AssignAdam/beta2_powerAdam/apply_grad_op_0/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@Conv2D/W
�
Adam/apply_grad_op_0/updateNoOp^Adam/apply_grad_op_0/Assign^Adam/apply_grad_op_0/Assign_1/^Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam/^Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam
�
Adam/apply_grad_op_0/valueConst^Adam/apply_grad_op_0/update* 
_class
loc:@Training_step*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Adam/apply_grad_op_0	AssignAddTraining_stepAdam/apply_grad_op_0/value*
use_locking( *
T0* 
_class
loc:@Training_step*
_output_shapes
: 
]
Adam/Merge/MergeSummaryMergeSummaryLossAdam/Loss/raw*
N*
_output_shapes
: 
.
Adam/train_op_0NoOp^Adam/apply_grad_op_0
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
save/SaveV2/shape_and_slicesConst*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesAccuracy/Mean/moving_avgAdam/beta1_powerAdam/beta2_powerConv2D/WConv2D/W/AdamConv2D/W/Adam_1Conv2D/bConv2D/b/AdamConv2D/b/Adam_1
Conv2D_1/WConv2D_1/W/AdamConv2D_1/W/Adam_1
Conv2D_1/bConv2D_1/b/AdamConv2D_1/b/Adam_1
Conv2D_2/WConv2D_2/W/AdamConv2D_2/W/Adam_1
Conv2D_2/bConv2D_2/b/AdamConv2D_2/b/Adam_1
Conv2D_3/WConv2D_3/W/AdamConv2D_3/W/Adam_1
Conv2D_3/bConv2D_3/b/AdamConv2D_3/b/Adam_1
Conv2D_4/WConv2D_4/W/AdamConv2D_4/W/Adam_1
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Crossentropy/Mean/moving_avgFullyConnected/WFullyConnected/W/AdamFullyConnected/W/Adam_1FullyConnected/bFullyConnected/b/AdamFullyConnected/b/Adam_1FullyConnected_1/WFullyConnected_1/W/AdamFullyConnected_1/W/Adam_1FullyConnected_1/bFullyConnected_1/b/AdamFullyConnected_1/b/Adam_1Global_StepTraining_stepis_trainingval_accval_loss*A
dtypes7
523

}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:3*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523

�
save/AssignAssignAccuracy/Mean/moving_avgsave/RestoreV2*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_1AssignAdam/beta1_powersave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save/Assign_2AssignAdam/beta2_powersave/RestoreV2:2*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_3AssignConv2D/Wsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
�
save/Assign_4AssignConv2D/W/Adamsave/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
�
save/Assign_5AssignConv2D/W/Adam_1save/RestoreV2:5*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save/Assign_6AssignConv2D/bsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save/Assign_7AssignConv2D/b/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save/Assign_8AssignConv2D/b/Adam_1save/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/b
�
save/Assign_9Assign
Conv2D_1/Wsave/RestoreV2:9*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save/Assign_10AssignConv2D_1/W/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save/Assign_11AssignConv2D_1/W/Adam_1save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save/Assign_12Assign
Conv2D_1/bsave/RestoreV2:12*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_13AssignConv2D_1/b/Adamsave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
�
save/Assign_14AssignConv2D_1/b/Adam_1save/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
�
save/Assign_15Assign
Conv2D_2/Wsave/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
save/Assign_16AssignConv2D_2/W/Adamsave/RestoreV2:16*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save/Assign_17AssignConv2D_2/W/Adam_1save/RestoreV2:17*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save/Assign_18Assign
Conv2D_2/bsave/RestoreV2:18*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_19AssignConv2D_2/b/Adamsave/RestoreV2:19*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_20AssignConv2D_2/b/Adam_1save/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save/Assign_21Assign
Conv2D_3/Wsave/RestoreV2:21*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
save/Assign_22AssignConv2D_3/W/Adamsave/RestoreV2:22*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
save/Assign_23AssignConv2D_3/W/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save/Assign_24Assign
Conv2D_3/bsave/RestoreV2:24*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_25AssignConv2D_3/b/Adamsave/RestoreV2:25*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save/Assign_26AssignConv2D_3/b/Adam_1save/RestoreV2:26*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
�
save/Assign_27Assign
Conv2D_4/Wsave/RestoreV2:27*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
save/Assign_28AssignConv2D_4/W/Adamsave/RestoreV2:28*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
save/Assign_29AssignConv2D_4/W/Adam_1save/RestoreV2:29*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save/Assign_30Assign
Conv2D_4/bsave/RestoreV2:30*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D_4/b
�
save/Assign_31AssignConv2D_4/b/Adamsave/RestoreV2:31*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D_4/b
�
save/Assign_32AssignConv2D_4/b/Adam_1save/RestoreV2:32*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_33AssignCrossentropy/Mean/moving_avgsave/RestoreV2:33*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_34AssignFullyConnected/Wsave/RestoreV2:34*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save/Assign_35AssignFullyConnected/W/Adamsave/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save/Assign_36AssignFullyConnected/W/Adam_1save/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save/Assign_37AssignFullyConnected/bsave/RestoreV2:37*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
save/Assign_38AssignFullyConnected/b/Adamsave/RestoreV2:38*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_39AssignFullyConnected/b/Adam_1save/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save/Assign_40AssignFullyConnected_1/Wsave/RestoreV2:40*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W
�
save/Assign_41AssignFullyConnected_1/W/Adamsave/RestoreV2:41*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
save/Assign_42AssignFullyConnected_1/W/Adam_1save/RestoreV2:42*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W
�
save/Assign_43AssignFullyConnected_1/bsave/RestoreV2:43*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_44AssignFullyConnected_1/b/Adamsave/RestoreV2:44*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
�
save/Assign_45AssignFullyConnected_1/b/Adam_1save/RestoreV2:45*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
�
save/Assign_46AssignGlobal_Stepsave/RestoreV2:46*
use_locking(*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: 
�
save/Assign_47AssignTraining_stepsave/RestoreV2:47*
T0* 
_class
loc:@Training_step*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_48Assignis_trainingsave/RestoreV2:48*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: 
�
save/Assign_49Assignval_accsave/RestoreV2:49*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@val_acc
�
save/Assign_50Assignval_losssave/RestoreV2:50*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@val_loss
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_1/SaveV2/tensor_namesConst*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
save_1/SaveV2/shape_and_slicesConst*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesAccuracy/Mean/moving_avgAdam/beta1_powerAdam/beta2_powerConv2D/WConv2D/W/AdamConv2D/W/Adam_1Conv2D/bConv2D/b/AdamConv2D/b/Adam_1
Conv2D_1/WConv2D_1/W/AdamConv2D_1/W/Adam_1
Conv2D_1/bConv2D_1/b/AdamConv2D_1/b/Adam_1
Conv2D_2/WConv2D_2/W/AdamConv2D_2/W/Adam_1
Conv2D_2/bConv2D_2/b/AdamConv2D_2/b/Adam_1
Conv2D_3/WConv2D_3/W/AdamConv2D_3/W/Adam_1
Conv2D_3/bConv2D_3/b/AdamConv2D_3/b/Adam_1
Conv2D_4/WConv2D_4/W/AdamConv2D_4/W/Adam_1
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Crossentropy/Mean/moving_avgFullyConnected/WFullyConnected/W/AdamFullyConnected/W/Adam_1FullyConnected/bFullyConnected/b/AdamFullyConnected/b/Adam_1FullyConnected_1/WFullyConnected_1/W/AdamFullyConnected_1/W/Adam_1FullyConnected_1/bFullyConnected_1/b/AdamFullyConnected_1/b/Adam_1Global_StepTraining_stepis_trainingval_accval_loss*A
dtypes7
523

�
save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save_1/Const
�
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:3*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss
�
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523

�
save_1/AssignAssignAccuracy/Mean/moving_avgsave_1/RestoreV2*
use_locking(*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_1AssignAdam/beta1_powersave_1/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_2AssignAdam/beta2_powersave_1/RestoreV2:2*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_3AssignConv2D/Wsave_1/RestoreV2:3*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save_1/Assign_4AssignConv2D/W/Adamsave_1/RestoreV2:4*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save_1/Assign_5AssignConv2D/W/Adam_1save_1/RestoreV2:5*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save_1/Assign_6AssignConv2D/bsave_1/RestoreV2:6*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_7AssignConv2D/b/Adamsave_1/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_8AssignConv2D/b/Adam_1save_1/RestoreV2:8*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_9Assign
Conv2D_1/Wsave_1/RestoreV2:9*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
�
save_1/Assign_10AssignConv2D_1/W/Adamsave_1/RestoreV2:10*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save_1/Assign_11AssignConv2D_1/W/Adam_1save_1/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save_1/Assign_12Assign
Conv2D_1/bsave_1/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save_1/Assign_13AssignConv2D_1/b/Adamsave_1/RestoreV2:13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save_1/Assign_14AssignConv2D_1/b/Adam_1save_1/RestoreV2:14*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_15Assign
Conv2D_2/Wsave_1/RestoreV2:15*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save_1/Assign_16AssignConv2D_2/W/Adamsave_1/RestoreV2:16*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
save_1/Assign_17AssignConv2D_2/W/Adam_1save_1/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
save_1/Assign_18Assign
Conv2D_2/bsave_1/RestoreV2:18*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
�
save_1/Assign_19AssignConv2D_2/b/Adamsave_1/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_20AssignConv2D_2/b/Adam_1save_1/RestoreV2:20*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
�
save_1/Assign_21Assign
Conv2D_3/Wsave_1/RestoreV2:21*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save_1/Assign_22AssignConv2D_3/W/Adamsave_1/RestoreV2:22*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save_1/Assign_23AssignConv2D_3/W/Adam_1save_1/RestoreV2:23*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save_1/Assign_24Assign
Conv2D_3/bsave_1/RestoreV2:24*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_25AssignConv2D_3/b/Adamsave_1/RestoreV2:25*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_26AssignConv2D_3/b/Adam_1save_1/RestoreV2:26*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_27Assign
Conv2D_4/Wsave_1/RestoreV2:27*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save_1/Assign_28AssignConv2D_4/W/Adamsave_1/RestoreV2:28*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save_1/Assign_29AssignConv2D_4/W/Adam_1save_1/RestoreV2:29*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save_1/Assign_30Assign
Conv2D_4/bsave_1/RestoreV2:30*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_31AssignConv2D_4/b/Adamsave_1/RestoreV2:31*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_32AssignConv2D_4/b/Adam_1save_1/RestoreV2:32*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D_4/b
�
save_1/Assign_33AssignCrossentropy/Mean/moving_avgsave_1/RestoreV2:33*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_34AssignFullyConnected/Wsave_1/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save_1/Assign_35AssignFullyConnected/W/Adamsave_1/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save_1/Assign_36AssignFullyConnected/W/Adam_1save_1/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save_1/Assign_37AssignFullyConnected/bsave_1/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_38AssignFullyConnected/b/Adamsave_1/RestoreV2:38*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_39AssignFullyConnected/b/Adam_1save_1/RestoreV2:39*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
save_1/Assign_40AssignFullyConnected_1/Wsave_1/RestoreV2:40*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
save_1/Assign_41AssignFullyConnected_1/W/Adamsave_1/RestoreV2:41*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_1/Assign_42AssignFullyConnected_1/W/Adam_1save_1/RestoreV2:42*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W
�
save_1/Assign_43AssignFullyConnected_1/bsave_1/RestoreV2:43*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_44AssignFullyConnected_1/b/Adamsave_1/RestoreV2:44*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_45AssignFullyConnected_1/b/Adam_1save_1/RestoreV2:45*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
�
save_1/Assign_46AssignGlobal_Stepsave_1/RestoreV2:46*
use_locking(*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_47AssignTraining_stepsave_1/RestoreV2:47*
use_locking(*
T0* 
_class
loc:@Training_step*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_48Assignis_trainingsave_1/RestoreV2:48*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_49Assignval_accsave_1/RestoreV2:49*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_50Assignval_losssave_1/RestoreV2:50*
T0*
_class
loc:@val_loss*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
[
save_2/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_2/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*�
value�B�BConv2D/WBConv2D/bB
Conv2D_1/WB
Conv2D_1/bB
Conv2D_2/WB
Conv2D_2/bB
Conv2D_3/WB
Conv2D_3/bB
Conv2D_4/WB
Conv2D_4/bBFullyConnected/WBFullyConnected/bBFullyConnected_1/WBFullyConnected_1/b
�
save_2/SaveV2/shape_and_slicesConst*/
value&B$B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save_2/SaveV2SaveV2save_2/Constsave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesConv2D/WConv2D/b
Conv2D_1/W
Conv2D_1/b
Conv2D_2/W
Conv2D_2/b
Conv2D_3/W
Conv2D_3/b
Conv2D_4/W
Conv2D_4/bFullyConnected/WFullyConnected/bFullyConnected_1/WFullyConnected_1/b*
dtypes
2
�
save_2/control_dependencyIdentitysave_2/Const^save_2/SaveV2*
T0*
_class
loc:@save_2/Const*
_output_shapes
: 
�
save_2/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�BConv2D/WBConv2D/bB
Conv2D_1/WB
Conv2D_1/bB
Conv2D_2/WB
Conv2D_2/bB
Conv2D_3/WB
Conv2D_3/bB
Conv2D_4/WB
Conv2D_4/bBFullyConnected/WBFullyConnected/bBFullyConnected_1/WBFullyConnected_1/b*
dtype0*
_output_shapes
:
�
!save_2/RestoreV2/shape_and_slicesConst"/device:CPU:0*/
value&B$B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2
�
save_2/AssignAssignConv2D/Wsave_2/RestoreV2*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save_2/Assign_1AssignConv2D/bsave_2/RestoreV2:1*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_2/Assign_2Assign
Conv2D_1/Wsave_2/RestoreV2:2*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save_2/Assign_3Assign
Conv2D_1/bsave_2/RestoreV2:3*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_2/Assign_4Assign
Conv2D_2/Wsave_2/RestoreV2:4*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save_2/Assign_5Assign
Conv2D_2/bsave_2/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save_2/Assign_6Assign
Conv2D_3/Wsave_2/RestoreV2:6*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
save_2/Assign_7Assign
Conv2D_3/bsave_2/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_8Assign
Conv2D_4/Wsave_2/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save_2/Assign_9Assign
Conv2D_4/bsave_2/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_10AssignFullyConnected/Wsave_2/RestoreV2:10*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
�
save_2/Assign_11AssignFullyConnected/bsave_2/RestoreV2:11*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_2/Assign_12AssignFullyConnected_1/Wsave_2/RestoreV2:12*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_2/Assign_13AssignFullyConnected_1/bsave_2/RestoreV2:13*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save_2/restore_allNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_2^save_2/Assign_3^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
�

initNoOp ^Accuracy/Mean/moving_avg/Assign^Adam/beta1_power/Assign^Adam/beta2_power/Assign^Conv2D/W/Adam/Assign^Conv2D/W/Adam_1/Assign^Conv2D/W/Assign^Conv2D/b/Adam/Assign^Conv2D/b/Adam_1/Assign^Conv2D/b/Assign^Conv2D_1/W/Adam/Assign^Conv2D_1/W/Adam_1/Assign^Conv2D_1/W/Assign^Conv2D_1/b/Adam/Assign^Conv2D_1/b/Adam_1/Assign^Conv2D_1/b/Assign^Conv2D_2/W/Adam/Assign^Conv2D_2/W/Adam_1/Assign^Conv2D_2/W/Assign^Conv2D_2/b/Adam/Assign^Conv2D_2/b/Adam_1/Assign^Conv2D_2/b/Assign^Conv2D_3/W/Adam/Assign^Conv2D_3/W/Adam_1/Assign^Conv2D_3/W/Assign^Conv2D_3/b/Adam/Assign^Conv2D_3/b/Adam_1/Assign^Conv2D_3/b/Assign^Conv2D_4/W/Adam/Assign^Conv2D_4/W/Adam_1/Assign^Conv2D_4/W/Assign^Conv2D_4/b/Adam/Assign^Conv2D_4/b/Adam_1/Assign^Conv2D_4/b/Assign$^Crossentropy/Mean/moving_avg/Assign^FullyConnected/W/Adam/Assign^FullyConnected/W/Adam_1/Assign^FullyConnected/W/Assign^FullyConnected/b/Adam/Assign^FullyConnected/b/Adam_1/Assign^FullyConnected/b/Assign^FullyConnected_1/W/Adam/Assign!^FullyConnected_1/W/Adam_1/Assign^FullyConnected_1/W/Assign^FullyConnected_1/b/Adam/Assign!^FullyConnected_1/b/Adam_1/Assign^FullyConnected_1/b/Assign^Global_Step/Assign^Training_step/Assign^is_training/Assign^val_acc/Assign^val_loss/Assign

init_1NoOp
"

group_depsNoOp^init^init_1
#
init_2NoOp^is_training/Assign
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_3/SaveV2/tensor_namesConst*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
save_3/SaveV2/shape_and_slicesConst*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save_3/SaveV2SaveV2save_3/Constsave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesAccuracy/Mean/moving_avgAdam/beta1_powerAdam/beta2_powerConv2D/WConv2D/W/AdamConv2D/W/Adam_1Conv2D/bConv2D/b/AdamConv2D/b/Adam_1
Conv2D_1/WConv2D_1/W/AdamConv2D_1/W/Adam_1
Conv2D_1/bConv2D_1/b/AdamConv2D_1/b/Adam_1
Conv2D_2/WConv2D_2/W/AdamConv2D_2/W/Adam_1
Conv2D_2/bConv2D_2/b/AdamConv2D_2/b/Adam_1
Conv2D_3/WConv2D_3/W/AdamConv2D_3/W/Adam_1
Conv2D_3/bConv2D_3/b/AdamConv2D_3/b/Adam_1
Conv2D_4/WConv2D_4/W/AdamConv2D_4/W/Adam_1
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Crossentropy/Mean/moving_avgFullyConnected/WFullyConnected/W/AdamFullyConnected/W/Adam_1FullyConnected/bFullyConnected/b/AdamFullyConnected/b/Adam_1FullyConnected_1/WFullyConnected_1/W/AdamFullyConnected_1/W/Adam_1FullyConnected_1/bFullyConnected_1/b/AdamFullyConnected_1/b/Adam_1Global_StepTraining_stepis_trainingval_accval_loss*A
dtypes7
523

�
save_3/control_dependencyIdentitysave_3/Const^save_3/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save_3/Const
�
save_3/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�3BAccuracy/Mean/moving_avgBAdam/beta1_powerBAdam/beta2_powerBConv2D/WBConv2D/W/AdamBConv2D/W/Adam_1BConv2D/bBConv2D/b/AdamBConv2D/b/Adam_1B
Conv2D_1/WBConv2D_1/W/AdamBConv2D_1/W/Adam_1B
Conv2D_1/bBConv2D_1/b/AdamBConv2D_1/b/Adam_1B
Conv2D_2/WBConv2D_2/W/AdamBConv2D_2/W/Adam_1B
Conv2D_2/bBConv2D_2/b/AdamBConv2D_2/b/Adam_1B
Conv2D_3/WBConv2D_3/W/AdamBConv2D_3/W/Adam_1B
Conv2D_3/bBConv2D_3/b/AdamBConv2D_3/b/Adam_1B
Conv2D_4/WBConv2D_4/W/AdamBConv2D_4/W/Adam_1B
Conv2D_4/bBConv2D_4/b/AdamBConv2D_4/b/Adam_1BCrossentropy/Mean/moving_avgBFullyConnected/WBFullyConnected/W/AdamBFullyConnected/W/Adam_1BFullyConnected/bBFullyConnected/b/AdamBFullyConnected/b/Adam_1BFullyConnected_1/WBFullyConnected_1/W/AdamBFullyConnected_1/W/Adam_1BFullyConnected_1/bBFullyConnected_1/b/AdamBFullyConnected_1/b/Adam_1BGlobal_StepBTraining_stepBis_trainingBval_accBval_loss*
dtype0*
_output_shapes
:3
�
!save_3/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:3*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523

�
save_3/AssignAssignAccuracy/Mean/moving_avgsave_3/RestoreV2*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_1AssignAdam/beta1_powersave_3/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_2AssignAdam/beta2_powersave_3/RestoreV2:2*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_3AssignConv2D/Wsave_3/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
�
save_3/Assign_4AssignConv2D/W/Adamsave_3/RestoreV2:4*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save_3/Assign_5AssignConv2D/W/Adam_1save_3/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
�
save_3/Assign_6AssignConv2D/bsave_3/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_7AssignConv2D/b/Adamsave_3/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_8AssignConv2D/b/Adam_1save_3/RestoreV2:8*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_9Assign
Conv2D_1/Wsave_3/RestoreV2:9*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save_3/Assign_10AssignConv2D_1/W/Adamsave_3/RestoreV2:10*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
�
save_3/Assign_11AssignConv2D_1/W/Adam_1save_3/RestoreV2:11*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
�
save_3/Assign_12Assign
Conv2D_1/bsave_3/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save_3/Assign_13AssignConv2D_1/b/Adamsave_3/RestoreV2:13*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_3/Assign_14AssignConv2D_1/b/Adam_1save_3/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_15Assign
Conv2D_2/Wsave_3/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
save_3/Assign_16AssignConv2D_2/W/Adamsave_3/RestoreV2:16*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0*
_class
loc:@Conv2D_2/W
�
save_3/Assign_17AssignConv2D_2/W/Adam_1save_3/RestoreV2:17*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save_3/Assign_18Assign
Conv2D_2/bsave_3/RestoreV2:18*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
�
save_3/Assign_19AssignConv2D_2/b/Adamsave_3/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save_3/Assign_20AssignConv2D_2/b/Adam_1save_3/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
�
save_3/Assign_21Assign
Conv2D_3/Wsave_3/RestoreV2:21*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save_3/Assign_22AssignConv2D_3/W/Adamsave_3/RestoreV2:22*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save_3/Assign_23AssignConv2D_3/W/Adam_1save_3/RestoreV2:23*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
save_3/Assign_24Assign
Conv2D_3/bsave_3/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_25AssignConv2D_3/b/Adamsave_3/RestoreV2:25*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
�
save_3/Assign_26AssignConv2D_3/b/Adam_1save_3/RestoreV2:26*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_3/Assign_27Assign
Conv2D_4/Wsave_3/RestoreV2:27*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save_3/Assign_28AssignConv2D_4/W/Adamsave_3/RestoreV2:28*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ *
use_locking(
�
save_3/Assign_29AssignConv2D_4/W/Adam_1save_3/RestoreV2:29*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
save_3/Assign_30Assign
Conv2D_4/bsave_3/RestoreV2:30*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_31AssignConv2D_4/b/Adamsave_3/RestoreV2:31*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D_4/b
�
save_3/Assign_32AssignConv2D_4/b/Adam_1save_3/RestoreV2:32*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D_4/b
�
save_3/Assign_33AssignCrossentropy/Mean/moving_avgsave_3/RestoreV2:33*
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_34AssignFullyConnected/Wsave_3/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save_3/Assign_35AssignFullyConnected/W/Adamsave_3/RestoreV2:35*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save_3/Assign_36AssignFullyConnected/W/Adam_1save_3/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
save_3/Assign_37AssignFullyConnected/bsave_3/RestoreV2:37*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
save_3/Assign_38AssignFullyConnected/b/Adamsave_3/RestoreV2:38*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save_3/Assign_39AssignFullyConnected/b/Adam_1save_3/RestoreV2:39*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_3/Assign_40AssignFullyConnected_1/Wsave_3/RestoreV2:40*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W
�
save_3/Assign_41AssignFullyConnected_1/W/Adamsave_3/RestoreV2:41*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_3/Assign_42AssignFullyConnected_1/W/Adam_1save_3/RestoreV2:42*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
save_3/Assign_43AssignFullyConnected_1/bsave_3/RestoreV2:43*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_44AssignFullyConnected_1/b/Adamsave_3/RestoreV2:44*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_3/Assign_45AssignFullyConnected_1/b/Adam_1save_3/RestoreV2:45*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_46AssignGlobal_Stepsave_3/RestoreV2:46*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Global_Step
�
save_3/Assign_47AssignTraining_stepsave_3/RestoreV2:47*
T0* 
_class
loc:@Training_step*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_48Assignis_trainingsave_3/RestoreV2:48*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_49Assignval_accsave_3/RestoreV2:49*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_50Assignval_losssave_3/RestoreV2:50*
use_locking(*
T0*
_class
loc:@val_loss*
validate_shape(*
_output_shapes
: 
�
save_3/restore_allNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9"�"�	
cond_context�	�	
�
Dropout/cond/cond_textDropout/cond/pred_id:0Dropout/cond/switch_t:0 *�
Dropout/cond/dropout/Cast:0
#Dropout/cond/dropout/GreaterEqual:0
#Dropout/cond/dropout/Shape/Switch:1
Dropout/cond/dropout/Shape:0
Dropout/cond/dropout/mul:0
Dropout/cond/dropout/mul_1:0
3Dropout/cond/dropout/random_uniform/RandomUniform:0
)Dropout/cond/dropout/random_uniform/max:0
)Dropout/cond/dropout/random_uniform/min:0
)Dropout/cond/dropout/random_uniform/mul:0
)Dropout/cond/dropout/random_uniform/sub:0
%Dropout/cond/dropout/random_uniform:0
Dropout/cond/dropout/rate:0
Dropout/cond/dropout/sub/x:0
Dropout/cond/dropout/sub:0
 Dropout/cond/dropout/truediv/x:0
Dropout/cond/dropout/truediv:0
Dropout/cond/pred_id:0
Dropout/cond/switch_t:0
FullyConnected/Relu:00
Dropout/cond/pred_id:0Dropout/cond/pred_id:0<
FullyConnected/Relu:0#Dropout/cond/dropout/Shape/Switch:1
�
Dropout/cond/cond_text_1Dropout/cond/pred_id:0Dropout/cond/switch_f:0*�
Dropout/cond/Switch_1:0
Dropout/cond/Switch_1:1
Dropout/cond/pred_id:0
Dropout/cond/switch_f:0
FullyConnected/Relu:00
Dropout/cond/pred_id:0Dropout/cond/pred_id:00
FullyConnected/Relu:0Dropout/cond/Switch_1:0"R
 layer_variables/FullyConnected_1.
,
FullyConnected_1/W:0
FullyConnected_1/b:0"0
layer_tensor/Dropout

Dropout/cond/Merge:0"
trainops

Adam":
layer_variables/Conv2D_1

Conv2D_1/W:0
Conv2D_1/b:0";
is_training_ops(
&
Dropout/Assign:0
Dropout/Assign_1:0":
layer_variables/Conv2D_2

Conv2D_2/W:0
Conv2D_2/b:0":
layer_variables/Conv2D_3

Conv2D_3/W:0
Conv2D_3/b:0":
layer_variables/Conv2D_4

Conv2D_4/W:0
Conv2D_4/b:0"8
layer_tensor/FullyConnected

FullyConnected/Relu:0"#
layer_tensor/input

	input/X:0"L
layer_variables/FullyConnected*
(
FullyConnected/W:0
FullyConnected/b:0"�
model_variables��
W

Conv2D/W:0Conv2D/W/AssignConv2D/W/read:02%Conv2D/W/Initializer/random_uniform:08
N

Conv2D/b:0Conv2D/b/AssignConv2D/b/read:02Conv2D/b/Initializer/Const:08
_
Conv2D_1/W:0Conv2D_1/W/AssignConv2D_1/W/read:02'Conv2D_1/W/Initializer/random_uniform:08
V
Conv2D_1/b:0Conv2D_1/b/AssignConv2D_1/b/read:02Conv2D_1/b/Initializer/Const:08
_
Conv2D_2/W:0Conv2D_2/W/AssignConv2D_2/W/read:02'Conv2D_2/W/Initializer/random_uniform:08
V
Conv2D_2/b:0Conv2D_2/b/AssignConv2D_2/b/read:02Conv2D_2/b/Initializer/Const:08
_
Conv2D_3/W:0Conv2D_3/W/AssignConv2D_3/W/read:02'Conv2D_3/W/Initializer/random_uniform:08
V
Conv2D_3/b:0Conv2D_3/b/AssignConv2D_3/b/read:02Conv2D_3/b/Initializer/Const:08
_
Conv2D_4/W:0Conv2D_4/W/AssignConv2D_4/W/read:02'Conv2D_4/W/Initializer/random_uniform:08
V
Conv2D_4/b:0Conv2D_4/b/AssignConv2D_4/b/read:02Conv2D_4/b/Initializer/Const:08
y
FullyConnected/W:0FullyConnected/W/AssignFullyConnected/W/read:02/FullyConnected/W/Initializer/truncated_normal:08
n
FullyConnected/b:0FullyConnected/b/AssignFullyConnected/b/read:02$FullyConnected/b/Initializer/Const:08
X
is_training:0is_training/Assignis_training/read:02is_training/Initializer/Const:0
�
FullyConnected_1/W:0FullyConnected_1/W/AssignFullyConnected_1/W/read:021FullyConnected_1/W/Initializer/truncated_normal:08
v
FullyConnected_1/b:0FullyConnected_1/b/AssignFullyConnected_1/b/read:02&FullyConnected_1/b/Initializer/Const:08"6
Adam_training_summaries

Loss:0
Adam/Loss/raw:0" 
is_training

is_training:0"
targets

targets/Y:0"
inputs

	input/X:0"�+
	variables�+�+
W

Conv2D/W:0Conv2D/W/AssignConv2D/W/read:02%Conv2D/W/Initializer/random_uniform:08
N

Conv2D/b:0Conv2D/b/AssignConv2D/b/read:02Conv2D/b/Initializer/Const:08
_
Conv2D_1/W:0Conv2D_1/W/AssignConv2D_1/W/read:02'Conv2D_1/W/Initializer/random_uniform:08
V
Conv2D_1/b:0Conv2D_1/b/AssignConv2D_1/b/read:02Conv2D_1/b/Initializer/Const:08
_
Conv2D_2/W:0Conv2D_2/W/AssignConv2D_2/W/read:02'Conv2D_2/W/Initializer/random_uniform:08
V
Conv2D_2/b:0Conv2D_2/b/AssignConv2D_2/b/read:02Conv2D_2/b/Initializer/Const:08
_
Conv2D_3/W:0Conv2D_3/W/AssignConv2D_3/W/read:02'Conv2D_3/W/Initializer/random_uniform:08
V
Conv2D_3/b:0Conv2D_3/b/AssignConv2D_3/b/read:02Conv2D_3/b/Initializer/Const:08
_
Conv2D_4/W:0Conv2D_4/W/AssignConv2D_4/W/read:02'Conv2D_4/W/Initializer/random_uniform:08
V
Conv2D_4/b:0Conv2D_4/b/AssignConv2D_4/b/read:02Conv2D_4/b/Initializer/Const:08
y
FullyConnected/W:0FullyConnected/W/AssignFullyConnected/W/read:02/FullyConnected/W/Initializer/truncated_normal:08
n
FullyConnected/b:0FullyConnected/b/AssignFullyConnected/b/read:02$FullyConnected/b/Initializer/Const:08
X
is_training:0is_training/Assignis_training/read:02is_training/Initializer/Const:0
�
FullyConnected_1/W:0FullyConnected_1/W/AssignFullyConnected_1/W/read:021FullyConnected_1/W/Initializer/truncated_normal:08
v
FullyConnected_1/b:0FullyConnected_1/b/AssignFullyConnected_1/b/read:02&FullyConnected_1/b/Initializer/Const:08
\
Training_step:0Training_step/AssignTraining_step/read:02Training_step/initial_value:0
T
Global_Step:0Global_Step/AssignGlobal_Step/read:02Global_Step/initial_value:0
H

val_loss:0val_loss/Assignval_loss/read:02val_loss/initial_value:0
D
	val_acc:0val_acc/Assignval_acc/read:02val_acc/initial_value:0
�
Accuracy/Mean/moving_avg:0Accuracy/Mean/moving_avg/AssignAccuracy/Mean/moving_avg/read:02,Accuracy/Mean/moving_avg/Initializer/zeros:0
�
Crossentropy/Mean/moving_avg:0#Crossentropy/Mean/moving_avg/Assign#Crossentropy/Mean/moving_avg/read:020Crossentropy/Mean/moving_avg/Initializer/zeros:0
h
Adam/beta1_power:0Adam/beta1_power/AssignAdam/beta1_power/read:02 Adam/beta1_power/initial_value:0
h
Adam/beta2_power:0Adam/beta2_power/AssignAdam/beta2_power/read:02 Adam/beta2_power/initial_value:0
`
Conv2D/W/Adam:0Conv2D/W/Adam/AssignConv2D/W/Adam/read:02!Conv2D/W/Adam/Initializer/zeros:0
h
Conv2D/W/Adam_1:0Conv2D/W/Adam_1/AssignConv2D/W/Adam_1/read:02#Conv2D/W/Adam_1/Initializer/zeros:0
`
Conv2D/b/Adam:0Conv2D/b/Adam/AssignConv2D/b/Adam/read:02!Conv2D/b/Adam/Initializer/zeros:0
h
Conv2D/b/Adam_1:0Conv2D/b/Adam_1/AssignConv2D/b/Adam_1/read:02#Conv2D/b/Adam_1/Initializer/zeros:0
h
Conv2D_1/W/Adam:0Conv2D_1/W/Adam/AssignConv2D_1/W/Adam/read:02#Conv2D_1/W/Adam/Initializer/zeros:0
p
Conv2D_1/W/Adam_1:0Conv2D_1/W/Adam_1/AssignConv2D_1/W/Adam_1/read:02%Conv2D_1/W/Adam_1/Initializer/zeros:0
h
Conv2D_1/b/Adam:0Conv2D_1/b/Adam/AssignConv2D_1/b/Adam/read:02#Conv2D_1/b/Adam/Initializer/zeros:0
p
Conv2D_1/b/Adam_1:0Conv2D_1/b/Adam_1/AssignConv2D_1/b/Adam_1/read:02%Conv2D_1/b/Adam_1/Initializer/zeros:0
h
Conv2D_2/W/Adam:0Conv2D_2/W/Adam/AssignConv2D_2/W/Adam/read:02#Conv2D_2/W/Adam/Initializer/zeros:0
p
Conv2D_2/W/Adam_1:0Conv2D_2/W/Adam_1/AssignConv2D_2/W/Adam_1/read:02%Conv2D_2/W/Adam_1/Initializer/zeros:0
h
Conv2D_2/b/Adam:0Conv2D_2/b/Adam/AssignConv2D_2/b/Adam/read:02#Conv2D_2/b/Adam/Initializer/zeros:0
p
Conv2D_2/b/Adam_1:0Conv2D_2/b/Adam_1/AssignConv2D_2/b/Adam_1/read:02%Conv2D_2/b/Adam_1/Initializer/zeros:0
h
Conv2D_3/W/Adam:0Conv2D_3/W/Adam/AssignConv2D_3/W/Adam/read:02#Conv2D_3/W/Adam/Initializer/zeros:0
p
Conv2D_3/W/Adam_1:0Conv2D_3/W/Adam_1/AssignConv2D_3/W/Adam_1/read:02%Conv2D_3/W/Adam_1/Initializer/zeros:0
h
Conv2D_3/b/Adam:0Conv2D_3/b/Adam/AssignConv2D_3/b/Adam/read:02#Conv2D_3/b/Adam/Initializer/zeros:0
p
Conv2D_3/b/Adam_1:0Conv2D_3/b/Adam_1/AssignConv2D_3/b/Adam_1/read:02%Conv2D_3/b/Adam_1/Initializer/zeros:0
h
Conv2D_4/W/Adam:0Conv2D_4/W/Adam/AssignConv2D_4/W/Adam/read:02#Conv2D_4/W/Adam/Initializer/zeros:0
p
Conv2D_4/W/Adam_1:0Conv2D_4/W/Adam_1/AssignConv2D_4/W/Adam_1/read:02%Conv2D_4/W/Adam_1/Initializer/zeros:0
h
Conv2D_4/b/Adam:0Conv2D_4/b/Adam/AssignConv2D_4/b/Adam/read:02#Conv2D_4/b/Adam/Initializer/zeros:0
p
Conv2D_4/b/Adam_1:0Conv2D_4/b/Adam_1/AssignConv2D_4/b/Adam_1/read:02%Conv2D_4/b/Adam_1/Initializer/zeros:0
�
FullyConnected/W/Adam:0FullyConnected/W/Adam/AssignFullyConnected/W/Adam/read:02)FullyConnected/W/Adam/Initializer/zeros:0
�
FullyConnected/W/Adam_1:0FullyConnected/W/Adam_1/AssignFullyConnected/W/Adam_1/read:02+FullyConnected/W/Adam_1/Initializer/zeros:0
�
FullyConnected/b/Adam:0FullyConnected/b/Adam/AssignFullyConnected/b/Adam/read:02)FullyConnected/b/Adam/Initializer/zeros:0
�
FullyConnected/b/Adam_1:0FullyConnected/b/Adam_1/AssignFullyConnected/b/Adam_1/read:02+FullyConnected/b/Adam_1/Initializer/zeros:0
�
FullyConnected_1/W/Adam:0FullyConnected_1/W/Adam/AssignFullyConnected_1/W/Adam/read:02+FullyConnected_1/W/Adam/Initializer/zeros:0
�
FullyConnected_1/W/Adam_1:0 FullyConnected_1/W/Adam_1/Assign FullyConnected_1/W/Adam_1/read:02-FullyConnected_1/W/Adam_1/Initializer/zeros:0
�
FullyConnected_1/b/Adam:0FullyConnected_1/b/Adam/AssignFullyConnected_1/b/Adam/read:02+FullyConnected_1/b/Adam/Initializer/zeros:0
�
FullyConnected_1/b/Adam_1:0 FullyConnected_1/b/Adam_1/Assign FullyConnected_1/b/Adam_1/read:02-FullyConnected_1/b/Adam_1/Initializer/zeros:0"$
train_op

Adam/apply_grad_op_0"?
layer_tensor/FullyConnected_1

FullyConnected_1/Softmax:0",
layer_tensor/Conv2D_1

Conv2D_1/Relu:0",
layer_tensor/Conv2D_2

Conv2D_2/Relu:0"�
layer_tensor/MaxPool2Ds
q
MaxPool2D/MaxPool:0
MaxPool2D_1/MaxPool:0
MaxPool2D_2/MaxPool:0
MaxPool2D_3/MaxPool:0
MaxPool2D_4/MaxPool:0",
layer_tensor/Conv2D_3

Conv2D_3/Relu:0",
layer_tensor/Conv2D_4

Conv2D_4/Relu:0"(
	summaries

Loss:0
Adam/Loss/raw:0"(
layer_tensor/Conv2D

Conv2D/Relu:0"�
trainable_variables�
�

W

Conv2D/W:0Conv2D/W/AssignConv2D/W/read:02%Conv2D/W/Initializer/random_uniform:08
N

Conv2D/b:0Conv2D/b/AssignConv2D/b/read:02Conv2D/b/Initializer/Const:08
_
Conv2D_1/W:0Conv2D_1/W/AssignConv2D_1/W/read:02'Conv2D_1/W/Initializer/random_uniform:08
V
Conv2D_1/b:0Conv2D_1/b/AssignConv2D_1/b/read:02Conv2D_1/b/Initializer/Const:08
_
Conv2D_2/W:0Conv2D_2/W/AssignConv2D_2/W/read:02'Conv2D_2/W/Initializer/random_uniform:08
V
Conv2D_2/b:0Conv2D_2/b/AssignConv2D_2/b/read:02Conv2D_2/b/Initializer/Const:08
_
Conv2D_3/W:0Conv2D_3/W/AssignConv2D_3/W/read:02'Conv2D_3/W/Initializer/random_uniform:08
V
Conv2D_3/b:0Conv2D_3/b/AssignConv2D_3/b/read:02Conv2D_3/b/Initializer/Const:08
_
Conv2D_4/W:0Conv2D_4/W/AssignConv2D_4/W/read:02'Conv2D_4/W/Initializer/random_uniform:08
V
Conv2D_4/b:0Conv2D_4/b/AssignConv2D_4/b/read:02Conv2D_4/b/Initializer/Const:08
y
FullyConnected/W:0FullyConnected/W/AssignFullyConnected/W/read:02/FullyConnected/W/Initializer/truncated_normal:08
n
FullyConnected/b:0FullyConnected/b/AssignFullyConnected/b/read:02$FullyConnected/b/Initializer/Const:08
�
FullyConnected_1/W:0FullyConnected_1/W/AssignFullyConnected_1/W/read:021FullyConnected_1/W/Initializer/truncated_normal:08
v
FullyConnected_1/b:0FullyConnected_1/b/AssignFullyConnected_1/b/read:02&FullyConnected_1/b/Initializer/Const:08"�
activations�
�
Conv2D/Relu:0
MaxPool2D/MaxPool:0
Conv2D_1/Relu:0
MaxPool2D_1/MaxPool:0
Conv2D_2/Relu:0
MaxPool2D_2/MaxPool:0
Conv2D_3/Relu:0
MaxPool2D_3/MaxPool:0
Conv2D_4/Relu:0
MaxPool2D_4/MaxPool:0
FullyConnected/Relu:0
FullyConnected_1/Softmax:0"4
layer_variables/Conv2D


Conv2D/W:0

Conv2D/b:0=�P�0       ���_	���>~��A*#

Loss���?

Adam/Loss/raw�>�?C�7 0       ���_	DL�>~��A*#

Loss#L�?

Adam/Loss/raw㌴?W�u0       ���_	e>~��A*#

Loss���?

Adam/Loss/raws��?�]��0       ���_	�Ę>~��A*#

Loss��?

Adam/Loss/raw�?���M0       ���_	
��>~��A*#

LossN��?

Adam/Loss/raw���?v�!f0       ���_	|D�>~��A*#

Loss�R�?

Adam/Loss/raw�4�?A�%�0       ���_	�	�>~��A*#

Loss��?

Adam/Loss/raw$I�?w�3�0       ���_	�Ы>~��A	*#

Loss��?

Adam/Loss/raw%��?RV��0       ���_	�u�>~��A
*#

Loss� �?

Adam/Loss/raw�v�?��V0       ���_	,�>~��A*#

Loss(��?

Adam/Loss/raw�@�?�6�y0       ���_	p\�>~��A*#

Loss�{�?

Adam/Loss/raw"(�?7u4Q0       ���_	77�>~��A*#

Loss���?

Adam/Loss/rawl��?$vX�0       ���_	{Q�>~��A*#

LossN��?

Adam/Loss/rawP�?4� ]0       ���_	u�>~��A*#

Loss�m�?

Adam/Loss/raw�!�?�{�D0       ���_	P4�>~��A*#

LossJ�?

Adam/Loss/raw���?��}0       ���_	���>~��A*#

Loss��?

Adam/Loss/raw�V�?�hMW0       ���_	a��>~��A*#

Loss�v�?

Adam/Loss/raw%��?*�x#0       ���_	K��>~��A*#

LossgҶ?

Adam/Loss/raw���?'ĲT0       ���_	��>~��A*#

Loss3C�?

Adam/Loss/raw!Ū?"��0       ���_	���>~��A*#

Loss}��?

Adam/Loss/rawZ��?�x�~0       ���_	�(?~��A*#

Loss�Z�?

Adam/Loss/raw���?J�XJ0       ���_	�
?~��A*#

Lossb��?

Adam/Loss/raw,2�?H��*0       ���_	N�?~��A*#

Loss�&�?

Adam/Loss/raw���?5+��0       ���_	�?~��A*#

Loss�?

Adam/Loss/raw(1�?hWDh0       ���_	.�?~��A*#

Loss�i�?

Adam/Loss/raw�$�?l�60       ���_	(�!?~��A*#

Loss��?

Adam/Loss/raw�ֻ?� H0       ���_	�'?~��A*#

Loss}��?

Adam/Loss/raw�x�?,U60       ���_	|�)?~��A*#

Loss�z�?

Adam/Loss/raw[@�?'�~0       ���_	�9+?~��A*#

Loss�D�?

Adam/Loss/raw)��?
�&0       ���_	�2?~��A*#

Loss���?

Adam/Loss/raw���?J�=0       ���_	}8?~��A *#

Loss.@�?

Adam/Loss/raw6n�?�W��0       ���_	(�=?~��A!*#

LossO1�?

Adam/Loss/rawؖ?g�yI0       ���_	(�C?~��A"*#

LossTg�?

Adam/Loss/rawn��?o�`f0       ���_	̙I?~��A#*#

LossY��?

Adam/Loss/raw��?D�@c0       ���_	�pO?~��A$*#

Loss�Ů?

Adam/Loss/rawJ�?�2'�0       ���_	��U?~��A%*#

LossD�?

Adam/Loss/raweG�?t���0       ���_	�[?~��A&*#

Loss���?

Adam/Loss/raw?��?Ђ(10       ���_	.�]?~��A'*#

Loss_,�?

Adam/Loss/raw���?R��h0       ���_	�J_?~��A(*#

Loss�g�?

Adam/Loss/raw@�q?���0       ���_	pe?~��A)*#

Lossܞ?

Adam/Loss/raw�G�?-1��0       ���_	q<k?~��A**#

Loss�ڝ?

Adam/Loss/raw�/�?z}�00       ���_	�#q?~��A+*#

Loss45�?

Adam/Loss/raw��?�XoL0       ���_	�v?~��A,*#

Loss-��?

Adam/Loss/raw9��?���0       ���_	�|?~��A-*#

Loss4	�?

Adam/Loss/raw��?���0       ���_	"�?~��A.*#

LossI0�?

Adam/Loss/raw�6�?	���0       ���_	KW�?~��A/*#

Lossω�?

Adam/Loss/raw�^�?A�;�0       ���_	B?�?~��A0*#

Loss�p�?

Adam/Loss/raw`z�?r.D0       ���_	��?~��A1*#

Loss���?

Adam/Loss/raw�)�?_+b0       ���_	㉒?~��A2*#

LossC �?

Adam/Loss/rawXo�?�*0       ���_	ܟ�?~��A3*#

Loss(�?

Adam/Loss/raw/��?z$��0       ���_	��?~��A4*#

Loss�ӡ?

Adam/Loss/raw���?]�U�0       ���_	�B�?~��A5*#

Lossb��?

Adam/Loss/rawRy�?g�G0       ���_	}�?~��A6*#

Loss�?

Adam/Loss/raw�?s��:0       ���_	I/�?~��A7*#

Loss��?

Adam/Loss/rawKq�?6d<�0       ���_	9�?~��A8*#

Loss��?

Adam/Loss/rawԐ�?jCGF0       ���_	�N�?~��A9*#

LossjK�?

Adam/Loss/raw(��?+r0       ���_	��?~��A:*#

Loss��?

Adam/Loss/raw�`�?�<|0       ���_	�1�?~��A;*#

Loss|��?

Adam/Loss/raw���?�+^�0       ���_	��?~��A<*#

Loss!Ś?

Adam/Loss/raw-�?S"��0       ���_	�#�?~��A=*#

LossK��?

Adam/Loss/rawIu?k��0       ���_	��?~��A>*#

Loss#��?

Adam/Loss/raw�$�?�e�0       ���_	���?~��A?*#

Loss/��?

Adam/Loss/raw�M?�&�0       ���_	v�?~��A@*#

Loss��?

Adam/Loss/raw�Y?"Sk�0       ���_	b�?~��AA*#

Loss��?

Adam/Loss/raw�-�?���0       ���_	m7�?~��AB*#

Loss���?

Adam/Loss/raw�1Y?o�v0       ���_	��?~��AC*#

Lossm�?

Adam/Loss/rawɊ�?�p�60       ���_	1��?~��AD*#

Loss�0�?

Adam/Loss/raw*2z?���w0       ���_	��@~��AE*#

Loss��?

Adam/Loss/raw���?_{��0       ���_	��@~��AF*#

Loss�q�?

Adam/Loss/raw��z?�F��0       ���_	�W
@~��AG*#

Loss�h�?

Adam/Loss/raw*b?4K{0       ���_	�#@~��AH*#

Loss��?

Adam/Loss/raw�~W?�q��0       ���_	�I"@~��AI*#

Loss���?

Adam/Loss/rawkro?X���0       ���_	�F.@~��AJ*#

Loss`?

Adam/Loss/raw�x?m��0       ���_	,e:@~��AK*#

Loss�n~?

Adam/Loss/raw&�e?a�.0       ���_	�IF@~��AL*#

Loss$�{?

Adam/Loss/raw��?C-ۧ0       ���_	�Q@~��AM*#

Loss 	r?

Adam/Loss/raw�@��B0       ���_	�)^@~��AN*#

Loss��?

Adam/Loss/raw�V�?�m*�0       ���_	b@~��AO*#

Loss��?

Adam/Loss/raw�Wp?ߎC0       ���_	g)e@~��AP*#

Loss���?

Adam/Loss/rawNwT?��p`0       ���_	3�q@~��AQ*#

Loss䅆?

Adam/Loss/raw��K?�G0       ���_	d}@~��AR*#

LossXD�?

Adam/Loss/raw�~'?L�@0       ���_	�<�@~��AS*#

Loss�}?

Adam/Loss/raw)CV?�?��0       ���_	3�@~��AT*#

Loss8'y?

Adam/Loss/rawJ5?�X�0       ���_	�ԡ@~��AU*#

Loss�Vr?

Adam/Loss/raw�ς?2tU�0       ���_	iW�@~��AV*#

LosssDt?

Adam/Loss/raw4�C?�� 0       ���_	��@~��AW*#

Loss�fo?

Adam/Loss/raw�^6@a��0       ���_	=�@~��AX*#

LossW4�?

Adam/Loss/raw �c?M�q:0       ���_	)��@~��AY*#

Loss�*�?

Adam/Loss/raw���?�pA0       ���_	~�@~��AZ*#

Loss���?

Adam/Loss/raw2.K?Qk��0       ���_	�@~��A[*#

Loss1Ո?

Adam/Loss/raw��?���0       ���_	b��@~��A\*#

LossC(�?

Adam/Loss/raw]�?�ՏE0       ���_	��@~��A]*#

Loss�?

Adam/Loss/rawLB�?��0       ���_	���@~��A^*#

Lossȋ�?

Adam/Loss/rawd\}?��H�0       ���_	��A~��A_*#

Loss9��?

Adam/Loss/raw� 5?�N��0       ���_	�A~��A`*#

Loss�B�?

Adam/Loss/rawzL@?�K��0       ���_	%XA~��Aa*#

Loss&�?

Adam/Loss/raw�k@�i��0       ���_	Gx+A~��Ab*#

Loss�	�?

Adam/Loss/raw�C�?;7T�0       ���_	�I/A~��Ac*#

Lossv�?

Adam/Loss/raw��W?��&0       ���_	�2A~��Ad*#

Loss8��?

Adam/Loss/rawn�>?#���0       ���_	nn?A~��Ae*#

Loss�?

Adam/Loss/raw=5g?����0       ���_	�nKA~��Af*#

Lossᘚ?

Adam/Loss/raw�e?���0       ���_	o�WA~��Ag*#

Loss���?

Adam/Loss/raw�"T?��y0       ���_	�!cA~��Ah*#

Loss�"�?

Adam/Loss/raw+�n?N�d�0       ���_	#�nA~��Ai*#

Loss�s�?

Adam/Loss/rawhWL?],�0       ���_	�^zA~��Aj*#

Loss�R�?

Adam/Loss/raw�~R?#|�0       ���_	�k�A~��Ak*#

Lossh�?

Adam/Loss/raw.+@�5��0       ���_	��A~��Al*#

Loss��?

Adam/Loss/raw�CG?���0       ���_	K�A~��Am*#

LossLؖ?

Adam/Loss/raw~rY?���0       ���_	@�A~��An*#

Loss���?

Adam/Loss/raw��7?ఐ0       ���_	��A~��Ao*#

Loss�%�?

Adam/Loss/raw41?sؼ�0       ���_	�{�A~��Ap*#

Loss��?

Adam/Loss/raw0�H?�]NQ0       ���_	q�A~��Aq*#

LossMV�?

Adam/Loss/rawP�I?���0       ���_	���A~��Ar*#

Loss|/�?

Adam/Loss/raw�Z?�o�0       ���_	_]�A~��As*#

Loss�g~?

Adam/Loss/raw��R?��30       ���_	<��A~��At*#

Loss	z?

Adam/Loss/rawL�C?$�d0       ���_	��A~��Au*#

Lossʛt?

Adam/Loss/raw�X!@,�0       ���_	_��A~��Av*#

Loss�W�?

Adam/Loss/raw,�0?B���0       ���_	t#B~��Aw*#

Loss���?

Adam/Loss/raw�^,?9i0       ���_	�B~��Ax*#

Loss�݃?

Adam/Loss/raw}?yZS�0       ���_	�"B~��Ay*#

Loss2}?

Adam/Loss/raw��H?=F!0       ���_	��B~��Az*#

LossY�w?

Adam/Loss/raw�N#?�D��0       ���_	�n*B~��A{*#

Losswao?

Adam/Loss/raw��X?��0       ���_	 �5B~��A|*#

Loss#m?

Adam/Loss/raw��"?��?�0       ���_	�AB~��A}*#

Lossεe?

Adam/Loss/raw�QJ?n l�0       ���_	\vMB~��A~*#

Loss��b?

Adam/Loss/raw�W)?���	0       ���_	I�YB~��A*#

LossT5]?

Adam/Loss/rawc;@���D1       ����	��eB~��A�*#

Lossi�?

Adam/Loss/raw��#?.��$1       ����	5�iB~��A�*#

LossP��?

Adam/Loss/rawx�c?sߧ1       ����	o�mB~��A�*#

LossA��?

Adam/Loss/raw��D?�f�1       ����	M{B~��A�*#

Loss 3}?

Adam/Loss/raw�>�:�g1       ����	���B~��A�*#

LossQ�o?

Adam/Loss/raw9'?.42�1       ����	�-�B~��A�*#

Lossh�h?

Adam/Loss/rawU�B? 팍1       ����	�ӷB~��A�*#

LossL�d?

Adam/Loss/raw Y?_��1       ����	S��B~��A�*#

Loss��\?

Adam/Loss/raw�77?�1��1       ����	��B~��A�*#

Lossw�X?

Adam/Loss/raw��!?G_j1       ����	���B~��A�*#

Loss@S?

Adam/Loss/raw^/A@��~1       ����	
aC~��A�*#

Loss��?

Adam/Loss/rawZ�%?m��V1       ����	��	C~��A�*#

LossꝀ?

Adam/Loss/raw��?v��1       ����	iC~��A�*#

Loss�u?

Adam/Loss/rawR��>l51       ����	,J#C~��A�*#

Loss�h?

Adam/Loss/raw{g?К��1       ����	J�2C~��A�*#

Loss��`?

Adam/Loss/raw�!*?ѯ\A1       ����	��CC~��A�*#

Lossa/[?

Adam/Loss/raw��>�U�1       ����	�hUC~��A�*#

Losse R?

Adam/Loss/raw�!?���1       ����	��pC~��A�*#

LossX�L?

Adam/Loss/rawg�	?�\1       ����	wK�C~��A�*#

Loss��E?

Adam/Loss/raw�T?��w�1       ����	_b�C~��A�*#

Loss�??

Adam/Loss/raw'�b@��e�1       ����	�^�C~��A�*#

Lossږ�?

Adam/Loss/raw�U�>N�a1       ����	\?�C~��A�*#

Loss��x?

Adam/Loss/raw��Y?z�M�1       ����	AG�C~��A�*#

Loss9�u?

Adam/Loss/raw%�7?��M1       ����	���C~��A�*#

Loss�o?

Adam/Loss/raw�5?�;�-1       ����	C�C~��A�*#

Loss4�i?

Adam/Loss/raw�,?�qh�1       ����	��C~��A�*#

Lossb�a?

Adam/Loss/raw
!?�1       ����	u��C~��A�*#

Lossx[?

Adam/Loss/raw�c'?#��1       ����	tb�C~��A�*#

Loss�BV?

Adam/Loss/raw���>��Q1       ����	+D~��A�*#

Loss�M?

Adam/Loss/rawWc1?���1       ����	��D~��A�*#

Loss EJ?

Adam/Loss/rawq�6@ظ<1       ����	HoD~��A�*#

Loss�?

Adam/Loss/raw�6?V�� 1       ����	,,"D~��A�*#

Loss�t?

Adam/Loss/raw1Ci?8kS�1       ����	7:%D~��A�*#

Loss`s?

Adam/Loss/raw�D?��1       ����	rl1D~��A�*#

Loss{�n?

Adam/Loss/raw �?}PO�1       ����	9�<D~��A�*#

Loss%�e?

Adam/Loss/raw�_?��4`1       ����	P�HD~��A�*#

Loss~Be?

Adam/Loss/raw�I�>$83�1       ����	{PUD~��A�*#

Loss��Z?

Adam/Loss/rawt�?�)1       ����	�aD~��A�*#

Loss��T?

Adam/Loss/raw�%?�/<�1       ����	��lD~��A�*#

Loss=�N?

Adam/Loss/raw8~�>�|^1       ����	"MxD~��A�*#

Loss��F?

Adam/Loss/raw��=@h��^1       ����	�m�D~��A�*#

Loss��~?

Adam/Loss/raw���>�c�1       ����	d@�D~��A�*#

Loss��p?

Adam/Loss/rawi ?����1       ����	s1�D~��A�*#

Loss3�h?

Adam/Loss/rawč?���1       ����	� �D~��A�*#

Lossu�_?

Adam/Loss/rawt�>��n1       ����	b�D~��A�*#

Loss�^U?

Adam/Loss/rawi�)?^,(81       ����	�D~��A�*#

Loss��P?

Adam/Loss/rawi��>�tn1       ����	ū�D~��A�*#

Loss*�H?

Adam/Loss/raw�?�h�1       ����	��D~��A�*#

Loss�6B?

Adam/Loss/rawW*?gG0�1       ����	�m�D~��A�*#

Loss��;?

Adam/Loss/raw���>�:�1       ����	w�D~��A�*#

LossFE2?

Adam/Loss/raw�ѓ@l>}1       ����	�c�D~��A�*#

LossxY�?

Adam/Loss/rawJJ?M��1       ����	ޒ�D~��A�*#

Loss<�?

Adam/Loss/raw�5�>է��1       ����	ލ�D~��A�*#

Lossgy?

Adam/Loss/raw'7�>>��(1       ����	�� E~��A�*#

Loss�j?

Adam/Loss/raw ��>��Ct1       ����	Y�E~��A�*#

Loss4%]?

Adam/Loss/raw{��>V �y1       ����	�E~��A�*#

Loss��R?

Adam/Loss/raw��>�>�31       ����	��$E~��A�*#

Loss֫F?

Adam/Loss/raw��>\�1       ����	��0E~��A�*#

Loss��<?

Adam/Loss/raw�?A1       ����	�J<E~��A�*#

Loss�6?

Adam/Loss/raw��>L�=-1       ����	x(IE~��A�*#

Loss�*?

Adam/Loss/raw+c�>C��1       ����	�3UE~��A�*#

Loss�W%?

Adam/Loss/raw���>TT�)1       ����	74YE~��A�*#

Loss6�?

Adam/Loss/raw��> ���1       ����	T8\E~��A�*#

Loss�%?

Adam/Loss/raw~�>��Rg1       ����	'�iE~��A�*#

Loss'V?

Adam/Loss/raw���>W��1       ����	.�uE~��A�*#

Loss�z
?

Adam/Loss/raw�>(�
1       ����	��E~��A�*#

Loss��?

Adam/Loss/raw
ja>�H�1       ����	Ɗ�E~��A�*#

Loss��>

Adam/Loss/raw���>��xg1       ����	���E~��A�*#

Loss/��>

Adam/Loss/raw�)�>�g�1       ����	ĳ�E~��A�*#

Lossw�>

Adam/Loss/raw���>�pk�1       ����	\�E~��A�*#

Loss�y�>

Adam/Loss/rawi��@L�6�1       ����	H��E~��A�*#

Lossb�y?

Adam/Loss/raw��$?�L��1       ����	�E~��A�*#

LossSq?

Adam/Loss/raw�K7?n�t1       ����	}��E~��A�*#

Loss��k?

Adam/Loss/raww��>@"+ 1       ����	�0�E~��A�*#

Lossx�`?

Adam/Loss/rawI�$?���{1       ����	���E~��A�*#

Loss��Z?

Adam/Loss/raw�)�>0~y1       ����	�B�E~��A�*#

Loss��N?

Adam/Loss/raw�-~>�)��1       ����	e��E~��A�*#

Loss��@?

Adam/Loss/rawc��>g��1       ����	���E~��A�*#

Lossd6?

Adam/Loss/rawhS�>��1       ����	��F~��A�*#

Loss��,?

Adam/Loss/raw��>[}�61       ����	��F~��A�*#

Loss�?&?

Adam/Loss/raw��k@�61       ����	��F~��A�*#

Loss��s?

Adam/Loss/raw���>�ꝟ1       ����	7�F~��A�*#

Loss�e?

Adam/Loss/raw%;�>
(.�1       ����	�kF~��A�*#

Loss��W?

Adam/Loss/raw��>��G�1       ����	�k!F~��A�*#

Loss`�J?

Adam/Loss/rawv��>�;xF1       ����	�(F~��A�*#

LossB�??

Adam/Loss/raw\��>_�Fs1       ����	iV0F~��A�*#

Loss�\6?

Adam/Loss/raw�>,'��1       ����	�7F~��A�*#

Loss�S,?

Adam/Loss/rawS�>(
V�1       ����	̝>F~��A�*#

Loss1!?

Adam/Loss/raw��>����1       ����	��EF~��A�*#

Loss��?

Adam/Loss/raw*��>x�@G1       ����	�LF~��A�*#

LossC0?

Adam/Loss/raw�ɒ@V_��1       ����	�YSF~��A�*#

Loss y?

Adam/Loss/raw\�>���S1       ����	�OUF~��A�*#

Loss��i?

Adam/Loss/rawf,?�得1       ����	�VF~��A�*#

Loss��c?

Adam/Loss/raw���>5��_1       ����	�U]F~��A�*#

Loss�iV?

Adam/Loss/rawΏ>���1       ����	� cF~��A�*#

Loss�)H?

Adam/Loss/rawVW�>�ц�1       ����	/�hF~��A�*#

Loss&P>?

Adam/Loss/rawf�p>��z�1       ����	0�nF~��A�*#

Loss�M1?

Adam/Loss/raw���>�{�1       ����	xbtF~��A�*#

Loss �*?

Adam/Loss/raw욥>"�(�1       ����	�\zF~��A�*#

LossB"?

Adam/Loss/rawW��>#�*1       ����	���F~��A�*#

Loss��?

Adam/Loss/raw�Έ@�F��1       ����	��F~��A�*#

Loss�w?

Adam/Loss/rawζ�>�ݸ�1       ����	���F~��A�*#

Loss��f?

Adam/Loss/rawg}>2T_1       ����	��F~��A�*#

Loss2�S?

Adam/Loss/raw@D>�c�1       ����	�@�F~��A�*#

Loss{;B?

Adam/Loss/raw>�O>�G�Q1       ����	��F~��A�*#

Loss� 4?

Adam/Loss/raw9I>-�7�1       ����	8P�F~��A�*#

Loss�'?

Adam/Loss/raw5b�>4���1       ����		��F~��A�*#

Loss˿?

Adam/Loss/raw��S>�r)�1       ����	_�F~��A�*#

LossC+?

Adam/Loss/raw�ǈ>�rU1       ����	��F~��A�*#

Loss�0?

Adam/Loss/raw�֞>�@��1       ����	�F~��A�*#

Loss*?

Adam/Loss/raw�o|@�)�p1       ����	�f�F~��A�*#

Loss]�]?

Adam/Loss/raw*:E>�L�g1       ����	)�F~��A�*#

Loss�pL?

Adam/Loss/raw?R>����1       ����	P��F~��A�*#

Loss�@=?

Adam/Loss/raw��>%C&�1       ����	�g�F~��A�*#

Losse�-?

Adam/Loss/raw̴k>N�P�1       ����	�Z�F~��A�*#

Loss1"?

Adam/Loss/raw-��>���1       ����	�e�F~��A�*#

Loss��?

Adam/Loss/raw���>�2K�1       ����	|��F~��A�*#

Loss��?

Adam/Loss/raw���>��e%1       ����	u�F~��A�*#

LossR�	?

Adam/Loss/raw�G>�ل�1       ����	5��F~��A�*#

Loss`��>

Adam/Loss/rawϓ�>O"��1       ����	�>�F~��A�*#

Loss�>

Adam/Loss/raw�T�@(��L1       ����	�|�F~��A�*#

Loss6�{?

Adam/Loss/rawS��=���1       ����	���F~��A�*#

Lossud?

Adam/Loss/raw���=k�t1       ����	Ǟ�F~��A�*#

Loss�P?

Adam/Loss/rawb�=RF�1       ����	�f G~��A�*#

LossH�=?

Adam/Loss/raw|�>u��1       ����	��G~��A�*#

Loss�7.?

Adam/Loss/rawPTQ>� 4X1       ����	�dG~��A�*#

LossV"?

Adam/Loss/raw9�[>P=��1       ����	�PG~��A�*#

LossbR?

Adam/Loss/raw�p>���1       ����	��G~��A�*#

Loss4?

Adam/Loss/raw�%l>uY=I1       ����	cI$G~��A�*#

Loss��?

Adam/Loss/raw�R>���M1       ����	0.G~��A�*#

Loss���>

Adam/Loss/raw�^@%�p1       ����	�l7G~��A�*#

Loss��H?

Adam/Loss/rawv�H>���N1       ����	}�;G~��A�*#

Loss��9?

Adam/Loss/rawJ��=�S�1       ����	�U>G~��A�*#

LossRN*?

Adam/Loss/raw��=蓐�1       ����	hIG~��A�*#

Loss=?

Adam/Loss/raw\�=@с1       ����	n�RG~��A�*#

Loss�?

Adam/Loss/rawěw>��:1       ����	6WYG~��A�*#

Lossy�?

Adam/Loss/raw�ڞ>�d��1       ����	�y`G~��A�*#

Lossxb?

Adam/Loss/raw�`>�`'d1       ����	��gG~��A�*#

Lossf�>

Adam/Loss/raw�>T81       ����	DioG~��A�*#

Loss���>

Adam/Loss/raw��>��1       ����	�vG~��A�*#

Loss��>

Adam/Loss/raw�5�@ ��1       ����	�}|G~��A�*#

Lossj�_?

Adam/Loss/raw��n>�U�K1       ����	�iG~��A�*#

Loss��O?

Adam/Loss/raw�W1>��I�1       ����	؁G~��A�*#

Losst0??

Adam/Loss/rawo��=�uro1       ����	�R�G~��A�*#

Loss��.?

Adam/Loss/raw��>�Ԍy1       ����	=��G~��A�*#

Loss4�$?

Adam/Loss/raw�_>7��1       ����	�c�G~��A�*#

Loss��?

Adam/Loss/raw�C}>ZA�u1       ����	rU�G~��A�*#

Loss��?

Adam/Loss/raw7�t>��A�1       ����	��G~��A�*#

Loss�s?

Adam/Loss/raw�HX>X��j1       ����	Y�G~��A�*#

Loss�6 ?

Adam/Loss/raw�І>�E��1       ����	J��G~��A�*#

Loss�C�>

Adam/Loss/raw>�K@'o1       ����	W�G~��A�*#

Loss�T??

Adam/Loss/raw�7G>���d1       ����	�p�G~��A�*#

Loss�-1?

Adam/Loss/raw��>O繽1       ����	���G~��A�*#

Loss'�&?

Adam/Loss/raw�=h>Mw�1       ����	
��G~��A�*#

Loss�?

Adam/Loss/rawf�>���\1       ����	�_�G~��A�*#

Loss�?

Adam/Loss/raw��>S�1       ����	��G~��A�*#

LossF�	?

Adam/Loss/raw<��>��1       ����	�F�G~��A�*#

Loss��?

Adam/Loss/raw��3>��]D1       ����	}��G~��A�*#

Loss1��>

Adam/Loss/raw��>��1       ����	��G~��A�*#

Loss���>

Adam/Loss/raw/�>��0e1       ����	�U�G~��A�*#

Loss���>

Adam/Loss/rawEɔ@��k?1       ����	ܵ�G~��A�*#

LossE�V?

Adam/Loss/rawq��> 7$c1       ����	���G~��A�*#

Loss��G?

Adam/Loss/raw�3�>>���1       ����	1��G~��A�*#

Loss۔:?

Adam/Loss/raw�E�=7*~k1       ����	  �G~��A�*#

Loss
+?

Adam/Loss/raw���>�P)1       ����	C H~��A�*#

LossS�"?

Adam/Loss/raw���>��b^1       ����	�XH~��A�*#

Loss�Y?

Adam/Loss/raw#a>�D��1       ����	z�H~��A�*#

LossF�?

Adam/Loss/raw�t�=����1       ����	d�H~��A�*#

Loss5%?

Adam/Loss/raw!�\>��V�1       ����	n H~��A�*#

Loss��>

Adam/Loss/raw=O>��=K1       ����	��&H~��A�*#

Loss'��>

Adam/Loss/rawD�@��-�1       ����	��-H~��A�*#

Loss~�t?

Adam/Loss/raw��P>f�o�1       ����	��/H~��A�*#

Loss�qa?

Adam/Loss/raw��>��U�1       ����	P�1H~��A�*#

LossƉR?

Adam/Loss/raw~��=�4��1       ����	NG9H~��A�*#

Lossͮ@?

Adam/Loss/rawn@>5%�1       ����	��?H~��A�*#

Loss�72?

Adam/Loss/raw��_>4ց:1       ����	�GH~��A�*#

Loss4�%?

Adam/Loss/raw�54>����1       ����	�uMH~��A�*#

LossR�?

Adam/Loss/rawe�>@�>1       ����	�,TH~��A�*#

Loss�d?

Adam/Loss/raw�I%>J��1       ����	&�ZH~��A�*#

LossxI?

Adam/Loss/rawx�>|�^1       ����	�jaH~��A�*#

Loss�>

Adam/Loss/raww�x@`�c1       ����	�hH~��A�*#

Loss7R?

Adam/Loss/raw:G>~^�?1       ����	DjH~��A�*#

LossR.B?

Adam/Loss/raw �G>��g�1       ����	��kH~��A�*#

Loss��3?

Adam/Loss/raw���=�w�01       ����	R{rH~��A�*#

Loss\�$?

Adam/Loss/rawɗ>c��91       ����	��xH~��A�*#

Loss��?

Adam/Loss/raw�>���"1       ����	��~H~��A�*#

Lossb[?

Adam/Loss/raw�a=>��Du1       ����	]�H~��A�*#

LossJ?

Adam/Loss/raw%Y�=V`�%1       ����	&p�H~��A�*#

Loss�5�>

Adam/Loss/raw�M>�p��1       ����	?��H~��A�*#

LossVD�>

Adam/Loss/raw*�>���!1       ����	�ԙH~��A�*#

Loss���>

Adam/Loss/raw�k@�e��1       ����	)W�H~��A�*#

Loss�f=?

Adam/Loss/raw�c�>����1       ����	�<�H~��A�*#

Lossʇ2?

Adam/Loss/raw�>���1       ����	��H~��A�*#

Lossj_'?

Adam/Loss/raw'�D>��1       ����	^��H~��A�*#

Loss��?

Adam/Loss/rawlg>��Rt1       ����	 p�H~��A�*#

Lossv�?

Adam/Loss/rawp��=|z�1       ����	���H~��A�*#

Loss�3?

Adam/Loss/rawl��=M���1       ����	a޽H~��A�*#

Lossޏ�>

Adam/Loss/raw�L�=y4P�1       ����	�Y�H~��A�*#

Loss̩�>

Adam/Loss/raw�>G;��1       ����	(�H~��A�*#

Loss���>

Adam/Loss/raw[��=:̌1       ����	�z�H~��A�*#

Losse��>

Adam/Loss/raw�Ӝ@LM1       ����	���H~��A�*#

LosstV?

Adam/Loss/raw��>�!1       ����	#��H~��A�*#

Loss)�D?

Adam/Loss/raw��>��f1       ����	m�H~��A�*#

Lossw�4?

Adam/Loss/raw:��=�g�1       ����	n�H~��A�*#

Loss,4%?

Adam/Loss/raw�B�>�n;C1       ����	i��H~��A�*#

LossHr?

Adam/Loss/raw�z�=E��1       ����	�S�H~��A�*#

Loss�+?

Adam/Loss/rawΗ�=7q1       ����	�H~��A�*#

Loss!�?

Adam/Loss/raw��=,�wW1       ����	$��H~��A�*#

Loss��>

Adam/Loss/raw>��#1       ����	�jI~��A�*#

Loss��>

Adam/Loss/raw�>	�m�1       ����	%�
I~��A�*#

Loss+��>

Adam/Loss/raw�@j��-1       ����	�[I~��A�*#

Loss��Y?

Adam/Loss/rawn>���1       ����	5I~��A�*#

LossG?

Adam/Loss/rawY�*>��<�1       ����	2SI~��A�*#

Lossdt7?

Adam/Loss/raw^��=H��1       ����	�:I~��A�*#

Loss�!(?

Adam/Loss/rawS�>"��1       ����	�}"I~��A�*#

LossS�?

Adam/Loss/raw��=Y�Qy1       ����	xF)I~��A�*#

Lossa?

Adam/Loss/raw��I=�H��1       ����	�/I~��A�*#

Loss�?

Adam/Loss/raw�I�=60�1       ����	B6I~��A�*#

Loss�S�>

Adam/Loss/rawAS;=nnXY1       ����	ْ<I~��A�*#

Loss^	�>

Adam/Loss/raw��>� c1       ����	�`CI~��A�*#

Loss�9�>

Adam/Loss/raw�l�@)�>51       ����	��II~��A�*#

Loss(�V?

Adam/Loss/rawLC>6.�]1       ����	��KI~��A�*#

Loss�D?

Adam/Loss/raw�=v=�:�51       ����	��MI~��A�*#

Loss��2?

Adam/Loss/raw�p�<�H�1       ����	��TI~��A�*#

Loss��!?

Adam/Loss/raw��=Ձ��1       ����	�l[I~��A�*#

Loss!P?

Adam/Loss/raw���=��P�1       ����	O>bI~��A�*#

Loss`'?

Adam/Loss/raw���=�*�1       ����	�hI~��A�*#

Loss�x�>

Adam/Loss/raw���=A	1       ����	��nI~��A�*#

Loss~��>

Adam/Loss/raw��~=m�	�1       ����	�iuI~��A�*#

Loss���>

Adam/Loss/raw^O>�؈�1       ����	��{I~��A�*#

LossM2�>

Adam/Loss/raw�Ȼ@i�f1       ����		O�I~��A�*#

LossJ�n?

Adam/Loss/raw�7�=��*1       ����	6?�I~��A�*#

Loss�X?

Adam/Loss/raw��$=hJ1       ����	��I~��A�*#

LossXD?

Adam/Loss/raw�j-=O�:�1       ����	�
�I~��A�*#

Loss��1?

Adam/Loss/raw��d=��%1       ����	�ړI~��A�*#

Losst=!?

Adam/Loss/raw쇒=v�1       ����	XS�I~��A�*#

Loss��?

Adam/Loss/raw �V=f׸1       ����	[�I~��A�*#

Loss%�?

Adam/Loss/raw8'�=�cM|1       ����	CV�I~��A�*#

Loss
F�>

Adam/Loss/raw��>f11       ����	J�I~��A�*#

Loss!��>

Adam/Loss/raw2��=;lO�1       ����	ro�I~��A�*#

Loss�U�>

Adam/Loss/raw��@��u1       ����	�4�I~��A�*#

LossCO?

Adam/Loss/raw��=�뷗1       ����	;3�I~��A�*#

Loss<_;?

Adam/Loss/raw�[s=��;[1       ����	��I~��A�*#

Loss�'*?

Adam/Loss/raw�l=����1       ����	�N�I~��A�*#

Loss,?

Adam/Loss/raw_��=��<1       ����	���I~��A�*#

Loss�?

Adam/Loss/raw��	>.��1       ����	.�I~��A�*#

Loss��?

Adam/Loss/raw�p{=D*!�1       ����	si�I~��A�*#

LossQ��>

Adam/Loss/raw]�=�� 1       ����	���I~��A�*#

LossO�>

Adam/Loss/rawn
>}���1       ����	�q�I~��A�*#

Loss{��>

Adam/Loss/rawK=(=M<��1       ����	\��I~��A�*#

Loss2��>

Adam/Loss/raw��@����1       ����	�0�I~��A�*#

Loss�j?

Adam/Loss/raw��R>�^��1       ����	��I~��A�*#

Loss�X?

Adam/Loss/rawk��=#�^�1       ����	���I~��A�*#

Loss�F?

Adam/Loss/raw_�p=JH�f1       ����	S�J~��A�*#

Loss�3?

Adam/Loss/raw��=Ԧ��1       ����	�%J~��A�*#

Loss��"?

Adam/Loss/raw�C�=���(1       ����	IEJ~��A�*#

Losst�?

Adam/Loss/raw_��=��>k1       ����	J�J~��A�*#

Loss�?

Adam/Loss/raw� >��1       ����	�J~��A�*#

Loss�8�>

Adam/Loss/rawц�=�b�1       ����	�%J~��A�*#

Loss��>

Adam/Loss/raw��G=m�Sh1       ����	:<+J~��A�*#

Loss��>

Adam/Loss/raw64�@��1       ����	h2J~��A�*#

Loss�K?

Adam/Loss/raw'�P=�?��1       ����	R�3J~��A�*#

Loss��8?

Adam/Loss/rawc��<?��1       ����	.�5J~��A�*#

LossA�&?

Adam/Loss/raw�u=�T��1       ����	��<J~��A�*#

Loss��?

Adam/Loss/raw} =4N��1       ����	R]CJ~��A�*#

Loss��?

Adam/Loss/raw�_�=�1       ����	�IJ~��A�*#

Loss!��>

Adam/Loss/raw�'>i�K�1       ����	)�OJ~��A�*#

Loss���>

Adam/Loss/raw�3�=�#W1       ����	F\VJ~��A�*#

Loss��>

Adam/Loss/raw+�,=��
1       ����	Z�\J~��A�*#

Loss��>

Adam/Loss/raw��=6�Q�1       ����	T;cJ~��A�*#

Loss8�>

Adam/Loss/raw6��@�#��1       ����	a�iJ~��A�*#

Lossf�G?

Adam/Loss/raw���=S��1       ����	��kJ~��A�*#

Loss!	6?

Adam/Loss/raw���<��6�1       ����	�mJ~��A�*#

Loss�O$?

Adam/Loss/raw]�=<O� �1       ����	WztJ~��A�*#

LossT-?

Adam/Loss/rawjS
=��B1       ����	E�zJ~��A�*#

LossQ9?

Adam/Loss/raw� =���1       ����	�Z�J~��A�*#

Loss[6�>

Adam/Loss/raw��=8��1       ����	>ȇJ~��A�*#

Loss�$�>

Adam/Loss/raw��=��S�1       ����	W$�J~��A�*#

Lossl��>

Adam/Loss/raw�a=c��1       ����	&8�J~��A�*#

LossJ�>

Adam/Loss/raw8J�='��1       ����	�
�J~��A�*#

Loss�>

Adam/Loss/raw+��@}&1       ����	ᚢJ~��A�*#

Loss�\^?

Adam/Loss/raw �k=��91       ����	`��J~��A�*#

Loss��I?

Adam/Loss/rawi�=����1       ����	T�J~��A�*#

Lossb6?

Adam/Loss/rawwW�<W���1       ����	�'�J~��A�*#

Loss��$?

Adam/Loss/raw�� >����1       ����	a�J~��A�*#

Loss{l?

Adam/Loss/raw�`=M�1       ����	�̹J~��A�*#

Loss{�	?

Adam/Loss/raw�o�=��1       ����	,�J~��A�*#

Loss���>

Adam/Loss/rawвM=��P�1       ����	{�J~��A�*#

Loss���>

Adam/Loss/rawk|3=�	�1       ����	�U�J~��A�*#

Loss��>

Adam/Loss/raw�H=�{m�1       ����	��J~��A�*#

Loss���>

Adam/Loss/raw*Z�@�Qv1       ����	:��J~��A�*#

Loss^G?

Adam/Loss/raw�:9>%f�1       ����	32�J~��A�*#

Loss�8?

Adam/Loss/rawY�=�+$W1       ����	�F�J~��A�*#

Loss�&?

Adam/Loss/raw+�<�,�1       ����	E�J~��A�*#

Loss�!?

Adam/Loss/raw"��=p�n1       ����	��J~��A�*#

Loss�P	?

Adam/Loss/raw<c�=��WV1       ����	���J~��A�*#

Loss7��>

Adam/Loss/raw*�<���1       ����	��J~��A�*#

Loss��>

Adam/Loss/rawd)�<��l1       ����	�oK~��A�*#

Loss���>

Adam/Loss/raw�c,=��2�1       ����	</
K~��A�*#

Loss�Y�>

Adam/Loss/raw�Z�={���1       ����	�AK~��A�*#

Loss؟�>

Adam/Loss/raw�@
��M1       ����	j�K~��A�*#

Loss� G?

Adam/Loss/rawB߉=����1       ����	F�K~��A�*#

Loss~�4?

Adam/Loss/rawv�=�N�)1       ����	�wK~��A�*#

Loss/�#?

Adam/Loss/raw~�<S��1       ����	��"K~��A�*#

Loss��?

Adam/Loss/raw��=$�S�1       ����	@�(K~��A�*#

Loss	/?

Adam/Loss/raw��=����1       ����	�T0K~��A�*#

Loss�L�>

Adam/Loss/rawJ��=��'(1       ����	N7K~��A�*#

Loss�h�>

Adam/Loss/rawQ��=Z��x1       ����	��=K~��A�*#

Loss���>

Adam/Loss/raw��=���1       ����	X EK~��A�*#

Loss�>

Adam/Loss/raw��S=���1       ����	�lKK~��A�*#

LossK0�>

Adam/Loss/raw��@�$1�1       ����	��QK~��A�*#

Loss�WC?

Adam/Loss/raw�5�=��Ͽ1       ����	�SK~��A�*#

LossW�1?

Adam/Loss/raw-=Q�}1       ����	 �UK~��A�*#

Lossͪ ?

Adam/Loss/raw��=V5k�1       ����	f,\K~��A�*#

Loss�z?

Adam/Loss/rawԤ3=�_��1       ����	a�bK~��A�*#

Loss�?

Adam/Loss/rawYG4=j�X�1       ����	!�iK~��A�*#

Loss���>

Adam/Loss/raw� =��|�1       ����	�2pK~��A�*#

Lossd��>

Adam/Loss/raw��=���J1       ����	(avK~��A�*#

Lossܺ�>

Adam/Loss/raw�F�=m6��1       ����	ƿ|K~��A�*#

Loss�ɵ>

Adam/Loss/rawh7=|}�1       ����	�ʃK~��A�*#

Loss�I�>

Adam/Loss/raw*�@=�ܔ1       ����	��K~��A�*#

Lossz�@?

Adam/Loss/rawhE�<]�̚1       ����	\��K~��A�*#

Loss�.?

Adam/Loss/raw�ڏ<��v�1       ����	39�K~��A�*#

Lossv?

Adam/Loss/raw�v�<���