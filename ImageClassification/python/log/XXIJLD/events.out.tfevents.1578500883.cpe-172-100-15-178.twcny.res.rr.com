       �K"	  �Ā��Abrain.Event:2�lu��     �c�	7��Ā��A"��

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
1Conv2D/W/Initializer/random_uniform/RandomUniformRandomUniform)Conv2D/W/Initializer/random_uniform/shape*
T0*
_class
loc:@Conv2D/W*
seed2 *
dtype0*&
_output_shapes
: *

seed 
�
'Conv2D/W/Initializer/random_uniform/subSub'Conv2D/W/Initializer/random_uniform/max'Conv2D/W/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
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
Conv2D/W/readIdentityConv2D/W*&
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
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
Conv2D/Conv2DConv2Dinput/XConv2D/W/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*/
_output_shapes
:���������22 *
	dilations
*
T0
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
MaxPool2D/MaxPoolMaxPoolConv2D/Relu*
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
 
�
+Conv2D_1/W/Initializer/random_uniform/shapeConst*%
valueB"          @   *
_class
loc:@Conv2D_1/W*
dtype0*
_output_shapes
:
�
)Conv2D_1/W/Initializer/random_uniform/minConst*
valueB
 *��z�*
_class
loc:@Conv2D_1/W*
dtype0*
_output_shapes
: 
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

seed *
T0*
_class
loc:@Conv2D_1/W*
seed2 *
dtype0*&
_output_shapes
: @
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
Conv2D_1/W*&
_output_shapes
: @*
T0*
_class
loc:@Conv2D_1/W
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
Conv2D_1/b*
_output_shapes
:@*
T0*
_class
loc:@Conv2D_1/b
�
Conv2D_1/Conv2DConv2DMaxPool2D/MaxPoolConv2D_1/W/read*
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
dtype0*
_output_shapes
:*%
valueB"      @   �   *
_class
loc:@Conv2D_2/W
�
)Conv2D_2/W/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�\1�*
_class
loc:@Conv2D_2/W
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
Conv2D_2/W%Conv2D_2/W/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
x
Conv2D_2/W/readIdentity
Conv2D_2/W*
T0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
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
Conv2D_2/bConv2D_2/b/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
l
Conv2D_2/b/readIdentity
Conv2D_2/b*
T0*
_class
loc:@Conv2D_2/b*
_output_shapes	
:�
�
Conv2D_2/Conv2DConv2DMaxPool2D_1/MaxPoolConv2D_2/W/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
�
Conv2D_2/BiasAddBiasAddConv2D_2/Conv2DConv2D_2/b/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
b
Conv2D_2/ReluReluConv2D_2/BiasAdd*0
_output_shapes
:����������*
T0
�
MaxPool2D_2/MaxPoolMaxPoolConv2D_2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0
�
+Conv2D_3/W/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      �   @   *
_class
loc:@Conv2D_3/W
�
)Conv2D_3/W/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *����*
_class
loc:@Conv2D_3/W
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
Conv2D_3/W/AssignAssign
Conv2D_3/W%Conv2D_3/W/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
x
Conv2D_3/W/readIdentity
Conv2D_3/W*'
_output_shapes
:�@*
T0*
_class
loc:@Conv2D_3/W
�
Conv2D_3/b/Initializer/ConstConst*
valueB@*    *
_class
loc:@Conv2D_3/b*
dtype0*
_output_shapes
:@
�

Conv2D_3/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_3/b*
	container 
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
Conv2D_3/BiasAddBiasAddConv2D_3/Conv2DConv2D_3/b/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
a
Conv2D_3/ReluReluConv2D_3/BiasAdd*/
_output_shapes
:���������@*
T0
�
MaxPool2D_3/MaxPoolMaxPoolConv2D_3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
�
+Conv2D_4/W/Initializer/random_uniform/shapeConst*%
valueB"      @       *
_class
loc:@Conv2D_4/W*
dtype0*
_output_shapes
:
�
)Conv2D_4/W/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�\1�*
_class
loc:@Conv2D_4/W
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
)Conv2D_4/W/Initializer/random_uniform/subSub)Conv2D_4/W/Initializer/random_uniform/max)Conv2D_4/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_4/W*
_output_shapes
: 
�
)Conv2D_4/W/Initializer/random_uniform/mulMul3Conv2D_4/W/Initializer/random_uniform/RandomUniform)Conv2D_4/W/Initializer/random_uniform/sub*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�
%Conv2D_4/W/Initializer/random_uniformAdd)Conv2D_4/W/Initializer/random_uniform/mul)Conv2D_4/W/Initializer/random_uniform/min*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�

Conv2D_4/W
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
_class
loc:@Conv2D_4/b*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
Conv2D_4/Conv2DConv2DMaxPool2D_3/MaxPoolConv2D_4/W/read*
T0*
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

�
Conv2D_4/BiasAddBiasAddConv2D_4/Conv2DConv2D_4/b/read*
T0*
data_formatNHWC*/
_output_shapes
:��������� 
a
Conv2D_4/ReluReluConv2D_4/BiasAdd*/
_output_shapes
:��������� *
T0
�
MaxPool2D_4/MaxPoolMaxPoolConv2D_4/Relu*
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
3FullyConnected/W/Initializer/truncated_normal/shapeConst*
valueB"       *#
_class
loc:@FullyConnected/W*
dtype0*
_output_shapes
:
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
4FullyConnected/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*#
_class
loc:@FullyConnected/W*
dtype0*
_output_shapes
: 
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
-FullyConnected/W/Initializer/truncated_normalAdd1FullyConnected/W/Initializer/truncated_normal/mul2FullyConnected/W/Initializer/truncated_normal/mean*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
FullyConnected/W
VariableV2*
shape:	 �*
dtype0*
_output_shapes
:	 �*
shared_name *#
_class
loc:@FullyConnected/W*
	container 
�
FullyConnected/W/AssignAssignFullyConnected/W-FullyConnected/W/Initializer/truncated_normal*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
FullyConnected/W/readIdentityFullyConnected/W*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
"FullyConnected/b/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    *#
_class
loc:@FullyConnected/b
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
FullyConnected/b/AssignAssignFullyConnected/b"FullyConnected/b/Initializer/Const*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
~
FullyConnected/b/readIdentityFullyConnected/b*
T0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
m
FullyConnected/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����    
�
FullyConnected/ReshapeReshapeMaxPool2D_4/MaxPoolFullyConnected/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:��������� 
�
FullyConnected/MatMulMatMulFullyConnected/ReshapeFullyConnected/W/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
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
is_training/readIdentityis_training*
T0
*
_class
loc:@is_training*
_output_shapes
: 
V
Dropout/Assign/valueConst*
dtype0
*
_output_shapes
: *
value	B
 Z
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
Dropout/Assign_1Assignis_trainingDropout/Assign_1/value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
*
_class
loc:@is_training
_
Dropout/cond/SwitchSwitchis_trainingis_training/read*
_output_shapes
: : *
T0

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
Dropout/cond/pred_idIdentityis_training/read*
T0
*
_output_shapes
: 
v
Dropout/cond/dropout/rateConst^Dropout/cond/switch_t*
valueB
 *��L>*
dtype0*
_output_shapes
: 
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
'Dropout/cond/dropout/random_uniform/minConst^Dropout/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
'Dropout/cond/dropout/random_uniform/maxConst^Dropout/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
1Dropout/cond/dropout/random_uniform/RandomUniformRandomUniformDropout/cond/dropout/Shape*
T0*
dtype0*(
_output_shapes
:����������*
seed2 *

seed 
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
!Dropout/cond/dropout/GreaterEqualGreaterEqual#Dropout/cond/dropout/random_uniformDropout/cond/dropout/rate*(
_output_shapes
:����������*
T0
�
Dropout/cond/dropout/mulMul#Dropout/cond/dropout/Shape/Switch:1Dropout/cond/dropout/truediv*
T0*(
_output_shapes
:����������
�
Dropout/cond/dropout/CastCast!Dropout/cond/dropout/GreaterEqual*
Truncate( *(
_output_shapes
:����������*

DstT0*

SrcT0

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
N**
_output_shapes
:����������: *
T0
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
$FullyConnected_1/b/Initializer/ConstConst*
dtype0*
_output_shapes
:*
valueB*    *%
_class
loc:@FullyConnected_1/b
�
FullyConnected_1/b
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@FullyConnected_1/b*
	container 
�
FullyConnected_1/b/AssignAssignFullyConnected_1/b$FullyConnected_1/b/Initializer/Const*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
FullyConnected_1/b/readIdentityFullyConnected_1/b*
T0*%
_class
loc:@FullyConnected_1/b*
_output_shapes
:
�
FullyConnected_1/MatMulMatMulDropout/cond/MergeFullyConnected_1/W/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
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
Accuracy/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
�
Accuracy/ArgMax_1ArgMax	targets/YAccuracy/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
�
Accuracy/EqualEqualAccuracy/ArgMaxAccuracy/ArgMax_1*
incompatible_shape_error(*
T0	*#
_output_shapes
:���������
r
Accuracy/CastCastAccuracy/Equal*
Truncate( *#
_output_shapes
:���������*

DstT0*

SrcT0

X
Accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
r
Accuracy/MeanMeanAccuracy/CastAccuracy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
d
"Crossentropy/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
Crossentropy/SumSumFullyConnected_1/Softmax"Crossentropy/Sum/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
}
Crossentropy/truedivRealDivFullyConnected_1/SoftmaxCrossentropy/Sum*'
_output_shapes
:���������*
T0
X
Crossentropy/Cast/xConst*
valueB
 *���.*
dtype0*
_output_shapes
: 
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
$Crossentropy/Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
Crossentropy/Sum_1SumCrossentropy/mul$Crossentropy/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
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
Training_step/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
Training_step
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Training_step/AssignAssignTraining_stepTraining_step/initial_value*
T0* 
_class
loc:@Training_step*
validate_shape(*
_output_shapes
: *
use_locking(
p
Training_step/readIdentityTraining_step*
_output_shapes
: *
T0* 
_class
loc:@Training_step
^
Global_Step/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
Global_Step
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Global_Step/AssignAssignGlobal_StepGlobal_Step/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Global_Step
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
AssignAssignGlobal_StepAdd*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: *
use_locking(
[
val_loss/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
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
val_loss/AssignAssignval_lossval_loss/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@val_loss
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
dtype0*
_output_shapes
:*
shape:
X
placeholder/val_accPlaceholder*
dtype0*
_output_shapes
:*
shape:
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
assign/val_accAssignval_accplaceholder/val_acc*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: *
use_locking(
�
*Accuracy/Mean/moving_avg/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    *+
_class!
loc:@Accuracy/Mean/moving_avg
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
Accuracy/Mean/moving_avg/AssignAssignAccuracy/Mean/moving_avg*Accuracy/Mean/moving_avg/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg
�
Accuracy/Mean/moving_avg/readIdentityAccuracy/Mean/moving_avg*
_output_shapes
: *
T0*+
_class!
loc:@Accuracy/Mean/moving_avg
U
moving_avg/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
moving_avg/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
moving_avg/AssignMovingAvg	AssignSubAccuracy/Mean/moving_avgmoving_avg/AssignMovingAvg/mul*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
_output_shapes
: *
use_locking( 
/

moving_avgNoOp^moving_avg/AssignMovingAvg
O
Adam/Total_LossIdentityCrossentropy/Mean*
_output_shapes
: *
T0
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
#Crossentropy/Mean/moving_avg/AssignAssignCrossentropy/Mean/moving_avg.Crossentropy/Mean/moving_avg/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg
�
!Crossentropy/Mean/moving_avg/readIdentityCrossentropy/Mean/moving_avg*
_output_shapes
: *
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg
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
Adam/moving_avg/addAddV2Adam/moving_avg/add/xTraining_step/read*
T0*
_output_shapes
: 
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
%Adam/moving_avg/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
#Adam/moving_avg/AssignMovingAvg/subSub%Adam/moving_avg/AssignMovingAvg/sub/xAdam/moving_avg/Minimum*
_output_shapes
: *
T0
�
%Adam/moving_avg/AssignMovingAvg/sub_1Sub!Crossentropy/Mean/moving_avg/readCrossentropy/Mean*
T0*
_output_shapes
: 
�
#Adam/moving_avg/AssignMovingAvg/mulMul%Adam/moving_avg/AssignMovingAvg/sub_1#Adam/moving_avg/AssignMovingAvg/sub*
T0*
_output_shapes
: 
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
Adam/gradients/FillFillAdam/gradients/ShapeAdam/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
�
3Adam/gradients/Crossentropy/Mean_grad/Reshape/shapeConst^Adam/moving_avg^moving_avg*
valueB:*
dtype0*
_output_shapes
:
�
-Adam/gradients/Crossentropy/Mean_grad/ReshapeReshapeAdam/gradients/Fill3Adam/gradients/Crossentropy/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
+Adam/gradients/Crossentropy/Mean_grad/ShapeShapeCrossentropy/Neg^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
*Adam/gradients/Crossentropy/Mean_grad/TileTile-Adam/gradients/Crossentropy/Mean_grad/Reshape+Adam/gradients/Crossentropy/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
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
*Adam/gradients/Crossentropy/Mean_grad/ProdProd-Adam/gradients/Crossentropy/Mean_grad/Shape_1+Adam/gradients/Crossentropy/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
-Adam/gradients/Crossentropy/Mean_grad/Const_1Const^Adam/moving_avg^moving_avg*
valueB: *
dtype0*
_output_shapes
:
�
,Adam/gradients/Crossentropy/Mean_grad/Prod_1Prod-Adam/gradients/Crossentropy/Mean_grad/Shape_2-Adam/gradients/Crossentropy/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
/Adam/gradients/Crossentropy/Mean_grad/Maximum/yConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
value	B :
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
2Adam/gradients/Crossentropy/Sum_1_grad/range/startConst^Adam/moving_avg^moving_avg*
value	B : *?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
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
0Adam/gradients/Crossentropy/Sum_1_grad/Maximum/yConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
value	B :*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape
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
(Adam/gradients/Crossentropy/mul_grad/SumSum(Adam/gradients/Crossentropy/mul_grad/Mul:Adam/gradients/Crossentropy/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
*Adam/gradients/Crossentropy/mul_grad/Sum_1Sum*Adam/gradients/Crossentropy/mul_grad/Mul_1<Adam/gradients/Crossentropy/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
.Adam/gradients/Crossentropy/mul_grad/Reshape_1Reshape*Adam/gradients/Crossentropy/mul_grad/Sum_1,Adam/gradients/Crossentropy/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
4Adam/gradients/Crossentropy/clip_by_value_grad/ShapeShape"Crossentropy/clip_by_value/Minimum^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_1Const^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB 
�
6Adam/gradients/Crossentropy/clip_by_value_grad/Shape_2Shape(Adam/gradients/Crossentropy/Log_grad/mul*
_output_shapes
:*
T0*
out_type0
�
:Adam/gradients/Crossentropy/clip_by_value_grad/zeros/ConstConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB
 *    
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
5Adam/gradients/Crossentropy/clip_by_value_grad/SelectSelect;Adam/gradients/Crossentropy/clip_by_value_grad/GreaterEqual(Adam/gradients/Crossentropy/Log_grad/mul4Adam/gradients/Crossentropy/clip_by_value_grad/zeros*'
_output_shapes
:���������*
T0
�
2Adam/gradients/Crossentropy/clip_by_value_grad/SumSum5Adam/gradients/Crossentropy/clip_by_value_grad/SelectDAdam/gradients/Crossentropy/clip_by_value_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0
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
<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/zerosFill>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_2BAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:���������*
T0*

index_type0
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
:Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SumSum=Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SelectLAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/ReshapeReshape:Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Sum<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
0Adam/gradients/Crossentropy/truediv_grad/Shape_1ShapeCrossentropy/Sum^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
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
,Adam/gradients/Crossentropy/truediv_grad/SumSum0Adam/gradients/Crossentropy/truediv_grad/RealDiv>Adam/gradients/Crossentropy/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
0Adam/gradients/Crossentropy/truediv_grad/ReshapeReshape,Adam/gradients/Crossentropy/truediv_grad/Sum.Adam/gradients/Crossentropy/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
,Adam/gradients/Crossentropy/truediv_grad/NegNegFullyConnected_1/Softmax^Adam/moving_avg^moving_avg*'
_output_shapes
:���������*
T0
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
*Adam/gradients/Crossentropy/Sum_grad/ShapeShapeFullyConnected_1/Softmax^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
)Adam/gradients/Crossentropy/Sum_grad/SizeConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
value	B :*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape
�
(Adam/gradients/Crossentropy/Sum_grad/addAddV2"Crossentropy/Sum/reduction_indices)Adam/gradients/Crossentropy/Sum_grad/Size*
_output_shapes
: *
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape
�
(Adam/gradients/Crossentropy/Sum_grad/modFloorMod(Adam/gradients/Crossentropy/Sum_grad/add)Adam/gradients/Crossentropy/Sum_grad/Size*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
_output_shapes
: 
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
0Adam/gradients/Crossentropy/Sum_grad/range/deltaConst^Adam/moving_avg^moving_avg*
value	B :*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
dtype0*
_output_shapes
: 
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
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
N*
_output_shapes
:
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
)Adam/gradients/Crossentropy/Sum_grad/TileTile,Adam/gradients/Crossentropy/Sum_grad/Reshape-Adam/gradients/Crossentropy/Sum_grad/floordiv*
T0*'
_output_shapes
:���������*

Tmultiples0
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
BAdam/gradients/FullyConnected_1/Softmax_grad/Sum/reduction_indicesConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB :
���������
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
2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1Mul0Adam/gradients/FullyConnected_1/Softmax_grad/subFullyConnected_1/Softmax*'
_output_shapes
:���������*
T0
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
4Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1MatMulDropout/cond/Merge2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
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
2Adam/gradients/Dropout/cond/dropout/mul_1_grad/MulMul2Adam/gradients/Dropout/cond/Merge_grad/cond_grad:1Dropout/cond/dropout/Cast*(
_output_shapes
:����������*
T0
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
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Mul_1MulDropout/cond/dropout/mul2Adam/gradients/Dropout/cond/Merge_grad/cond_grad:1*
T0*(
_output_shapes
:����������
�
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum_1Sum4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Mul_1FAdam/gradients/Dropout/cond/dropout/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
8Adam/gradients/Dropout/cond/dropout/mul_1_grad/Reshape_1Reshape4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum_16Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/ShapeShape#Dropout/cond/dropout/Shape/Switch:1^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
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
:*
	keep_dims( *

Tidx0
�
4Adam/gradients/Dropout/cond/dropout/mul_grad/ReshapeReshape0Adam/gradients/Dropout/cond/dropout/mul_grad/Sum2Adam/gradients/Dropout/cond/dropout/mul_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/Mul_1Mul#Dropout/cond/dropout/Shape/Switch:16Adam/gradients/Dropout/cond/dropout/mul_1_grad/Reshape*
T0*(
_output_shapes
:����������
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/Sum_1Sum2Adam/gradients/Dropout/cond/dropout/mul_grad/Mul_1DAdam/gradients/Dropout/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Adam/gradients/zeros_1/ConstConst^Adam/gradients/Identity_1^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB
 *    
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
0Adam/gradients/FullyConnected/Reshape_grad/ShapeShapeMaxPool2D_4/MaxPool^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
2Adam/gradients/FullyConnected/Reshape_grad/ReshapeReshape0Adam/gradients/FullyConnected/MatMul_grad/MatMul0Adam/gradients/FullyConnected/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
3Adam/gradients/MaxPool2D_4/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_4/ReluMaxPool2D_4/MaxPool2Adam/gradients/FullyConnected/Reshape_grad/Reshape*
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
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
�
3Adam/gradients/MaxPool2D_3/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_3/ReluMaxPool2D_3/MaxPool7Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropInput*
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
T0*
out_type0*
N* 
_output_shapes
::
�
7Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_3/Conv2D_grad/ShapeNConv2D_3/W/read*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*
paddingSAME*0
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
use_cudnn_on_gpu(
�
8Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_2/MaxPool,Adam/gradients/Conv2D_3/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�@
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
*Adam/gradients/Conv2D_2/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_2/MaxPool_grad/MaxPoolGradConv2D_2/Relu*
T0*0
_output_shapes
:����������
�
0Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
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
8Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_1/MaxPool,Adam/gradients/Conv2D_2/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_2/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*'
_output_shapes
:@�*
	dilations
*
T0
�
3Adam/gradients/MaxPool2D_1/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_1/ReluMaxPool2D_1/MaxPool7Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������

@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
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
7Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_1/Conv2D_grad/ShapeNConv2D_1/W/read*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*
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
:���������

 
�
8Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D/MaxPool,Adam/gradients/Conv2D_1/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*&
_output_shapes
: @
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
.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGradBiasAddGrad(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
(Adam/gradients/Conv2D/Conv2D_grad/ShapeNShapeNinput/XConv2D/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
5Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(Adam/gradients/Conv2D/Conv2D_grad/ShapeNConv2D/W/read(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
paddingSAME*/
_output_shapes
:���������22*
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
6Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput/X*Adam/gradients/Conv2D/Conv2D_grad/ShapeN:1(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
�
Adam/global_norm/L2LossL2Loss6Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*
T0*I
_class?
=;loc:@Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
Adam/global_norm/L2Loss_1L2Loss.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0*A
_class7
53loc:@Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad
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
Adam/global_norm/L2Loss_7L2Loss0Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0*C
_class9
75loc:@Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGrad
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
Adam/global_norm/L2Loss_13L2Loss8Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0*K
_classA
?=loc:@Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad
�
Adam/global_norm/stackPackAdam/global_norm/L2LossAdam/global_norm/L2Loss_1Adam/global_norm/L2Loss_2Adam/global_norm/L2Loss_3Adam/global_norm/L2Loss_4Adam/global_norm/L2Loss_5Adam/global_norm/L2Loss_6Adam/global_norm/L2Loss_7Adam/global_norm/L2Loss_8Adam/global_norm/L2Loss_9Adam/global_norm/L2Loss_10Adam/global_norm/L2Loss_11Adam/global_norm/L2Loss_12Adam/global_norm/L2Loss_13*
N*
_output_shapes
:*
T0*

axis 
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
Adam/global_norm/Const_1Const^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB
 *   @
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
 Adam/clip_by_global_norm/MinimumMinimum Adam/clip_by_global_norm/truediv"Adam/clip_by_global_norm/truediv_1*
_output_shapes
: *
T0
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
 Adam/clip_by_global_norm/Const_1Const^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB
 *  �
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
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_0IdentityAdam/clip_by_global_norm/mul_1*
T0*I
_class?
=;loc:@Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
Adam/clip_by_global_norm/mul_2Mul.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*A
_class7
53loc:@Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
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
Adam/clip_by_global_norm/mul_4Mul0Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_3IdentityAdam/clip_by_global_norm/mul_4*
_output_shapes
:@*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad
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
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_5IdentityAdam/clip_by_global_norm/mul_6*
_output_shapes	
:�*
T0*C
_class9
75loc:@Adam/gradients/Conv2D_2/BiasAdd_grad/BiasAddGrad
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
Adam/clip_by_global_norm/mul_11Mul2Adam/gradients/FullyConnected/MatMul_grad/MatMul_1Adam/clip_by_global_norm/Select*
_output_shapes
:	 �*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_10IdentityAdam/clip_by_global_norm/mul_11*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1*
_output_shapes
:	 �
�
Adam/clip_by_global_norm/mul_12Mul6Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*I
_class?
=;loc:@Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
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
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_13IdentityAdam/clip_by_global_norm/mul_14*
_output_shapes
:*
T0*K
_classA
?=loc:@Adam/gradients/FullyConnected_1/BiasAdd_grad/BiasAddGrad
�
Adam/beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Conv2D/W*
dtype0*
_output_shapes
: 
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
Adam/beta1_power/readIdentityAdam/beta1_power*
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
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
Conv2D/W/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
: *
_class
loc:@Conv2D/W*%
valueB *    
�
Conv2D/W/Adam
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
Conv2D/W/Adam/AssignAssignConv2D/W/AdamConv2D/W/Adam/Initializer/zeros*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
{
Conv2D/W/Adam/readIdentityConv2D/W/Adam*
T0*
_class
loc:@Conv2D/W*&
_output_shapes
: 
�
!Conv2D/W/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
: *
_class
loc:@Conv2D/W*%
valueB *    
�
Conv2D/W/Adam_1
VariableV2*
_class
loc:@Conv2D/W*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name 
�
Conv2D/W/Adam_1/AssignAssignConv2D/W/Adam_1!Conv2D/W/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
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
!Conv2D_1/W/Adam/Initializer/zerosFill1Conv2D_1/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_1/W/Adam/Initializer/zeros/Const*&
_output_shapes
: @*
T0*
_class
loc:@Conv2D_1/W*

index_type0
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
Conv2D_1/W/Adam/AssignAssignConv2D_1/W/Adam!Conv2D_1/W/Adam/Initializer/zeros*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
Conv2D_1/W/Adam/readIdentityConv2D_1/W/Adam*&
_output_shapes
: @*
T0*
_class
loc:@Conv2D_1/W
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
_class
loc:@Conv2D_1/W*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#Conv2D_1/W/Adam_1/Initializer/zerosFill3Conv2D_1/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_1/W/Adam_1/Initializer/zeros/Const*&
_output_shapes
: @*
T0*
_class
loc:@Conv2D_1/W*

index_type0
�
Conv2D_1/W/Adam_1
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
Conv2D_1/W/Adam_1/AssignAssignConv2D_1/W/Adam_1#Conv2D_1/W/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
�
Conv2D_1/W/Adam_1/readIdentityConv2D_1/W/Adam_1*&
_output_shapes
: @*
T0*
_class
loc:@Conv2D_1/W
�
!Conv2D_1/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@Conv2D_1/b*
valueB@*    
�
Conv2D_1/b/Adam
VariableV2*
_class
loc:@Conv2D_1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
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
#Conv2D_1/b/Adam_1/Initializer/zerosConst*
_class
loc:@Conv2D_1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
Conv2D_1/b/Adam_1
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
Conv2D_2/W/Adam/AssignAssignConv2D_2/W/Adam!Conv2D_2/W/Adam/Initializer/zeros*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0*
_class
loc:@Conv2D_2/W
�
Conv2D_2/W/Adam/readIdentityConv2D_2/W/Adam*
T0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�
3Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Conv2D_2/W*%
valueB"      @   �   *
dtype0*
_output_shapes
:
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
#Conv2D_2/W/Adam_1/Initializer/zerosFill3Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_2/W/Adam_1/Initializer/zeros/Const*'
_output_shapes
:@�*
T0*
_class
loc:@Conv2D_2/W*

index_type0
�
Conv2D_2/W/Adam_1
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
Conv2D_2/b/Adam/AssignAssignConv2D_2/b/Adam!Conv2D_2/b/Adam/Initializer/zeros*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
v
Conv2D_2/b/Adam/readIdentityConv2D_2/b/Adam*
T0*
_class
loc:@Conv2D_2/b*
_output_shapes	
:�
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
Conv2D_2/b/Adam_1/AssignAssignConv2D_2/b/Adam_1#Conv2D_2/b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
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
_class
loc:@Conv2D_3/W*
valueB
 *    *
dtype0*
_output_shapes
: 
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
VariableV2*
shared_name *
_class
loc:@Conv2D_3/W*
	container *
shape:�@*
dtype0*'
_output_shapes
:�@
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
_class
loc:@Conv2D_3/W*
	container *
shape:�@*
dtype0*'
_output_shapes
:�@*
shared_name 
�
Conv2D_3/W/Adam_1/AssignAssignConv2D_3/W/Adam_1#Conv2D_3/W/Adam_1/Initializer/zeros*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
Conv2D_3/W/Adam_1/readIdentityConv2D_3/W/Adam_1*
T0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
!Conv2D_3/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@Conv2D_3/b*
valueB@*    
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
Conv2D_3/b/Adam/readIdentityConv2D_3/b/Adam*
_output_shapes
:@*
T0*
_class
loc:@Conv2D_3/b
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
_class
loc:@Conv2D_4/W*%
valueB"      @       *
dtype0*
_output_shapes
:
�
'Conv2D_4/W/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_4/W*
valueB
 *    
�
!Conv2D_4/W/Adam/Initializer/zerosFill1Conv2D_4/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_4/W/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Conv2D_4/W*

index_type0*&
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
3Conv2D_4/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Conv2D_4/W*%
valueB"      @       *
dtype0*
_output_shapes
:
�
)Conv2D_4/W/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_4/W*
valueB
 *    
�
#Conv2D_4/W/Adam_1/Initializer/zerosFill3Conv2D_4/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_4/W/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Conv2D_4/W*

index_type0*&
_output_shapes
:@ 
�
Conv2D_4/W/Adam_1
VariableV2*
shape:@ *
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@Conv2D_4/W*
	container 
�
Conv2D_4/W/Adam_1/AssignAssignConv2D_4/W/Adam_1#Conv2D_4/W/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
Conv2D_4/W/Adam_1/readIdentityConv2D_4/W/Adam_1*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
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
Conv2D_4/b/Adam/AssignAssignConv2D_4/b/Adam!Conv2D_4/b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
u
Conv2D_4/b/Adam/readIdentityConv2D_4/b/Adam*
T0*
_class
loc:@Conv2D_4/b*
_output_shapes
: 
�
#Conv2D_4/b/Adam_1/Initializer/zerosConst*
_class
loc:@Conv2D_4/b*
valueB *    *
dtype0*
_output_shapes
: 
�
Conv2D_4/b/Adam_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D_4/b*
	container 
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
9FullyConnected/W/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@FullyConnected/W*
valueB"       *
dtype0*
_output_shapes
:
�
/FullyConnected/W/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *#
_class
loc:@FullyConnected/W*
valueB
 *    
�
)FullyConnected/W/Adam_1/Initializer/zerosFill9FullyConnected/W/Adam_1/Initializer/zeros/shape_as_tensor/FullyConnected/W/Adam_1/Initializer/zeros/Const*
_output_shapes
:	 �*
T0*#
_class
loc:@FullyConnected/W*

index_type0
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
FullyConnected/W/Adam_1/AssignAssignFullyConnected/W/Adam_1)FullyConnected/W/Adam_1/Initializer/zeros*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
�
FullyConnected/W/Adam_1/readIdentityFullyConnected/W/Adam_1*
_output_shapes
:	 �*
T0*#
_class
loc:@FullyConnected/W
�
7FullyConnected/b/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*#
_class
loc:@FullyConnected/b*
valueB:�
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
FullyConnected/b/Adam/AssignAssignFullyConnected/b/Adam'FullyConnected/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
FullyConnected/b/Adam/readIdentityFullyConnected/b/Adam*
T0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
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
FullyConnected/b/Adam_1/AssignAssignFullyConnected/b/Adam_1)FullyConnected/b/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
FullyConnected/b/Adam_1/readIdentityFullyConnected/b/Adam_1*
T0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
�
9FullyConnected_1/W/Adam/Initializer/zeros/shape_as_tensorConst*%
_class
loc:@FullyConnected_1/W*
valueB"      *
dtype0*
_output_shapes
:
�
/FullyConnected_1/W/Adam/Initializer/zeros/ConstConst*%
_class
loc:@FullyConnected_1/W*
valueB
 *    *
dtype0*
_output_shapes
: 
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
FullyConnected_1/W/Adam/AssignAssignFullyConnected_1/W/Adam)FullyConnected_1/W/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
FullyConnected_1/W/Adam/readIdentityFullyConnected_1/W/Adam*
_output_shapes
:	�*
T0*%
_class
loc:@FullyConnected_1/W
�
;FullyConnected_1/W/Adam_1/Initializer/zeros/shape_as_tensorConst*%
_class
loc:@FullyConnected_1/W*
valueB"      *
dtype0*
_output_shapes
:
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
+FullyConnected_1/W/Adam_1/Initializer/zerosFill;FullyConnected_1/W/Adam_1/Initializer/zeros/shape_as_tensor1FullyConnected_1/W/Adam_1/Initializer/zeros/Const*
_output_shapes
:	�*
T0*%
_class
loc:@FullyConnected_1/W*

index_type0
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
FullyConnected_1/W/Adam_1/readIdentityFullyConnected_1/W/Adam_1*
T0*%
_class
loc:@FullyConnected_1/W*
_output_shapes
:	�
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@FullyConnected_1/b
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
 FullyConnected_1/b/Adam_1/AssignAssignFullyConnected_1/b/Adam_1+FullyConnected_1/b/Adam_1/Initializer/zeros*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
Adam/apply_grad_op_0/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
.Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam	ApplyAdamConv2D/WConv2D/W/AdamConv2D/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_0*
use_locking( *
T0*
_class
loc:@Conv2D/W*
use_nesterov( *&
_output_shapes
: 
�
.Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam	ApplyAdamConv2D/bConv2D/b/AdamConv2D/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_1*
T0*
_class
loc:@Conv2D/b*
use_nesterov( *
_output_shapes
: *
use_locking( 
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
Conv2D_1/bConv2D_1/b/AdamConv2D_1/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_3*
T0*
_class
loc:@Conv2D_1/b*
use_nesterov( *
_output_shapes
:@*
use_locking( 
�
0Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam	ApplyAdam
Conv2D_2/WConv2D_2/W/AdamConv2D_2/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_4*
use_nesterov( *'
_output_shapes
:@�*
use_locking( *
T0*
_class
loc:@Conv2D_2/W
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
Conv2D_3/WConv2D_3/W/AdamConv2D_3/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_6*
use_nesterov( *'
_output_shapes
:�@*
use_locking( *
T0*
_class
loc:@Conv2D_3/W
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
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_9*
use_locking( *
T0*
_class
loc:@Conv2D_4/b*
use_nesterov( *
_output_shapes
: 
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
8Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam	ApplyAdamFullyConnected_1/bFullyConnected_1/b/AdamFullyConnected_1/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_13*
T0*%
_class
loc:@FullyConnected_1/b*
use_nesterov( *
_output_shapes
:*
use_locking( 
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
Adam/apply_grad_op_0/mul_1MulAdam/beta2_power/readAdam/apply_grad_op_0/beta2/^Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam/^Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam*
T0*
_class
loc:@Conv2D/W*
_output_shapes
: 
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
Adam/apply_grad_op_0/valueConst^Adam/apply_grad_op_0/update*
valueB
 *  �?* 
_class
loc:@Training_step*
dtype0*
_output_shapes
: 
�
Adam/apply_grad_op_0	AssignAddTraining_stepAdam/apply_grad_op_0/value*
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@Training_step
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
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:3*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
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
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*A
dtypes7
523
*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::
�
save/AssignAssignAccuracy/Mean/moving_avgsave/RestoreV2*
use_locking(*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
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
save/Assign_2AssignAdam/beta2_powersave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
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
save/Assign_5AssignConv2D/W/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
�
save/Assign_6AssignConv2D/bsave/RestoreV2:6*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/b
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
Conv2D_1/Wsave/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save/Assign_10AssignConv2D_1/W/Adamsave/RestoreV2:10*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
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
Conv2D_1/bsave/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save/Assign_13AssignConv2D_1/b/Adamsave/RestoreV2:13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save/Assign_14AssignConv2D_1/b/Adam_1save/RestoreV2:14*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
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
save/Assign_19AssignConv2D_2/b/Adamsave/RestoreV2:19*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_20AssignConv2D_2/b/Adam_1save/RestoreV2:20*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
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
save/Assign_22AssignConv2D_3/W/Adamsave/RestoreV2:22*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
save/Assign_23AssignConv2D_3/W/Adam_1save/RestoreV2:23*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save/Assign_24Assign
Conv2D_3/bsave/RestoreV2:24*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
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
save/Assign_26AssignConv2D_3/b/Adam_1save/RestoreV2:26*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_27Assign
Conv2D_4/Wsave/RestoreV2:27*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
save/Assign_28AssignConv2D_4/W/Adamsave/RestoreV2:28*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ *
use_locking(
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
Conv2D_4/bsave/RestoreV2:30*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
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
save/Assign_33AssignCrossentropy/Mean/moving_avgsave/RestoreV2:33*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg
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
save/Assign_35AssignFullyConnected/W/Adamsave/RestoreV2:35*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
�
save/Assign_36AssignFullyConnected/W/Adam_1save/RestoreV2:36*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save/Assign_37AssignFullyConnected/bsave/RestoreV2:37*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_39AssignFullyConnected/b/Adam_1save/RestoreV2:39*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_44AssignFullyConnected_1/b/Adamsave/RestoreV2:44*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
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
save/Assign_46AssignGlobal_Stepsave/RestoreV2:46*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Global_Step
�
save/Assign_47AssignTraining_stepsave/RestoreV2:47*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@Training_step
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
dtype0*
_output_shapes
: *
shape: 
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
save_1/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:3*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
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
save_1/AssignAssignAccuracy/Mean/moving_avgsave_1/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg
�
save_1/Assign_1AssignAdam/beta1_powersave_1/RestoreV2:1*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: *
use_locking(
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
save_1/Assign_3AssignConv2D/Wsave_1/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
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
save_1/Assign_5AssignConv2D/W/Adam_1save_1/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
�
save_1/Assign_6AssignConv2D/bsave_1/RestoreV2:6*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/b
�
save_1/Assign_7AssignConv2D/b/Adamsave_1/RestoreV2:7*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/b
�
save_1/Assign_8AssignConv2D/b/Adam_1save_1/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Conv2D/b*
validate_shape(*
_output_shapes
: 
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
save_1/Assign_10AssignConv2D_1/W/Adamsave_1/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save_1/Assign_11AssignConv2D_1/W/Adam_1save_1/RestoreV2:11*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
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
save_1/Assign_13AssignConv2D_1/b/Adamsave_1/RestoreV2:13*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
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
save_1/Assign_17AssignConv2D_2/W/Adam_1save_1/RestoreV2:17*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
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
save_1/Assign_19AssignConv2D_2/b/Adamsave_1/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�
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
save_1/Assign_22AssignConv2D_3/W/Adamsave_1/RestoreV2:22*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save_1/Assign_23AssignConv2D_3/W/Adam_1save_1/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save_1/Assign_24Assign
Conv2D_3/bsave_1/RestoreV2:24*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
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
Conv2D_4/bsave_1/RestoreV2:30*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D_4/b
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
save_1/Assign_32AssignConv2D_4/b/Adam_1save_1/RestoreV2:32*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_33AssignCrossentropy/Mean/moving_avgsave_1/RestoreV2:33*
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
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
save_1/Assign_37AssignFullyConnected/bsave_1/RestoreV2:37*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save_1/Assign_39AssignFullyConnected/b/Adam_1save_1/RestoreV2:39*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_1/Assign_40AssignFullyConnected_1/Wsave_1/RestoreV2:40*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W
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
save_1/Assign_42AssignFullyConnected_1/W/Adam_1save_1/RestoreV2:42*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_1/Assign_43AssignFullyConnected_1/bsave_1/RestoreV2:43*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
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
save_1/Assign_46AssignGlobal_Stepsave_1/RestoreV2:46*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Global_Step
�
save_1/Assign_47AssignTraining_stepsave_1/RestoreV2:47*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@Training_step
�
save_1/Assign_48Assignis_trainingsave_1/RestoreV2:48*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: 
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
shape: *
dtype0*
_output_shapes
: 
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
save_2/control_dependencyIdentitysave_2/Const^save_2/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save_2/Const
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
save_2/AssignAssignConv2D/Wsave_2/RestoreV2*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save_2/Assign_1AssignConv2D/bsave_2/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/b
�
save_2/Assign_2Assign
Conv2D_1/Wsave_2/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save_2/Assign_3Assign
Conv2D_1/bsave_2/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_4Assign
Conv2D_2/Wsave_2/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
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
Conv2D_3/Wsave_2/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
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
Conv2D_4/Wsave_2/RestoreV2:8*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ *
use_locking(
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
save_2/Assign_10AssignFullyConnected/Wsave_2/RestoreV2:10*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
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
save_2/Assign_12AssignFullyConnected_1/Wsave_2/RestoreV2:12*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W
�
save_2/Assign_13AssignFullyConnected_1/bsave_2/RestoreV2:13*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
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
save_3/control_dependencyIdentitysave_3/Const^save_3/SaveV2*
T0*
_class
loc:@save_3/Const*
_output_shapes
: 
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
!save_3/RestoreV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523

�
save_3/AssignAssignAccuracy/Mean/moving_avgsave_3/RestoreV2*
use_locking(*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
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
save_3/Assign_3AssignConv2D/Wsave_3/RestoreV2:3*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
�
save_3/Assign_4AssignConv2D/W/Adamsave_3/RestoreV2:4*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
�
save_3/Assign_5AssignConv2D/W/Adam_1save_3/RestoreV2:5*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
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
Conv2D_1/Wsave_3/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
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
save_3/Assign_11AssignConv2D_1/W/Adam_1save_3/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
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
save_3/Assign_13AssignConv2D_1/b/Adamsave_3/RestoreV2:13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_1/b
�
save_3/Assign_14AssignConv2D_1/b/Adam_1save_3/RestoreV2:14*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
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
save_3/Assign_17AssignConv2D_2/W/Adam_1save_3/RestoreV2:17*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0*
_class
loc:@Conv2D_2/W
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
save_3/Assign_20AssignConv2D_2/b/Adam_1save_3/RestoreV2:20*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
�
save_3/Assign_21Assign
Conv2D_3/Wsave_3/RestoreV2:21*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save_3/Assign_22AssignConv2D_3/W/Adamsave_3/RestoreV2:22*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
�
save_3/Assign_23AssignConv2D_3/W/Adam_1save_3/RestoreV2:23*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
�
save_3/Assign_24Assign
Conv2D_3/bsave_3/RestoreV2:24*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
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
save_3/Assign_26AssignConv2D_3/b/Adam_1save_3/RestoreV2:26*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
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
save_3/Assign_31AssignConv2D_4/b/Adamsave_3/RestoreV2:31*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
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
save_3/Assign_33AssignCrossentropy/Mean/moving_avgsave_3/RestoreV2:33*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_34AssignFullyConnected/Wsave_3/RestoreV2:34*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save_3/Assign_35AssignFullyConnected/W/Adamsave_3/RestoreV2:35*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
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
save_3/Assign_37AssignFullyConnected/bsave_3/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�
�
save_3/Assign_38AssignFullyConnected/b/Adamsave_3/RestoreV2:38*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
save_3/Assign_39AssignFullyConnected/b/Adam_1save_3/RestoreV2:39*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
�
save_3/Assign_40AssignFullyConnected_1/Wsave_3/RestoreV2:40*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
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
save_3/Assign_44AssignFullyConnected_1/b/Adamsave_3/RestoreV2:44*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
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
save_3/Assign_46AssignGlobal_Stepsave_3/RestoreV2:46*
use_locking(*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: 
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
save_3/restore_allNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9"��ө@q�     ��^	<��Ā��AJ��
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
)Conv2D/W/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@Conv2D/W*%
valueB"             
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

seed *
T0*
_class
loc:@Conv2D/W*
seed2 *
dtype0*&
_output_shapes
: 
�
'Conv2D/W/Initializer/random_uniform/subSub'Conv2D/W/Initializer/random_uniform/max'Conv2D/W/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
'Conv2D/W/Initializer/random_uniform/mulMul1Conv2D/W/Initializer/random_uniform/RandomUniform'Conv2D/W/Initializer/random_uniform/sub*
T0*
_class
loc:@Conv2D/W*&
_output_shapes
: 
�
#Conv2D/W/Initializer/random_uniformAdd'Conv2D/W/Initializer/random_uniform/mul'Conv2D/W/Initializer/random_uniform/min*&
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
Conv2D/W
VariableV2*
shared_name *
_class
loc:@Conv2D/W*
	container *
shape: *
dtype0*&
_output_shapes
: 
�
Conv2D/W/AssignAssignConv2D/W#Conv2D/W/Initializer/random_uniform*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
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
Conv2D/Conv2DConv2Dinput/XConv2D/W/read*
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
:���������22 
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
MaxPool2D/MaxPoolMaxPoolConv2D/Relu*
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
 
�
+Conv2D_1/W/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@Conv2D_1/W*%
valueB"          @   
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
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_1/W*
valueB
 *��z=
�
3Conv2D_1/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_1/W/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: @*

seed *
T0*
_class
loc:@Conv2D_1/W*
seed2 
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
_class
loc:@Conv2D_1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�

Conv2D_1/b
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
Conv2D_1/b*
_output_shapes
:@*
T0*
_class
loc:@Conv2D_1/b
�
Conv2D_1/Conv2DConv2DMaxPool2D/MaxPoolConv2D_1/W/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*/
_output_shapes
:���������

@*
	dilations

�
Conv2D_1/BiasAddBiasAddConv2D_1/Conv2DConv2D_1/b/read*
data_formatNHWC*/
_output_shapes
:���������

@*
T0
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
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_2/W*
valueB
 *�\1=
�
3Conv2D_2/W/Initializer/random_uniform/RandomUniformRandomUniform+Conv2D_2/W/Initializer/random_uniform/shape*
seed2 *
dtype0*'
_output_shapes
:@�*

seed *
T0*
_class
loc:@Conv2D_2/W
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
Conv2D_2/W%Conv2D_2/W/Initializer/random_uniform*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0*
_class
loc:@Conv2D_2/W
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
_class
loc:@Conv2D_2/b*
valueB�*    *
dtype0*
_output_shapes	
:�
�

Conv2D_2/b
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
Conv2D_2/b*
_output_shapes	
:�*
T0*
_class
loc:@Conv2D_2/b
�
Conv2D_2/Conv2DConv2DMaxPool2D_1/MaxPoolConv2D_2/W/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0
�
Conv2D_2/BiasAddBiasAddConv2D_2/Conv2DConv2D_2/b/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
b
Conv2D_2/ReluReluConv2D_2/BiasAdd*0
_output_shapes
:����������*
T0
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
_class
loc:@Conv2D_3/W*%
valueB"      �   @   *
dtype0*
_output_shapes
:
�
)Conv2D_3/W/Initializer/random_uniform/minConst*
_class
loc:@Conv2D_3/W*
valueB
 *����*
dtype0*
_output_shapes
: 
�
)Conv2D_3/W/Initializer/random_uniform/maxConst*
_class
loc:@Conv2D_3/W*
valueB
 *���<*
dtype0*
_output_shapes
: 
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
VariableV2*
shared_name *
_class
loc:@Conv2D_3/W*
	container *
shape:�@*
dtype0*'
_output_shapes
:�@
�
Conv2D_3/W/AssignAssign
Conv2D_3/W%Conv2D_3/W/Initializer/random_uniform*
validate_shape(*'
_output_shapes
:�@*
use_locking(*
T0*
_class
loc:@Conv2D_3/W
x
Conv2D_3/W/readIdentity
Conv2D_3/W*'
_output_shapes
:�@*
T0*
_class
loc:@Conv2D_3/W
�
Conv2D_3/b/Initializer/ConstConst*
_class
loc:@Conv2D_3/b*
valueB@*    *
dtype0*
_output_shapes
:@
�

Conv2D_3/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Conv2D_3/b*
	container 
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
Conv2D_3/Conv2DConv2DMaxPool2D_2/MaxPoolConv2D_3/W/read*
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
�
Conv2D_3/BiasAddBiasAddConv2D_3/Conv2DConv2D_3/b/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
a
Conv2D_3/ReluReluConv2D_3/BiasAdd*/
_output_shapes
:���������@*
T0
�
MaxPool2D_3/MaxPoolMaxPoolConv2D_3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
�
+Conv2D_4/W/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@Conv2D_4/W*%
valueB"      @       
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
dtype0*
_output_shapes
: *
_class
loc:@Conv2D_4/b*
valueB *    
�

Conv2D_4/b
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Conv2D_4/b
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
Conv2D_4/Conv2DConv2DMaxPool2D_3/MaxPoolConv2D_4/W/read*
paddingSAME*/
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
use_cudnn_on_gpu(
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
MaxPool2D_4/MaxPoolMaxPoolConv2D_4/Relu*
ksize
*
paddingSAME*/
_output_shapes
:��������� *
T0*
strides
*
data_formatNHWC
�
3FullyConnected/W/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@FullyConnected/W*
valueB"       
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
1FullyConnected/W/Initializer/truncated_normal/mulMul=FullyConnected/W/Initializer/truncated_normal/TruncatedNormal4FullyConnected/W/Initializer/truncated_normal/stddev*
_output_shapes
:	 �*
T0*#
_class
loc:@FullyConnected/W
�
-FullyConnected/W/Initializer/truncated_normalAdd1FullyConnected/W/Initializer/truncated_normal/mul2FullyConnected/W/Initializer/truncated_normal/mean*
T0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
�
FullyConnected/W
VariableV2*
shape:	 �*
dtype0*
_output_shapes
:	 �*
shared_name *#
_class
loc:@FullyConnected/W*
	container 
�
FullyConnected/W/AssignAssignFullyConnected/W-FullyConnected/W/Initializer/truncated_normal*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
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
FullyConnected/b/AssignAssignFullyConnected/b"FullyConnected/b/Initializer/Const*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
FullyConnected/ReshapeReshapeMaxPool2D_4/MaxPoolFullyConnected/Reshape/shape*'
_output_shapes
:��������� *
T0*
Tshape0
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
dtype0
*
_output_shapes
: *
_class
loc:@is_training*
value	B
 Z 
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
is_training/AssignAssignis_trainingis_training/Initializer/Const*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: *
use_locking(
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
Dropout/AssignAssignis_trainingDropout/Assign/value*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: *
use_locking(
X
Dropout/Assign_1/valueConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
�
Dropout/Assign_1Assignis_trainingDropout/Assign_1/value*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: 
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
Dropout/cond/pred_idIdentityis_training/read*
T0
*
_output_shapes
: 
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
'Dropout/cond/dropout/random_uniform/minConst^Dropout/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
'Dropout/cond/dropout/random_uniform/maxConst^Dropout/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
Dropout/cond/dropout/subSubDropout/cond/dropout/sub/xDropout/cond/dropout/rate*
_output_shapes
: *
T0
{
Dropout/cond/dropout/truediv/xConst^Dropout/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
Dropout/cond/Switch_1SwitchFullyConnected/ReluDropout/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*&
_class
loc:@FullyConnected/Relu
�
Dropout/cond/MergeMergeDropout/cond/Switch_1Dropout/cond/dropout/mul_1*
T0*
N**
_output_shapes
:����������: 
�
5FullyConnected_1/W/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
_class
loc:@FullyConnected_1/W*
valueB"      
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
dtype0*
_output_shapes
:	�*

seed *
T0*%
_class
loc:@FullyConnected_1/W*
seed2 
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
FullyConnected_1/b/AssignAssignFullyConnected_1/b$FullyConnected_1/b/Initializer/Const*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:
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
FullyConnected_1/SoftmaxSoftmaxFullyConnected_1/BiasAdd*'
_output_shapes
:���������*
T0
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
Accuracy/ArgMaxArgMaxFullyConnected_1/SoftmaxAccuracy/ArgMax/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
]
Accuracy/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
Accuracy/ArgMax_1ArgMax	targets/YAccuracy/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
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
Crossentropy/SumSumFullyConnected_1/Softmax"Crossentropy/Sum/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
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
Crossentropy/MeanMeanCrossentropy/NegCrossentropy/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
`
Training_step/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
Training_step
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
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
Training_step/readIdentityTraining_step*
_output_shapes
: *
T0* 
_class
loc:@Training_step
^
Global_Step/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
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
Global_Step/AssignAssignGlobal_StepGlobal_Step/initial_value*
use_locking(*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: 
j
Global_Step/readIdentityGlobal_Step*
_output_shapes
: *
T0*
_class
loc:@Global_Step
J
Add/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
val_loss/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
val_loss
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
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
val_acc/AssignAssignval_accval_acc/initial_value*
use_locking(*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: 
^
val_acc/readIdentityval_acc*
_output_shapes
: *
T0*
_class
loc:@val_acc
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
assign/val_lossAssignval_lossplaceholder/val_loss*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@val_loss
�
assign/val_accAssignval_accplaceholder/val_acc*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: *
use_locking(
�
*Accuracy/Mean/moving_avg/Initializer/zerosConst*+
_class!
loc:@Accuracy/Mean/moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
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
Accuracy/Mean/moving_avg/AssignAssignAccuracy/Mean/moving_avg*Accuracy/Mean/moving_avg/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
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
 moving_avg/AssignMovingAvg/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
|
moving_avg/AssignMovingAvg/subSub moving_avg/AssignMovingAvg/sub/xmoving_avg/Minimum*
_output_shapes
: *
T0
v
 moving_avg/AssignMovingAvg/sub_1SubAccuracy/Mean/moving_avg/readAccuracy/Mean*
_output_shapes
: *
T0
�
moving_avg/AssignMovingAvg/mulMul moving_avg/AssignMovingAvg/sub_1moving_avg/AssignMovingAvg/sub*
_output_shapes
: *
T0
�
moving_avg/AssignMovingAvg	AssignSubAccuracy/Mean/moving_avgmoving_avg/AssignMovingAvg/mul*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
_output_shapes
: *
use_locking( 
/

moving_avgNoOp^moving_avg/AssignMovingAvg
O
Adam/Total_LossIdentityCrossentropy/Mean*
T0*
_output_shapes
: 
�
.Crossentropy/Mean/moving_avg/Initializer/zerosConst*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Crossentropy/Mean/moving_avg
VariableV2*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
Adam/moving_avg/addAddV2Adam/moving_avg/add/xTraining_step/read*
T0*
_output_shapes
: 
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
%Adam/moving_avg/AssignMovingAvg/sub_1Sub!Crossentropy/Mean/moving_avg/readCrossentropy/Mean*
T0*
_output_shapes
: 
�
#Adam/moving_avg/AssignMovingAvg/mulMul%Adam/moving_avg/AssignMovingAvg/sub_1#Adam/moving_avg/AssignMovingAvg/sub*
T0*
_output_shapes
: 
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
*Adam/gradients/Crossentropy/Mean_grad/TileTile-Adam/gradients/Crossentropy/Mean_grad/Reshape+Adam/gradients/Crossentropy/Mean_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
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
*Adam/gradients/Crossentropy/Mean_grad/ProdProd-Adam/gradients/Crossentropy/Mean_grad/Shape_1+Adam/gradients/Crossentropy/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
*Adam/gradients/Crossentropy/Mean_grad/CastCast.Adam/gradients/Crossentropy/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
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
.Adam/gradients/Crossentropy/Sum_1_grad/Shape_1Const^Adam/moving_avg^moving_avg*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
2Adam/gradients/Crossentropy/Sum_1_grad/range/startConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
value	B : 
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
+Adam/gradients/Crossentropy/Sum_1_grad/FillFill.Adam/gradients/Crossentropy/Sum_1_grad/Shape_11Adam/gradients/Crossentropy/Sum_1_grad/Fill/value*
_output_shapes
: *
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*

index_type0
�
4Adam/gradients/Crossentropy/Sum_1_grad/DynamicStitchDynamicStitch,Adam/gradients/Crossentropy/Sum_1_grad/range*Adam/gradients/Crossentropy/Sum_1_grad/mod,Adam/gradients/Crossentropy/Sum_1_grad/Shape+Adam/gradients/Crossentropy/Sum_1_grad/Fill*
T0*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
N*
_output_shapes
:
�
0Adam/gradients/Crossentropy/Sum_1_grad/Maximum/yConst^Adam/moving_avg^moving_avg*?
_class5
31loc:@Adam/gradients/Crossentropy/Sum_1_grad/Shape*
value	B :*
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
*Adam/gradients/Crossentropy/mul_grad/ShapeShape	targets/Y^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
,Adam/gradients/Crossentropy/mul_grad/Shape_1ShapeCrossentropy/Log^Adam/moving_avg^moving_avg*
_output_shapes
:*
T0*
out_type0
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
*Adam/gradients/Crossentropy/mul_grad/Sum_1Sum*Adam/gradients/Crossentropy/mul_grad/Mul_1<Adam/gradients/Crossentropy/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
;Adam/gradients/Crossentropy/clip_by_value_grad/GreaterEqualGreaterEqual"Crossentropy/clip_by_value/MinimumCrossentropy/Cast/x^Adam/moving_avg^moving_avg*'
_output_shapes
:���������*
T0
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
4Adam/gradients/Crossentropy/clip_by_value_grad/Sum_1Sum7Adam/gradients/Crossentropy/clip_by_value_grad/Select_1FAdam/gradients/Crossentropy/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/zerosFill>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape_2BAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:���������*
T0*

index_type0
�
@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/LessEqual	LessEqualCrossentropy/truedivCrossentropy/Cast_1/x^Adam/moving_avg^moving_avg*'
_output_shapes
:���������*
T0
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
:Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SumSum=Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/SelectLAdam/gradients/Crossentropy/clip_by_value/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/ReshapeReshape:Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Sum<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
?Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Select_1Select@Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/LessEqual<Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/zeros6Adam/gradients/Crossentropy/clip_by_value_grad/Reshape*'
_output_shapes
:���������*
T0
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
>Adam/gradients/Crossentropy/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs.Adam/gradients/Crossentropy/truediv_grad/Shape0Adam/gradients/Crossentropy/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0Adam/gradients/Crossentropy/truediv_grad/RealDivRealDiv>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/ReshapeCrossentropy/Sum*'
_output_shapes
:���������*
T0
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
,Adam/gradients/Crossentropy/truediv_grad/mulMul>Adam/gradients/Crossentropy/clip_by_value/Minimum_grad/Reshape2Adam/gradients/Crossentropy/truediv_grad/RealDiv_2*'
_output_shapes
:���������*
T0
�
.Adam/gradients/Crossentropy/truediv_grad/Sum_1Sum,Adam/gradients/Crossentropy/truediv_grad/mul@Adam/gradients/Crossentropy/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
2Adam/gradients/Crossentropy/truediv_grad/Reshape_1Reshape.Adam/gradients/Crossentropy/truediv_grad/Sum_10Adam/gradients/Crossentropy/truediv_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
*Adam/gradients/Crossentropy/Sum_grad/ShapeShapeFullyConnected_1/Softmax^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
)Adam/gradients/Crossentropy/Sum_grad/SizeConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
value	B :
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
)Adam/gradients/Crossentropy/Sum_grad/FillFill,Adam/gradients/Crossentropy/Sum_grad/Shape_1/Adam/gradients/Crossentropy/Sum_grad/Fill/value*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*

index_type0*
_output_shapes
: 
�
2Adam/gradients/Crossentropy/Sum_grad/DynamicStitchDynamicStitch*Adam/gradients/Crossentropy/Sum_grad/range(Adam/gradients/Crossentropy/Sum_grad/mod*Adam/gradients/Crossentropy/Sum_grad/Shape)Adam/gradients/Crossentropy/Sum_grad/Fill*
T0*=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
N*
_output_shapes
:
�
.Adam/gradients/Crossentropy/Sum_grad/Maximum/yConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *=
_class3
1/loc:@Adam/gradients/Crossentropy/Sum_grad/Shape*
value	B :
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
,Adam/gradients/Crossentropy/Sum_grad/ReshapeReshape2Adam/gradients/Crossentropy/truediv_grad/Reshape_12Adam/gradients/Crossentropy/Sum_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0
�
)Adam/gradients/Crossentropy/Sum_grad/TileTile,Adam/gradients/Crossentropy/Sum_grad/Reshape-Adam/gradients/Crossentropy/Sum_grad/floordiv*
T0*'
_output_shapes
:���������*

Tmultiples0
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
0Adam/gradients/FullyConnected_1/Softmax_grad/SumSum0Adam/gradients/FullyConnected_1/Softmax_grad/mulBAdam/gradients/FullyConnected_1/Softmax_grad/Sum/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
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
2Adam/gradients/FullyConnected_1/MatMul_grad/MatMulMatMul2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1FullyConnected_1/W/read*
transpose_a( *(
_output_shapes
:����������*
transpose_b(*
T0
�
4Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1MatMulDropout/cond/Merge2Adam/gradients/FullyConnected_1/Softmax_grad/mul_1*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	�
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
:*

Tidx0*
	keep_dims( 
�
6Adam/gradients/Dropout/cond/dropout/mul_1_grad/ReshapeReshape2Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Mul_1MulDropout/cond/dropout/mul2Adam/gradients/Dropout/cond/Merge_grad/cond_grad:1*
T0*(
_output_shapes
:����������
�
4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum_1Sum4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Mul_1FAdam/gradients/Dropout/cond/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
8Adam/gradients/Dropout/cond/dropout/mul_1_grad/Reshape_1Reshape4Adam/gradients/Dropout/cond/dropout/mul_1_grad/Sum_16Adam/gradients/Dropout/cond/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
BAdam/gradients/Dropout/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2Adam/gradients/Dropout/cond/dropout/mul_grad/Shape4Adam/gradients/Dropout/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
2Adam/gradients/Dropout/cond/dropout/mul_grad/Mul_1Mul#Dropout/cond/dropout/Shape/Switch:16Adam/gradients/Dropout/cond/dropout/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
2Adam/gradients/Dropout/cond/dropout/mul_grad/Sum_1Sum2Adam/gradients/Dropout/cond/dropout/mul_grad/Mul_1DAdam/gradients/Dropout/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
6Adam/gradients/Dropout/cond/dropout/mul_grad/Reshape_1Reshape2Adam/gradients/Dropout/cond/dropout/mul_grad/Sum_14Adam/gradients/Dropout/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Adam/gradients/Switch_1SwitchFullyConnected/ReluDropout/cond/pred_id^Adam/moving_avg^moving_avg*<
_output_shapes*
(:����������:����������*
T0
q
Adam/gradients/Identity_1IdentityAdam/gradients/Switch_1*
T0*(
_output_shapes
:����������
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
N*(
_output_shapes
:����������*
T0*F
_class<
:8loc:@Adam/gradients/Dropout/cond/Switch_1_grad/cond_grad
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
transpose_a( *'
_output_shapes
:��������� *
transpose_b(*
T0
�
2Adam/gradients/FullyConnected/MatMul_grad/MatMul_1MatMulFullyConnected/Reshape0Adam/gradients/FullyConnected/Relu_grad/ReluGrad*
T0*
transpose_a(*
_output_shapes
:	 �*
transpose_b( 
�
0Adam/gradients/FullyConnected/Reshape_grad/ShapeShapeMaxPool2D_4/MaxPool^Adam/moving_avg^moving_avg*
T0*
out_type0*
_output_shapes
:
�
2Adam/gradients/FullyConnected/Reshape_grad/ReshapeReshape0Adam/gradients/FullyConnected/MatMul_grad/MatMul0Adam/gradients/FullyConnected/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
3Adam/gradients/MaxPool2D_4/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_4/ReluMaxPool2D_4/MaxPool2Adam/gradients/FullyConnected/Reshape_grad/Reshape*
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
*Adam/gradients/Conv2D_4/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_4/MaxPool_grad/MaxPoolGradConv2D_4/Relu*/
_output_shapes
:��������� *
T0
�
0Adam/gradients/Conv2D_4/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
�
*Adam/gradients/Conv2D_4/Conv2D_grad/ShapeNShapeNMaxPool2D_3/MaxPoolConv2D_4/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
7Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_4/Conv2D_grad/ShapeNConv2D_4/W/read*Adam/gradients/Conv2D_4/Relu_grad/ReluGrad*
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
:���������@
�
8Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_3/MaxPool,Adam/gradients/Conv2D_4/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_4/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*&
_output_shapes
:@ *
	dilations
*
T0
�
3Adam/gradients/MaxPool2D_3/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_3/ReluMaxPool2D_3/MaxPool7Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropInput*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
�
*Adam/gradients/Conv2D_3/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_3/MaxPool_grad/MaxPoolGradConv2D_3/Relu*
T0*/
_output_shapes
:���������@
�
0Adam/gradients/Conv2D_3/BiasAdd_grad/BiasAddGradBiasAddGrad*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
*Adam/gradients/Conv2D_3/Conv2D_grad/ShapeNShapeNMaxPool2D_2/MaxPoolConv2D_3/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
7Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_3/Conv2D_grad/ShapeNConv2D_3/W/read*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0
�
8Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_2/MaxPool,Adam/gradients/Conv2D_3/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_3/Relu_grad/ReluGrad*'
_output_shapes
:�@*
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
3Adam/gradients/MaxPool2D_2/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_2/ReluMaxPool2D_2/MaxPool7Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
�
*Adam/gradients/Conv2D_2/Relu_grad/ReluGradReluGrad3Adam/gradients/MaxPool2D_2/MaxPool_grad/MaxPoolGradConv2D_2/Relu*
T0*0
_output_shapes
:����������
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
8Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D_1/MaxPool,Adam/gradients/Conv2D_2/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_2/Relu_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*'
_output_shapes
:@�*
	dilations
*
T0
�
3Adam/gradients/MaxPool2D_1/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D_1/ReluMaxPool2D_1/MaxPool7Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingSAME*/
_output_shapes
:���������

@*
T0*
data_formatNHWC*
strides

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
7Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*Adam/gradients/Conv2D_1/Conv2D_grad/ShapeNConv2D_1/W/read*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*
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
:���������

 
�
8Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool2D/MaxPool,Adam/gradients/Conv2D_1/Conv2D_grad/ShapeN:1*Adam/gradients/Conv2D_1/Relu_grad/ReluGrad*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0
�
1Adam/gradients/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradConv2D/ReluMaxPool2D/MaxPool7Adam/gradients/Conv2D_1/Conv2D_grad/Conv2DBackpropInput*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������22 *
T0
�
(Adam/gradients/Conv2D/Relu_grad/ReluGradReluGrad1Adam/gradients/MaxPool2D/MaxPool_grad/MaxPoolGradConv2D/Relu*
T0*/
_output_shapes
:���������22 
�
.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGradBiasAddGrad(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
(Adam/gradients/Conv2D/Conv2D_grad/ShapeNShapeNinput/XConv2D/W/read^Adam/moving_avg^moving_avg*
T0*
out_type0*
N* 
_output_shapes
::
�
5Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(Adam/gradients/Conv2D/Conv2D_grad/ShapeNConv2D/W/read(Adam/gradients/Conv2D/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*/
_output_shapes
:���������22*
	dilations
*
T0
�
6Adam/gradients/Conv2D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput/X*Adam/gradients/Conv2D/Conv2D_grad/ShapeN:1(Adam/gradients/Conv2D/Relu_grad/ReluGrad*&
_output_shapes
: *
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
Adam/global_norm/L2Loss_3L2Loss0Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0*C
_class9
75loc:@Adam/gradients/Conv2D_1/BiasAdd_grad/BiasAddGrad
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
Adam/global_norm/L2Loss_6L2Loss8Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: *
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_3/Conv2D_grad/Conv2DBackpropFilter
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
N*
_output_shapes
:*
T0*

axis 

Adam/global_norm/ConstConst^Adam/moving_avg^moving_avg*
valueB: *
dtype0*
_output_shapes
:
�
Adam/global_norm/SumSumAdam/global_norm/stackAdam/global_norm/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
|
Adam/global_norm/Const_1Const^Adam/moving_avg^moving_avg*
valueB
 *   @*
dtype0*
_output_shapes
: 
l
Adam/global_norm/mulMulAdam/global_norm/SumAdam/global_norm/Const_1*
_output_shapes
: *
T0
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
Adam/clip_by_global_norm/ConstConst^Adam/moving_avg^moving_avg*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
 Adam/clip_by_global_norm/MinimumMinimum Adam/clip_by_global_norm/truediv"Adam/clip_by_global_norm/truediv_1*
_output_shapes
: *
T0
�
Adam/clip_by_global_norm/mul/xConst^Adam/moving_avg^moving_avg*
dtype0*
_output_shapes
: *
valueB
 *  �@
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
Adam/clip_by_global_norm/mul_2Mul.Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*A
_class7
53loc:@Adam/gradients/Conv2D/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
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
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_4IdentityAdam/clip_by_global_norm/mul_5*'
_output_shapes
:@�*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter
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
Adam/clip_by_global_norm/mul_9Mul8Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilterAdam/clip_by_global_norm/Select*&
_output_shapes
:@ *
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter
�
4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_8IdentityAdam/clip_by_global_norm/mul_9*
T0*K
_classA
?=loc:@Adam/gradients/Conv2D_4/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@ 
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
Adam/clip_by_global_norm/mul_11Mul2Adam/gradients/FullyConnected/MatMul_grad/MatMul_1Adam/clip_by_global_norm/Select*
_output_shapes
:	 �*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_10IdentityAdam/clip_by_global_norm/mul_11*
T0*E
_class;
97loc:@Adam/gradients/FullyConnected/MatMul_grad/MatMul_1*
_output_shapes
:	 �
�
Adam/clip_by_global_norm/mul_12Mul6Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGradAdam/clip_by_global_norm/Select*
T0*I
_class?
=;loc:@Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_11IdentityAdam/clip_by_global_norm/mul_12*
T0*I
_class?
=;loc:@Adam/gradients/FullyConnected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Adam/clip_by_global_norm/mul_13Mul4Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1Adam/clip_by_global_norm/Select*
T0*G
_class=
;9loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_12IdentityAdam/clip_by_global_norm/mul_13*
_output_shapes
:	�*
T0*G
_class=
;9loc:@Adam/gradients/FullyConnected_1/MatMul_grad/MatMul_1
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
Adam/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@Conv2D/W*
valueB
 *w�?
�
Adam/beta2_power
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
Conv2D/W/Adam/AssignAssignConv2D/W/AdamConv2D/W/Adam/Initializer/zeros*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: *
use_locking(
{
Conv2D/W/Adam/readIdentityConv2D/W/Adam*&
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
�
!Conv2D/W/Adam_1/Initializer/zerosConst*%
valueB *    *
_class
loc:@Conv2D/W*
dtype0*&
_output_shapes
: 
�
Conv2D/W/Adam_1
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
Conv2D/W/Adam_1/AssignAssignConv2D/W/Adam_1!Conv2D/W/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W

Conv2D/W/Adam_1/readIdentityConv2D/W/Adam_1*&
_output_shapes
: *
T0*
_class
loc:@Conv2D/W
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
VariableV2*
_class
loc:@Conv2D/b*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
!Conv2D/b/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@Conv2D/b*
dtype0*
_output_shapes
: 
�
Conv2D/b/Adam_1
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
!Conv2D_1/b/Adam/Initializer/zerosConst*
valueB@*    *
_class
loc:@Conv2D_1/b*
dtype0*
_output_shapes
:@
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
Conv2D_1/b/Adam_1/readIdentityConv2D_1/b/Adam_1*
T0*
_class
loc:@Conv2D_1/b*
_output_shapes
:@
�
1Conv2D_2/W/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   �   *
_class
loc:@Conv2D_2/W
�
'Conv2D_2/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Conv2D_2/W*
dtype0*
_output_shapes
: 
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
_class
loc:@Conv2D_2/W*
	container *
shape:@�*
dtype0*'
_output_shapes
:@�*
shared_name 
�
Conv2D_2/W/Adam/AssignAssignConv2D_2/W/Adam!Conv2D_2/W/Adam/Initializer/zeros*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�*
use_locking(
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
#Conv2D_2/W/Adam_1/Initializer/zerosFill3Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensor)Conv2D_2/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�
Conv2D_2/W/Adam_1
VariableV2*
	container *
shape:@�*
dtype0*'
_output_shapes
:@�*
shared_name *
_class
loc:@Conv2D_2/W
�
Conv2D_2/W/Adam_1/AssignAssignConv2D_2/W/Adam_1#Conv2D_2/W/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
�
Conv2D_2/W/Adam_1/readIdentityConv2D_2/W/Adam_1*
T0*
_class
loc:@Conv2D_2/W*'
_output_shapes
:@�
�
!Conv2D_2/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Conv2D_2/b
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
Conv2D_2/b/Adam/AssignAssignConv2D_2/b/Adam!Conv2D_2/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
v
Conv2D_2/b/Adam/readIdentityConv2D_2/b/Adam*
T0*
_class
loc:@Conv2D_2/b*
_output_shapes	
:�
�
#Conv2D_2/b/Adam_1/Initializer/zerosConst*
valueB�*    *
_class
loc:@Conv2D_2/b*
dtype0*
_output_shapes	
:�
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
1Conv2D_3/W/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      �   @   *
_class
loc:@Conv2D_3/W*
dtype0*
_output_shapes
:
�
'Conv2D_3/W/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Conv2D_3/W
�
!Conv2D_3/W/Adam/Initializer/zerosFill1Conv2D_3/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_3/W/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Conv2D_3/W*'
_output_shapes
:�@
�
Conv2D_3/W/Adam
VariableV2*
shared_name *
_class
loc:@Conv2D_3/W*
	container *
shape:�@*
dtype0*'
_output_shapes
:�@
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
3Conv2D_3/W/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      �   @   *
_class
loc:@Conv2D_3/W*
dtype0*
_output_shapes
:
�
)Conv2D_3/W/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Conv2D_3/W
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
dtype0*
_output_shapes
:@*
valueB@*    *
_class
loc:@Conv2D_3/b
�
Conv2D_3/b/Adam
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
Conv2D_3/b/Adam_1/AssignAssignConv2D_3/b/Adam_1#Conv2D_3/b/Adam_1/Initializer/zeros*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
y
Conv2D_3/b/Adam_1/readIdentityConv2D_3/b/Adam_1*
T0*
_class
loc:@Conv2D_3/b*
_output_shapes
:@
�
1Conv2D_4/W/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      @       *
_class
loc:@Conv2D_4/W*
dtype0*
_output_shapes
:
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
!Conv2D_4/W/Adam/Initializer/zerosFill1Conv2D_4/W/Adam/Initializer/zeros/shape_as_tensor'Conv2D_4/W/Adam/Initializer/zeros/Const*&
_output_shapes
:@ *
T0*

index_type0*
_class
loc:@Conv2D_4/W
�
Conv2D_4/W/Adam
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
Conv2D_4/W/Adam/AssignAssignConv2D_4/W/Adam!Conv2D_4/W/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
�
Conv2D_4/W/Adam/readIdentityConv2D_4/W/Adam*&
_output_shapes
:@ *
T0*
_class
loc:@Conv2D_4/W
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
Conv2D_4/W/Adam_1/readIdentityConv2D_4/W/Adam_1*
T0*
_class
loc:@Conv2D_4/W*&
_output_shapes
:@ 
�
!Conv2D_4/b/Adam/Initializer/zerosConst*
valueB *    *
_class
loc:@Conv2D_4/b*
dtype0*
_output_shapes
: 
�
Conv2D_4/b/Adam
VariableV2*
_class
loc:@Conv2D_4/b*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
Conv2D_4/b/Adam/AssignAssignConv2D_4/b/Adam!Conv2D_4/b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
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
Conv2D_4/b/Adam_1/AssignAssignConv2D_4/b/Adam_1#Conv2D_4/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D_4/b
y
Conv2D_4/b/Adam_1/readIdentityConv2D_4/b/Adam_1*
T0*
_class
loc:@Conv2D_4/b*
_output_shapes
: 
�
7FullyConnected/W/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"       *#
_class
loc:@FullyConnected/W
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
'FullyConnected/W/Adam/Initializer/zerosFill7FullyConnected/W/Adam/Initializer/zeros/shape_as_tensor-FullyConnected/W/Adam/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@FullyConnected/W*
_output_shapes
:	 �
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
)FullyConnected/W/Adam_1/Initializer/zerosFill9FullyConnected/W/Adam_1/Initializer/zeros/shape_as_tensor/FullyConnected/W/Adam_1/Initializer/zeros/Const*
_output_shapes
:	 �*
T0*

index_type0*#
_class
loc:@FullyConnected/W
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
FullyConnected/W/Adam_1/AssignAssignFullyConnected/W/Adam_1)FullyConnected/W/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
�
FullyConnected/W/Adam_1/readIdentityFullyConnected/W/Adam_1*
_output_shapes
:	 �*
T0*#
_class
loc:@FullyConnected/W
�
7FullyConnected/b/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:�*#
_class
loc:@FullyConnected/b*
dtype0*
_output_shapes
:
�
-FullyConnected/b/Adam/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@FullyConnected/b*
dtype0*
_output_shapes
: 
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
FullyConnected/b/Adam/AssignAssignFullyConnected/b/Adam'FullyConnected/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
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
/FullyConnected/b/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@FullyConnected/b*
dtype0*
_output_shapes
: 
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
FullyConnected/b/Adam_1/AssignAssignFullyConnected/b/Adam_1)FullyConnected/b/Adam_1/Initializer/zeros*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
FullyConnected/b/Adam_1/readIdentityFullyConnected/b/Adam_1*
T0*#
_class
loc:@FullyConnected/b*
_output_shapes	
:�
�
9FullyConnected_1/W/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *%
_class
loc:@FullyConnected_1/W*
dtype0*
_output_shapes
:
�
/FullyConnected_1/W/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *%
_class
loc:@FullyConnected_1/W
�
)FullyConnected_1/W/Adam/Initializer/zerosFill9FullyConnected_1/W/Adam/Initializer/zeros/shape_as_tensor/FullyConnected_1/W/Adam/Initializer/zeros/Const*
_output_shapes
:	�*
T0*

index_type0*%
_class
loc:@FullyConnected_1/W
�
FullyConnected_1/W/Adam
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
valueB"      *%
_class
loc:@FullyConnected_1/W*
dtype0*
_output_shapes
:
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
VariableV2*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *%
_class
loc:@FullyConnected_1/W*
	container 
�
 FullyConnected_1/W/Adam_1/AssignAssignFullyConnected_1/W/Adam_1+FullyConnected_1/W/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
FullyConnected_1/W/Adam_1/readIdentityFullyConnected_1/W/Adam_1*
_output_shapes
:	�*
T0*%
_class
loc:@FullyConnected_1/W
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@FullyConnected_1/b
�
FullyConnected_1/b/Adam/AssignAssignFullyConnected_1/b/Adam)FullyConnected_1/b/Adam/Initializer/zeros*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
FullyConnected_1/b/Adam/readIdentityFullyConnected_1/b/Adam*
T0*%
_class
loc:@FullyConnected_1/b*
_output_shapes
:
�
+FullyConnected_1/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *%
_class
loc:@FullyConnected_1/b
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
 FullyConnected_1/b/Adam_1/AssignAssignFullyConnected_1/b/Adam_1+FullyConnected_1/b/Adam_1/Initializer/zeros*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
.Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam	ApplyAdamConv2D/WConv2D/W/AdamConv2D/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_0*
use_locking( *
T0*
_class
loc:@Conv2D/W*
use_nesterov( *&
_output_shapes
: 
�
.Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam	ApplyAdamConv2D/bConv2D/b/AdamConv2D/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_1*
T0*
_class
loc:@Conv2D/b*
use_nesterov( *
_output_shapes
: *
use_locking( 
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
Conv2D_2/bConv2D_2/b/AdamConv2D_2/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_5*
use_locking( *
T0*
_class
loc:@Conv2D_2/b*
use_nesterov( *
_output_shapes	
:�
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
Conv2D_4/WConv2D_4/W/AdamConv2D_4/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_8*
use_locking( *
T0*
_class
loc:@Conv2D_4/W*
use_nesterov( *&
_output_shapes
:@ 
�
0Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam	ApplyAdam
Conv2D_4/bConv2D_4/b/AdamConv2D_4/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon4Adam/clip_by_global_norm/Adam/clip_by_global_norm/_9*
use_locking( *
T0*
_class
loc:@Conv2D_4/b*
use_nesterov( *
_output_shapes
: 
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
6Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam	ApplyAdamFullyConnected/bFullyConnected/b/AdamFullyConnected/b/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_11*
use_locking( *
T0*#
_class
loc:@FullyConnected/b*
use_nesterov( *
_output_shapes	
:�
�
8Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam	ApplyAdamFullyConnected_1/WFullyConnected_1/W/AdamFullyConnected_1/W/Adam_1Adam/beta1_power/readAdam/beta2_power/read"Adam/apply_grad_op_0/learning_rateAdam/apply_grad_op_0/beta1Adam/apply_grad_op_0/beta2Adam/apply_grad_op_0/epsilon5Adam/clip_by_global_norm/Adam/clip_by_global_norm/_12*
use_locking( *
T0*%
_class
loc:@FullyConnected_1/W*
use_nesterov( *
_output_shapes
:	�
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
Adam/apply_grad_op_0/AssignAssignAdam/beta1_powerAdam/apply_grad_op_0/mul*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: *
use_locking( 
�
Adam/apply_grad_op_0/mul_1MulAdam/beta2_power/readAdam/apply_grad_op_0/beta2/^Adam/apply_grad_op_0/update_Conv2D/W/ApplyAdam/^Adam/apply_grad_op_0/update_Conv2D/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_1/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_2/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_3/b/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/W/ApplyAdam1^Adam/apply_grad_op_0/update_Conv2D_4/b/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/W/ApplyAdam7^Adam/apply_grad_op_0/update_FullyConnected/b/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/W/ApplyAdam9^Adam/apply_grad_op_0/update_FullyConnected_1/b/ApplyAdam*
T0*
_class
loc:@Conv2D/W*
_output_shapes
: 
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
dtype0*
_output_shapes
: * 
_class
loc:@Training_step*
valueB
 *  �?
�
Adam/apply_grad_op_0	AssignAddTraining_stepAdam/apply_grad_op_0/value*
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@Training_step
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
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*A
dtypes7
523
*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::
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
save/Assign_1AssignAdam/beta1_powersave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
�
save/Assign_2AssignAdam/beta2_powersave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*
_output_shapes
: 
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
save/Assign_5AssignConv2D/W/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
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
save/Assign_11AssignConv2D_1/W/Adam_1save/RestoreV2:11*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save/Assign_12Assign
Conv2D_1/bsave/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
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
save/Assign_14AssignConv2D_1/b/Adam_1save/RestoreV2:14*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
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
Conv2D_2/bsave/RestoreV2:18*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
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
save/Assign_20AssignConv2D_2/b/Adam_1save/RestoreV2:20*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Conv2D_2/b
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
Conv2D_3/bsave/RestoreV2:24*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_25AssignConv2D_3/b/Adamsave/RestoreV2:25*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
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
Conv2D_4/Wsave/RestoreV2:27*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
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
Conv2D_4/bsave/RestoreV2:30*
use_locking(*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: 
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
save/Assign_33AssignCrossentropy/Mean/moving_avgsave/RestoreV2:33*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg
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
save/Assign_35AssignFullyConnected/W/Adamsave/RestoreV2:35*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
�
save/Assign_36AssignFullyConnected/W/Adam_1save/RestoreV2:36*
validate_shape(*
_output_shapes
:	 �*
use_locking(*
T0*#
_class
loc:@FullyConnected/W
�
save/Assign_37AssignFullyConnected/bsave/RestoreV2:37*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_43AssignFullyConnected_1/bsave/RestoreV2:43*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_45AssignFullyConnected_1/b/Adam_1save/RestoreV2:45*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_49Assignval_accsave/RestoreV2:49*
T0*
_class
loc:@val_acc*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_50Assignval_losssave/RestoreV2:50*
use_locking(*
T0*
_class
loc:@val_loss*
validate_shape(*
_output_shapes
: 
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
[
save_1/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
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
save_1/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:3*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
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
save_1/Assign_3AssignConv2D/Wsave_1/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
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
save_1/Assign_5AssignConv2D/W/Adam_1save_1/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Conv2D/W*
validate_shape(*&
_output_shapes
: 
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
save_1/Assign_8AssignConv2D/b/Adam_1save_1/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/b
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
save_1/Assign_10AssignConv2D_1/W/Adamsave_1/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save_1/Assign_11AssignConv2D_1/W/Adam_1save_1/RestoreV2:11*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
�
save_1/Assign_12Assign
Conv2D_1/bsave_1/RestoreV2:12*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_13AssignConv2D_1/b/Adamsave_1/RestoreV2:13*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_14AssignConv2D_1/b/Adam_1save_1/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
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
save_1/Assign_19AssignConv2D_2/b/Adamsave_1/RestoreV2:19*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_1/Assign_20AssignConv2D_2/b/Adam_1save_1/RestoreV2:20*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save_1/Assign_22AssignConv2D_3/W/Adamsave_1/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save_1/Assign_23AssignConv2D_3/W/Adam_1save_1/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save_1/Assign_24Assign
Conv2D_3/bsave_1/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@
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
Conv2D_4/Wsave_1/RestoreV2:27*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ *
use_locking(
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
Conv2D_4/bsave_1/RestoreV2:30*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
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
save_1/Assign_33AssignCrossentropy/Mean/moving_avgsave_1/RestoreV2:33*
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_34AssignFullyConnected/Wsave_1/RestoreV2:34*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
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
save_1/Assign_37AssignFullyConnected/bsave_1/RestoreV2:37*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*#
_class
loc:@FullyConnected/b
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
save_1/Assign_39AssignFullyConnected/b/Adam_1save_1/RestoreV2:39*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save_1/Assign_42AssignFullyConnected_1/W/Adam_1save_1/RestoreV2:42*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_1/Assign_43AssignFullyConnected_1/bsave_1/RestoreV2:43*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
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
save_1/Assign_46AssignGlobal_Stepsave_1/RestoreV2:46*
T0*
_class
loc:@Global_Step*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_47AssignTraining_stepsave_1/RestoreV2:47*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@Training_step
�
save_1/Assign_48Assignis_trainingsave_1/RestoreV2:48*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: 
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
save_1/Assign_50Assignval_losssave_1/RestoreV2:50*
use_locking(*
T0*
_class
loc:@val_loss*
validate_shape(*
_output_shapes
: 
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
shape: *
dtype0*
_output_shapes
: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
dtype0*
_output_shapes
: 
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
save_2/AssignAssignConv2D/Wsave_2/RestoreV2*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/W
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
Conv2D_1/Wsave_2/RestoreV2:2*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*
_class
loc:@Conv2D_1/W
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
Conv2D_2/bsave_2/RestoreV2:5*
T0*
_class
loc:@Conv2D_2/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_2/Assign_6Assign
Conv2D_3/Wsave_2/RestoreV2:6*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
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
Conv2D_4/Wsave_2/RestoreV2:8*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
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
save_2/Assign_10AssignFullyConnected/Wsave_2/RestoreV2:10*
use_locking(*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �
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
save_2/Assign_12AssignFullyConnected_1/Wsave_2/RestoreV2:12*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�
�
save_2/Assign_13AssignFullyConnected_1/bsave_2/RestoreV2:13*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@FullyConnected_1/b
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
save_3/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_3/SaveV2/tensor_namesConst*
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
save_3/control_dependencyIdentitysave_3/Const^save_3/SaveV2*
T0*
_class
loc:@save_3/Const*
_output_shapes
: 
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
!save_3/RestoreV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523

�
save_3/AssignAssignAccuracy/Mean/moving_avgsave_3/RestoreV2*
use_locking(*
T0*+
_class!
loc:@Accuracy/Mean/moving_avg*
validate_shape(*
_output_shapes
: 
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
save_3/Assign_8AssignConv2D/b/Adam_1save_3/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Conv2D/b
�
save_3/Assign_9Assign
Conv2D_1/Wsave_3/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save_3/Assign_10AssignConv2D_1/W/Adamsave_3/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save_3/Assign_11AssignConv2D_1/W/Adam_1save_3/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Conv2D_1/W*
validate_shape(*&
_output_shapes
: @
�
save_3/Assign_12Assign
Conv2D_1/bsave_3/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Conv2D_1/b*
validate_shape(*
_output_shapes
:@
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
save_3/Assign_17AssignConv2D_2/W/Adam_1save_3/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Conv2D_2/W*
validate_shape(*'
_output_shapes
:@�
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
Conv2D_3/Wsave_3/RestoreV2:21*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save_3/Assign_22AssignConv2D_3/W/Adamsave_3/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@
�
save_3/Assign_23AssignConv2D_3/W/Adam_1save_3/RestoreV2:23*
T0*
_class
loc:@Conv2D_3/W*
validate_shape(*'
_output_shapes
:�@*
use_locking(
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
save_3/Assign_25AssignConv2D_3/b/Adamsave_3/RestoreV2:25*
T0*
_class
loc:@Conv2D_3/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_3/Assign_26AssignConv2D_3/b/Adam_1save_3/RestoreV2:26*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Conv2D_3/b
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
save_3/Assign_28AssignConv2D_4/W/Adamsave_3/RestoreV2:28*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Conv2D_4/W
�
save_3/Assign_29AssignConv2D_4/W/Adam_1save_3/RestoreV2:29*
use_locking(*
T0*
_class
loc:@Conv2D_4/W*
validate_shape(*&
_output_shapes
:@ 
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
save_3/Assign_32AssignConv2D_4/b/Adam_1save_3/RestoreV2:32*
T0*
_class
loc:@Conv2D_4/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_33AssignCrossentropy/Mean/moving_avgsave_3/RestoreV2:33*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*/
_class%
#!loc:@Crossentropy/Mean/moving_avg
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
save_3/Assign_35AssignFullyConnected/W/Adamsave_3/RestoreV2:35*
T0*#
_class
loc:@FullyConnected/W*
validate_shape(*
_output_shapes
:	 �*
use_locking(
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
save_3/Assign_37AssignFullyConnected/bsave_3/RestoreV2:37*
T0*#
_class
loc:@FullyConnected/b*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save_3/Assign_42AssignFullyConnected_1/W/Adam_1save_3/RestoreV2:42*
T0*%
_class
loc:@FullyConnected_1/W*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
save_3/Assign_43AssignFullyConnected_1/bsave_3/RestoreV2:43*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
save_3/Assign_45AssignFullyConnected_1/b/Adam_1save_3/RestoreV2:45*
T0*%
_class
loc:@FullyConnected_1/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
save_3/Assign_48Assignis_trainingsave_3/RestoreV2:48*
T0
*
_class
loc:@is_training*
validate_shape(*
_output_shapes
: *
use_locking(
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
save_3/restore_allNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9"�"#
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

Conv2D/b:0"�	
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
FullyConnected/Relu:05]��0       ���_	j�ŀ��A*#

Loss t�?

Adam/Loss/raw��?`�Ld0       ���_	C�	ŀ��A*#

Loss�w�?

Adam/Loss/raw"&�?��q�0       ���_	jŀ��A*#

Loss���?

Adam/Loss/rawf4�?�
�0       ���_	�%ŀ��A*#

Loss9��?

Adam/Loss/rawȋ�?9��b0       ���_	:Bŀ��A*#

Loss�E�?

Adam/Loss/rawx#�?n^�0       ���_	�ŀ��A*#

Loss:1�?

Adam/Loss/raw9+�?����0       ���_	��!ŀ��A*#

Loss٭�?

Adam/Loss/raw�?���t0       ���_	XW#ŀ��A	*#

Loss��?

Adam/Loss/raw�z�?8h�0       ���_	�$ŀ��A
*#

Lossϓ�?

Adam/Loss/rawU��?(���0       ���_	>y)ŀ��A*#

LossmY�?

Adam/Loss/raw��?�Q�0       ���_	(I.ŀ��A*#

Loss��?

Adam/Loss/raw�k�?�F�0       ���_	��4ŀ��A*#

Lossw��?

Adam/Loss/raw�4�?,=�0       ���_	*6;ŀ��A*#

LossA��?

Adam/Loss/rawP�?'��30       ���_	��Aŀ��A*#

Loss��?

Adam/Loss/rawfp�?3g�0       ���_	��Gŀ��A*#

Loss��?

Adam/Loss/raw��?y20       ���_	�Nŀ��A*#

LossXǷ?

Adam/Loss/raw��?�r�#0       ���_	?tTŀ��A*#

LossN�?

Adam/Loss/rawYǻ?��Q�0       ���_	��Uŀ��A*#

Loss��?

Adam/Loss/raw��?�Ak0       ���_	@TWŀ��A*#

Loss(;�?

Adam/Loss/rawWQ�?�nn�0       ���_	 `\ŀ��A*#

Loss79�?

Adam/Loss/rawp_�?R��y0       ���_	�&aŀ��A*#

Loss��?

Adam/Loss/raw>q�?����0       ���_	��eŀ��A*#

Loss���?

Adam/Loss/raw���?��,�0       ���_	m�jŀ��A*#

Loss���?

Adam/Loss/rawT�?8�0       ���_	�Boŀ��A*#

LossD��?

Adam/Loss/rawip�?OzH0       ���_	*tŀ��A*#

Loss7�?

Adam/Loss/rawD�?9�#0       ���_	K�xŀ��A*#

Loss_ �?

Adam/Loss/rawF��?kD�0       ���_	�X}ŀ��A*#

Loss��?

Adam/Loss/raw��?���Q0       ���_	m�~ŀ��A*#

Lossߣ?

Adam/Loss/raw��?S�p�0       ���_	�!�ŀ��A*#

Loss^�?

Adam/Loss/raw&�?�g͝0       ���_	@Q�ŀ��A*#

LossT\�?

Adam/Loss/rawl�^?I�]w0       ���_	�(�ŀ��A *#

Loss'Y�?

Adam/Loss/raw"ߔ?+p�d0       ���_	2Ǝŀ��A!*#

Loss�Ǐ?

Adam/Loss/raw���?a�/�0       ���_	9~�ŀ��A"*#

Loss��?

Adam/Loss/rawr0�?�p��0       ���_	~��ŀ��A#*#

Loss
:�?

Adam/Loss/rawcz�?�i|0       ���_	�C�ŀ��A$*#

Loss���?

Adam/Loss/rawH�L?@��0       ���_	T�ŀ��A%*#

Loss�m�?

Adam/Loss/raw+I@� �0       ���_	aƦŀ��A&*#

Loss0��?

Adam/Loss/rawg�?<��0       ���_	DP�ŀ��A'*#

Loss{ġ?

Adam/Loss/raw�e?�xe)0       ���_	��ŀ��A(*#

Loss��?

Adam/Loss/raw��a?�{�D0       ���_		l�ŀ��A)*#

Loss%��?

Adam/Loss/rawHY�?�U�20       ���_	�*�ŀ��A**#

Losss��?

Adam/Loss/raw�=b?P��g0       ���_	cзŀ��A+*#

Loss9q�?

Adam/Loss/rawn�v?Uu0       ���_	a��ŀ��A,*#

Loss���?

Adam/Loss/raw�w?')�Z0       ���_	���ŀ��A-*#

Lossd�?

Adam/Loss/raw@�_?=��0       ���_	�7�ŀ��A.*#

Loss���?

Adam/Loss/raw�w?|1��0       ���_	��ŀ��A/*#

LossV��?

Adam/Loss/raw�	�?Um�0       ���_	���ŀ��A0*#

Loss枓?

Adam/Loss/raw��j?����0       ���_	���ŀ��A1*#

Loss�Վ?

Adam/Loss/raw{i�?����0       ���_	�)�ŀ��A2*#

LossGR�?

Adam/Loss/raw�̊?�~?0       ���_	���ŀ��A3*#

Loss�z�?

Adam/Loss/raw	}?��0       ���_	9��ŀ��A4*#

Loss��?

Adam/Loss/raw��q?��G0       ���_	Qh�ŀ��A5*#

Loss���?

Adam/Loss/raw��?�D�0       ���_	�*�ŀ��A6*#

Lossc܉?

Adam/Loss/raw��y?�j0       ���_	��ŀ��A7*#

Loss�?

Adam/Loss/raw���?U^0       ���_	�:�ŀ��A8*#

Lossjˇ?

Adam/Loss/raw��X?o�0       ���_	[A�ŀ��A9*#

Lossa��?

Adam/Loss/raw��@�3(60       ���_	���ŀ��A:*#

Loss0�?

Adam/Loss/raw��t?ŷ$0       ���_	�tƀ��A;*#

Loss'0�?

Adam/Loss/raw�_t?27{�0       ���_	4�ƀ��A<*#

Loss�?

Adam/Loss/raw�a?࿁R0       ���_	Z�ƀ��A=*#

Loss�č?

Adam/Loss/raw�v�?�a�O0       ���_	oƀ��A>*#

LossS�?

Adam/Loss/rawσ?�F��0       ���_	6�ƀ��A?*#

Loss�c�?

Adam/Loss/raw�m?�?��0       ���_	��ƀ��A@*#

Loss�S�?

Adam/Loss/raw��?)�m�0       ���_	*vƀ��AA*#

Lossz�?

Adam/Loss/raw�m?�i6�0       ���_	�?#ƀ��AB*#

Loss
��?

Adam/Loss/raw��?%���0       ���_	T�'ƀ��AC*#

Loss���?

Adam/Loss/rawr�6@�6�0       ���_	��-ƀ��AD*#

Loss��?

Adam/Loss/rawoIn?��0       ���_	�0ƀ��AE*#

Loss�֞?

Adam/Loss/raw�n?d�V0       ���_	�1ƀ��AF*#

LossoL�?

Adam/Loss/raw��v?��ɂ0       ���_	�$8ƀ��AG*#

Loss ǖ?

Adam/Loss/raw�wX?�$^0       ���_	s�<ƀ��AH*#

Loss���?

Adam/Loss/raw���?; ݯ0       ���_	�Aƀ��AI*#

Lossd�?

Adam/Loss/raw�j�?x�hd0       ���_	x�Fƀ��AJ*#

Loss��?

Adam/Loss/raw.[�?�s�0       ���_	�rKƀ��AK*#

Lossn��?

Adam/Loss/rawK�z?)�d0       ���_	��Qƀ��AL*#

Loss�f�?

Adam/Loss/rawFn[?�|�x0       ���_	EJWƀ��AM*#

Loss���?

Adam/Loss/raw��@#�w0       ���_	�,\ƀ��AN*#

Loss���?

Adam/Loss/rawll�?��V0       ���_	��]ƀ��AO*#

Loss�F�?

Adam/Loss/raw#c�?9f�0       ���_	��^ƀ��AP*#

Loss�&�?

Adam/Loss/raw�՞?���U0       ���_	N�dƀ��AQ*#

Lossp�?

Adam/Loss/rawjr?VC�u0       ���_	��iƀ��AR*#

Loss�S�?

Adam/Loss/rawl��?y}�0       ���_	P�nƀ��AS*#

LossZ'�?

Adam/Loss/raw,T?$E_0       ���_	�]sƀ��AT*#

Loss:��?

Adam/Loss/raw��O?�+Vb0       ���_	�'xƀ��AU*#

Loss?

Adam/Loss/raw˙r?�|�0       ���_	�}ƀ��AV*#

Loss�3�?

Adam/Loss/raw��k?.040       ���_	��ƀ��AW*#

LossA��?

Adam/Loss/rawk�?���30       ���_	цƀ��AX*#

LossE�?

Adam/Loss/raw��}?C�ʑ0       ���_	�L�ƀ��AY*#

Loss_��?

Adam/Loss/raw>Dn?]w	=0       ���_	㋉ƀ��AZ*#

LossY��?

Adam/Loss/rawni\?���0       ���_	�n�ƀ��A[*#

Loss��?

Adam/Loss/rawT�O?C*��0       ���_	��ƀ��A\*#

LossZ8�?

Adam/Loss/raw��h?�8�0       ���_	�W�ƀ��A]*#

Loss��?

Adam/Loss/raw(�_?�T�50       ���_	�L�ƀ��A^*#

Loss��?

Adam/Loss/raw�m?�sj�0       ���_	r��ƀ��A_*#

LossX�?

Adam/Loss/raw�z0?xIb�0       ���_	�)�ƀ��A`*#

Loss~r?

Adam/Loss/raw���?���0       ���_	���ƀ��Aa*#

Loss8��?

Adam/Loss/raw�@��0       ���_	�"�ƀ��Ab*#

Loss�C�?

Adam/Loss/raw>BU?nDA0       ���_	�¸ƀ��Ac*#

Loss�f�?

Adam/Loss/raw��?�^��0       ���_	�W�ƀ��Ad*#

Loss���?

Adam/Loss/rawSk?9x0       ���_	�w�ƀ��Ae*#

Loss�L�?

Adam/Loss/rawz46?^-��0       ���_	ɕ�ƀ��Af*#

Loss�(y?

Adam/Loss/raw�;?/�!P0       ���_	W��ƀ��Ag*#

Losse�r?

Adam/Loss/raw��I?�b��0       ���_	zR�ƀ��Ah*#

Lossj�n?

Adam/Loss/raw��?��T90       ���_	���ƀ��Ai*#

Lossxf?

Adam/Loss/rawXL8?]��0       ���_	z�ƀ��Aj*#

Lossra?

Adam/Loss/rawo<?����0       ���_	K �ƀ��Ak*#

Loss��]?

Adam/Loss/raw�m7@J�0       ���_	�D�ƀ��Al*#

Losswx�?

Adam/Loss/raw�~#? �cd0       ���_	���ƀ��Am*#

Loss���?

Adam/Loss/raw�1�?��J@0       ���_	f��ƀ��An*#

Loss'��?

Adam/Loss/raw��p?�R�U0       ���_	K"�ƀ��Ao*#

Lossf�?

Adam/Loss/raw� �?�/�D0       ���_	�,�ƀ��Ap*#

Loss���?

Adam/Loss/raw���?k�{0       ���_	���ƀ��Aq*#

LossH�?

Adam/Loss/raw�x{?�&J0       ���_	2��ƀ��Ar*#

Lossp�?

Adam/Loss/raw�r?M�0       ���_	��ǀ��As*#

Loss7f�?

Adam/Loss/rawL�<?%��
0       ���_	P�ǀ��At*#

Loss7�}?

Adam/Loss/raw�T?��0       ���_	�ǀ��Au*#

Loss��y?

Adam/Loss/raw��@5O9�0       ���_	T ǀ��Av*#

Lossqe�?

Adam/Loss/raw�?�1�0       ���_	'�ǀ��Aw*#

Lossy�?

Adam/Loss/raws�f?����0       ���_	0eǀ��Ax*#

Lossf�?

Adam/Loss/raw�]?��0       ���_	u�ǀ��Ay*#

Lossj��?

Adam/Loss/raw��,?�2�<0       ���_	�+!ǀ��Az*#

Lossi/~?

Adam/Loss/rawaK?�8�0       ���_	�,&ǀ��A{*#

Loss�y?

Adam/Loss/raw��??]�� 0       ���_	6�*ǀ��A|*#

Loss�Ys?

Adam/Loss/rawi�;?1ŵ&0       ���_	k�/ǀ��A}*#

LossP�m?

Adam/Loss/raw�?�D)0       ���_	��4ǀ��A~*#

Loss��e?

Adam/Loss/raw	{>?����0       ���_	��9ǀ��A*#

Loss��a?

Adam/Loss/raw=�D@�#eO1       ����	�>ǀ��A�*#

Loss���?

Adam/Loss/rawr%?��5�1       ����	�@ǀ��A�*#

Loss��?

Adam/Loss/raw::u?!b=/1       ����	��Aǀ��A�*#

Lossم?

Adam/Loss/rawp@?nn1       ����	 Gǀ��A�*#

Loss�?

Adam/Loss/raw��W?Fa�n1       ����	f�Kǀ��A�*#

Loss��?

Adam/Loss/raw�щ?<T��1       ����	��Pǀ��A�*#

Loss6׀?

Adam/Loss/raw��a?�U��1       ����	�kUǀ��A�*#

Loss�~?

Adam/Loss/raw1?���1       ����	$Zǀ��A�*#

Loss*�r?

Adam/Loss/raw��@?p�91       ����	��^ǀ��A�*#

LossX�m?

Adam/Loss/raw��2?��Q1       ����	D�cǀ��A�*#

Loss.�g?

Adam/Loss/rawazP@Sy�r1       ����	��hǀ��A�*#

Loss���?

Adam/Loss/raw �,?�W��1       ����	�:jǀ��A�*#

Loss��?

Adam/Loss/raw@�B?K�=j1       ����	�~kǀ��A�*#

Loss��?

Adam/Loss/rawW�,?�sw1       ����	��pǀ��A�*#

Loss�т?

Adam/Loss/raw�?&?�쿉1       ����	>;uǀ��A�*#

Loss�|?

Adam/Loss/raw�/?��u�1       ����	zǀ��A�*#

Loss�5r?

Adam/Loss/rawvD
?��|x1       ����	d�~ǀ��A�*#

Loss��g?

Adam/Loss/raw��L?��81       ����	΃ǀ��A�*#

Loss�e?

Adam/Loss/rawZ��>T`T�1       ����	]وǀ��A�*#

Loss]�Z?

Adam/Loss/raw�?Q��51       ����	���ǀ��A�*#

LossN�R?

Adam/Loss/raw�KE@���1       ����	,��ǀ��A�*#

LossQ�?

Adam/Loss/raw��6?x�s1       ����	�'�ǀ��A�*#

Loss�?

Adam/Loss/raw^��>��	�1       ����	�W�ǀ��A�*#

Loss��t?

Adam/Loss/raw�K�>��$1       ����	2�ǀ��A�*#

LossW�f?

Adam/Loss/raw�n?�g�T1       ����	��ǀ��A�*#

Lossv�^?

Adam/Loss/raw��
?t"x�1       ����	�ݣǀ��A�*#

LossJKV?

Adam/Loss/raw���>np��1       ����	�Шǀ��A�*#

Loss�'M?

Adam/Loss/raw�/�>��M1       ����	u��ǀ��A�*#

Loss5fE?

Adam/Loss/raw(?<]��1       ����	�g�ǀ��A�*#

Lossc�>?

Adam/Loss/raw)?�'4�1       ����	�0�ǀ��A�*#

Loss��9?

Adam/Loss/raw?�T@C�A�1       ����	��ǀ��A�*#

LossdH|?

Adam/Loss/rawiN?���1       ����	ǀ��A�*#

Lossd�r?

Adam/Loss/raw���>�a��1       ����	�Ӿǀ��A�*#

Loss�if?

Adam/Loss/raw _�>gmˏ1       ����	���ǀ��A�*#

LossI�V?

Adam/Loss/raw��?ݭ1       ����	��ǀ��A�*#

LossU�P?

Adam/Loss/raw%A'?1U�1       ����	�w�ǀ��A�*#

Lossj�L?

Adam/Loss/raw�>$*F1       ����	�]�ǀ��A�*#

Loss7D?

Adam/Loss/raw��	?ٯ��1       ����	cD�ǀ��A�*#

Loss�Z>?

Adam/Loss/raw�?�_��1       ����	��ǀ��A�*#

Loss8�9?

Adam/Loss/raw3?��o1       ����	��ǀ��A�*#

Loss7�6?

Adam/Loss/raw0Wy@����1       ����	b��ǀ��A�*#

Loss	
�?

Adam/Loss/rawy?����1       ����	��ǀ��A�*#

Loss{?

Adam/Loss/rawjB�>�p71       ����	�2�ǀ��A�*#

LossTUn?

Adam/Loss/raw��>���J1       ����	��ǀ��A�*#

Loss�a?

Adam/Loss/rawI@�>�ta_1       ����	u�ǀ��A�*#

Loss��V?

Adam/Loss/raw��>��|1       ����	���ǀ��A�*#

Loss��L?

Adam/Loss/rawĺ�>H;�1       ����	 ��ǀ��A�*#

LossݐC?

Adam/Loss/raw|��>�X N1       ����	�] Ȁ��A�*#

Loss�};?

Adam/Loss/raw�ݧ>;�J 1       ����	Q5Ȁ��A�*#

Loss�"1?

Adam/Loss/rawd9�>���1       ����	s/
Ȁ��A�*#

Loss�{)?

Adam/Loss/rawh��@����1       ����	_Ȁ��A�*#

Lossp�?

Adam/Loss/raw@�?#�p1       ����	�uȀ��A�*#

LossPDt?

Adam/Loss/raw�(/?tW�1       ����	��Ȁ��A�*#

Loss([m?

Adam/Loss/raw��?:&�N1       ����	�>Ȁ��A�*#

Loss�d?

Adam/Loss/raw4��>D�*	1       ����	@Ȁ��A�*#

Loss�jU?

Adam/Loss/raw3�?YW�1       ����	
!Ȁ��A�*#

Loss'M?

Adam/Loss/raw�B�>$�v1       ����	��%Ȁ��A�*#

Loss*0B?

Adam/Loss/raw��>�P�K1       ����	c�*Ȁ��A�*#

Loss�l5?

Adam/Loss/rawY�?��w�1       ����	�x/Ȁ��A�*#

Loss8*3?

Adam/Loss/rawt��>5���1       ����	;4Ȁ��A�*#

LossE
.?

Adam/Loss/raw7|@�LTo1       ����	Z9Ȁ��A�*#

Loss�?

Adam/Loss/rawM?E+�1       ����	��:Ȁ��A�*#

Loss�.u?

Adam/Loss/rawW��?��,1       ����	c�;Ȁ��A�*#

Losst�}?

Adam/Loss/rawCŀ?[�(B1       ����	�,AȀ��A�*#

Loss�S~?

Adam/Loss/raw��^?�W!�1       ����	��EȀ��A�*#

LossU%{?

Adam/Loss/raw1PW?&�u�1       ����	]�JȀ��A�*#

Loss�w?

Adam/Loss/raw���>��V�1       ����	0�OȀ��A�*#

Loss��j?

Adam/Loss/raw�] ?Z�@�1       ����	��TȀ��A�*#

LossG?`?

Adam/Loss/raw���>��ݑ1       ����	�YȀ��A�*#

Loss�RU?

Adam/Loss/raw56?>=�r1       ����	F}^Ȁ��A�*#

Loss�5R?

Adam/Loss/raw�� @֩:H1       ����	JcȀ��A�*#

Lossj}?

Adam/Loss/raw��?V��N1       ����	l�dȀ��A�*#

Loss>�r?

Adam/Loss/raw&�>^0�-1       ����	�(fȀ��A�*#

Loss7a?

Adam/Loss/rawҹ?>��"�1       ����	�RkȀ��A�*#

Loss�|O?

Adam/Loss/raw�p�>\]PV1       ����	]:pȀ��A�*#

Loss��G?

Adam/Loss/rawU�>�ö�1       ����	�8uȀ��A�*#

Loss�=?

Adam/Loss/raw��>��]�1       ����	zȀ��A�*#

Loss��6?

Adam/Loss/raw4�>a��{1       ����	G�~Ȁ��A�*#

Loss;g0?

Adam/Loss/raw�w?�11       ����	<ۃȀ��A�*#

Loss��.?

Adam/Loss/raw~F�>=X
\1       ����	��Ȁ��A�*#

Loss��$?

Adam/Loss/raw��l@A��1       ����	ɭ�Ȁ��A�*#

Loss�r?

Adam/Loss/raw��>Yo��1       ����	$�Ȁ��A�*#

Loss�lg?

Adam/Loss/raw&�
?��5�1       ����	�\�Ȁ��A�*#

Loss'^?

Adam/Loss/raw��>���1       ����	᳕Ȁ��A�*#

LosstS?

Adam/Loss/raw�n>��T1       ����	%��Ȁ��A�*#

LossO�C?

Adam/Loss/raw���>h�!�1       ����	Tp�Ȁ��A�*#

Loss[*9?

Adam/Loss/raw
�>I�1       ����	�<�Ȁ��A�*#

Loss��-?

Adam/Loss/raw���>|Q��1       ����	��Ȁ��A�*#

LossM�#?

Adam/Loss/rawI7�>̴:1       ����	�Ȁ��A�*#

Losso�?

Adam/Loss/raw�*�>���1       ����	��Ȁ��A�*#

Loss��?

Adam/Loss/rawB�{@z6r1       ����	p��Ȁ��A�*#

Lossڰi?

Adam/Loss/rawT1�>h�Lj1       ����	�g�Ȁ��A�*#

Loss;{Y?

Adam/Loss/raw���=x���1       ����	i��Ȁ��A�*#

Loss�E?

Adam/Loss/raw�|�=�j�1       ����	��Ȁ��A�*#

Loss��3?

Adam/Loss/raw3��>�~o1       ����	:y�Ȁ��A�*#

Loss..*?

Adam/Loss/raw*�>�.��1       ����	�Q�Ȁ��A�*#

Loss�� ?

Adam/Loss/raw��Q>�W1       ����	2�Ȁ��A�*#

Loss\?

Adam/Loss/raw��>W?�1       ����	*��Ȁ��A�*#

Loss�?

Adam/Loss/raw���>��n1       ����	V��Ȁ��A�*#

Losslt?

Adam/Loss/raw:��>,6l�1       ����	?�Ȁ��A�*#

Loss�?

Adam/Loss/raw���@��V91       ����	U0�Ȁ��A�*#

Loss��y?

Adam/Loss/rawRwB?Y7,�1       ����	���Ȁ��A�*#

Loss�Rt?

Adam/Loss/raw 1?��1       ����	�W�Ȁ��A�*#

LossW�j?

Adam/Loss/raw�I�>fw�a1       ����	�~�Ȁ��A�*#

Loss��\?

Adam/Loss/rawX?�>���1       ����	:�Ȁ��A�*#

Loss�O?

Adam/Loss/raw��r>�-81       ����	�$�Ȁ��A�*#

LossTl@?

Adam/Loss/raw�>���1       ����	g�Ȁ��A�*#

Lossl1?

Adam/Loss/raw�k?�̿m1       ����	�(�Ȁ��A�*#

Loss�.?

Adam/Loss/raw6��>���1       ����	�;ɀ��A�*#

Losst�(?

Adam/Loss/raw��f>OL��1       ����	�%ɀ��A�*#

Loss��?

Adam/Loss/raw�
I@�1       ����	�ɀ��A�*#

Loss�m^?

Adam/Loss/raw4�>���
1       ����	?vɀ��A�*#

Loss��Q?

Adam/Loss/raw�(?mV��1       ����	�ɀ��A�*#

Loss��K?

Adam/Loss/raw���>���U1       ����	�ɀ��A�*#

Loss�IA?

Adam/Loss/raw��>_K��1       ����	��ɀ��A�*#

Loss�5?

Adam/Loss/rawf�>��01       ����	V�ɀ��A�*#

Loss}u,?

Adam/Loss/raw2�g>f4�:1       ����	��#ɀ��A�*#

LossX !?

Adam/Loss/raw
!�>���1       ����	$�(ɀ��A�*#

Loss��?

Adam/Loss/rawK�>^��1       ����	�-ɀ��A�*#

LosspW?

Adam/Loss/raw���>��1       ����	�q2ɀ��A�*#

Loss�=	?

Adam/Loss/raw��d@��'}1       ����	W7ɀ��A�*#

LossZW?

Adam/Loss/raw�A�>��o�1       ����	#�8ɀ��A�*#

Loss7H?

Adam/Loss/raw�>�1       ����	^,:ɀ��A�*#

Loss�08?

Adam/Loss/raw���=���+1       ����	=F?ɀ��A�*#

LossM�(?

Adam/Loss/raw�3v>
&�1       ����	�8Dɀ��A�*#

Loss�?

Adam/Loss/raw��w>d�21       ����	2<Iɀ��A�*#

Loss�x?

Adam/Loss/raw�>=>w�Ws1       ����	80Nɀ��A�*#

Loss�Z
?

Adam/Loss/rawC�|>���!1       ����	�Sɀ��A�*#

Loss��?

Adam/Loss/raw���>����1       ����	�Xɀ��A�*#

Loss���>

Adam/Loss/raw��t>�P �1       ����	K]ɀ��A�*#

Loss�>

Adam/Loss/raw���@�ᦟ1       ����	�bɀ��A�*#

Loss�8]?

Adam/Loss/raw��]>ֺZ�1       ����	g�cɀ��A�*#

Lossw�L?

Adam/Loss/raw���>(�E�1       ����	&�dɀ��A�*#

Loss�B?

Adam/Loss/raw֞�>��"L1       ����	�iɀ��A�*#

Loss��5?

Adam/Loss/raw�#�>�b��1       ����	Z�nɀ��A�*#

Loss��-?

Adam/Loss/raw垛>�~�1       ����	��sɀ��A�*#

Loss-I$?

Adam/Loss/raw<F.>��.%1       ����	��xɀ��A�*#

Loss�6?

Adam/Loss/raw8��=D(C1       ����	Ԟ}ɀ��A�*#

Loss_�?

Adam/Loss/rawl�>��<11       ����	��ɀ��A�*#

Loss#3?

Adam/Loss/raw���>>H}'1       ����	�q�ɀ��A�*#

Lossg�>

Adam/Loss/raw^�@S��u1       ����	/��ɀ��A�*#

Lossa`?

Adam/Loss/rawF6�>O���1       ����	��ɀ��A�*#

Loss�3Q?

Adam/Loss/raw� �=�J71       ����	@�ɀ��A�*#

Loss �=?

Adam/Loss/raw(s=]<�
1       ����	9��ɀ��A�*#

Loss��+?

Adam/Loss/rawW�D>���.1       ����	��ɀ��A�*#

Loss��?

Adam/Loss/raw.[�=㾡�1       ����	��ɀ��A�*#

Lossd�?

Adam/Loss/rawf�9>$X��1       ����	�|�ɀ��A�*#

Loss6�?

Adam/Loss/raw��=�Nk�1       ����	=a�ɀ��A�*#

Loss���>

Adam/Loss/raw��4>��:C1       ����	�2�ɀ��A�*#

Lossq��>

Adam/Loss/rawUl@>�.�1       ����	��ɀ��A�*#

Loss���>

Adam/Loss/raw��@S�-)1       ����	a߶ɀ��A�*#

Loss~4X?

Adam/Loss/raw��=�u�1       ����	iq�ɀ��A�*#

Loss��E?

Adam/Loss/raw�~:>:��1       ����	��ɀ��A�*#

Loss}6?

Adam/Loss/rawi��=�i��1       ����	d�ɀ��A�*#

Loss��&?

Adam/Loss/raw���=�4D<1       ����	��ɀ��A�*#

Loss2'?

Adam/Loss/rawz��=K�d�1       ����	���ɀ��A�*#

Loss��?

Adam/Loss/raw���=y.ډ1       ����	���ɀ��A�*#

Loss���>

Adam/Loss/raw�|>}m�1       ����	���ɀ��A�*#

Loss�d�>

Adam/Loss/raw���=Ճ*�1       ����	���ɀ��A�*#

Loss�S�>

Adam/Loss/raw8��=T���1       ����	��ɀ��A�*#

Lossz��>

Adam/Loss/raw�ъ@j2��1       ����	���ɀ��A�*#

Loss�sI?

Adam/Loss/rawf5>N{Y�1       ����	v�ɀ��A�*#

Loss��9?

Adam/Loss/raw�d\=k	Ҳ1       ����	`��ɀ��A�*#

Loss�(?

Adam/Loss/raw�ls<�e�Y1       ����	� �ɀ��A�*#

Loss�%?

Adam/Loss/raw7�4= �t�1       ����	B%�ɀ��A�*#

Loss�
?

Adam/Loss/rawT�>B��1       ����	��ɀ��A�*#

Loss�0�>

Adam/Loss/raw~>(ʟ1       ����	.��ɀ��A�*#

Losss,�>

Adam/Loss/raw���=��1       ����	��ɀ��A�*#

Loss�_�>

Adam/Loss/raw<��=��j�1       ����	J�ʀ��A�*#

Lossk��>

Adam/Loss/raw�`>˝e71       ����	\�ʀ��A�*#

Loss�"�>

Adam/Loss/rawo$�@��H~1       ����	Otʀ��A�*#

Loss J?

Adam/Loss/rawF� >B��G1       ����	� ʀ��A�*#

Loss��9?

Adam/Loss/raw��=��o1       ����	�Jʀ��A�*#

Loss��)?

Adam/Loss/raw�E=�v��1       ����	7�ʀ��A�*#

Loss��?

Adam/Loss/raw��=�dʯ1       ����	��ʀ��A�*#

LossU$?

Adam/Loss/raw9=>�`��1       ����	��ʀ��A�*#

Lossn�?

Adam/Loss/rawe�:>A\��1       ����	I$ʀ��A�*#

Loss��>

Adam/Loss/raww�h=���_1       ����	w�(ʀ��A�*#

Loss�w�>

Adam/Loss/raw��=�e8>1       ����	|�-ʀ��A�*#

Loss�_�>

Adam/Loss/rawe�=h�gV1       ����	/�2ʀ��A�*#

Loss�|�>

Adam/Loss/raw$��@��1       ����	.�7ʀ��A�*#

Loss0<D?

Adam/Loss/raw�4�=� �1       ����	nQ9ʀ��A�*#

Loss�V3?

Adam/Loss/raw��>��t�1       ����	:�:ʀ��A�*#

Loss��$?

Adam/Loss/raw 7W=�O>�1       ����	��?ʀ��A�*#

Loss��?

Adam/Loss/rawC��=�I(1       ����	��Dʀ��A�*#

Loss��?

Adam/Loss/rawXV�=��w1       ����	��Iʀ��A�*#

Loss��>

Adam/Loss/rawK&]=칠�1       ����	�^Nʀ��A�*#

Loss���>

Adam/Loss/raw\��<=5�1       ����	d"Sʀ��A�*#

Loss�S�>

Adam/Loss/raw�]�=�5��1       ����	�gXʀ��A�*#

Loss��>

Adam/Loss/raw�r;=�mR1       ����	�m]ʀ��A�*#

Loss��>

Adam/Loss/rawl%�@��@�1       ����	�tbʀ��A�*#

Loss��^?

Adam/Loss/raw�>H=�a:1       ����	dʀ��A�*#

Losss�I?

Adam/Loss/raw��<"M�1       ����	�ieʀ��A�*#

Loss$16?

Adam/Loss/rawƺ�<��M1       ����	Ɔjʀ��A�*#

Loss6$?

Adam/Loss/raw���=2���1       ����	�loʀ��A�*#

Loss�1?

Adam/Loss/raw��=(C1       ����	�mtʀ��A�*#

Loss��?

Adam/Loss/raw�L�=��Y1       ����	J@yʀ��A�*#

Loss���>

Adam/Loss/raww�=%Oz1       ����	�V~ʀ��A�*#

Lossw��>

Adam/Loss/rawL|B=�ݏ1       ����	(e�ʀ��A�*#

Loss�h�>

Adam/Loss/raw��=P�˥1       ����	�P�ʀ��A�*#

Loss�=�>

Adam/Loss/rawM��@^� 1       ����	XS�ʀ��A�*#

Loss�uR?

Adam/Loss/raw���=nc�1       ����	a�ʀ��A�*#

Lossg??

Adam/Loss/rawq\�<�M�1       ����	�9�ʀ��A�*#

Loss��,?

Adam/Loss/raw5��<���|1       ����	���ʀ��A�*#

Loss��?

Adam/Loss/rawɉL=�3u�1       ����	���ʀ��A�*#

Lossr�?

Adam/Loss/raw6��=�H��1       ����	���ʀ��A�*#

Loss��?

Adam/Loss/raw���=愢�1       ����	J��ʀ��A�*#

Loss N�>

Adam/Loss/raw�KL=��`�1       ����	�y�ʀ��A�*#

Loss���>

Adam/Loss/rawv��={���1       ����	�[�ʀ��A�*#

Loss��>

Adam/Loss/raw$u�=^�1       ����	E-�ʀ��A�*#

Loss�E�>

Adam/Loss/rawbm8=�7I1       ����	&�ʀ��A�*#

Loss�>

Adam/Loss/raw�+�<0��1       ����	꯹ʀ��A�*#

Loss˨�>

Adam/Loss/raw�H�;��ڇ1       ����	��ʀ��A�*#

Loss���>

Adam/Loss/raw�#V;YqQ1       ����	y#�ʀ��A�*#

Loss��y>

Adam/Loss/raw��I=��;1       ����	��ʀ��A�*#

Loss�e>

Adam/Loss/raw�w>�!p?1       ����	�ʀ��A�*#

Loss\>

Adam/Loss/raw��</�6�1       ����	6�ʀ��A�*#

Loss��G>

Adam/Loss/raw �=�xs1       ����	T�ʀ��A�*#

Loss�/7>

Adam/Loss/raw�J{=���1       ����	o'�ʀ��A�*#

Loss8&+>

Adam/Loss/raw!6!<j��1       ����	���ʀ��A�*#

Loss�
>

Adam/Loss/rawl�@~;��1       ����	i��ʀ��A�*#

Loss��>?

Adam/Loss/raw�3=�<:l1       ����	i�ʀ��A�*#

Loss��,?

Adam/Loss/rawA�<]�;�1       ����	P��ʀ��A�*#

Loss1�?

Adam/Loss/raw�3<���1       ����	���ʀ��A�*#

LossM�?

Adam/Loss/raw�=���1       ����	.��ʀ��A�*#

LossRO ?

Adam/Loss/raw4�T=SM��1       ����	9��ʀ��A�*#

Lossw��>

Adam/Loss/raw��{<��g1       ����	]��ʀ��A�*#

Lossi�>

Adam/Loss/rawb�=W;=!1       ����	���ʀ��A�*#

Lossܒ�>

Adam/Loss/raw��?>�7�1       ����	�Yˀ��A�*#

Loss���>

Adam/Loss/rawLd&>����1       ����	�6ˀ��A�*#

LossY�>

Adam/Loss/raw	��@
��1       ����	ˀ��A�*#

LossJk8?

Adam/Loss/raw.C�=�]A^1       ����	�ˀ��A�*#

Loss3�'?

Adam/Loss/raw7�<��z1       ����	)ˀ��A�*#

Lossf�?

Adam/Loss/raw���<с�n1       ����	��ˀ��A�*#

Loss�U	?

Adam/Loss/rawi?=T�� 1       ����	1{ˀ��A�*#

Loss�>

Adam/Loss/rawj9�=s���1       ����	�eˀ��A�*#

Loss2�>

Adam/Loss/raw|8=�)1       ����	X$ˀ��A�*#

LossS��>

Adam/Loss/raw��G>�1       ����	�2)ˀ��A�*#

Loss���>

Adam/Loss/raw�/�=�D�1       ����	#�-ˀ��A�*#

Losssm�>

Adam/Loss/raw �#=��81       ����	��2ˀ��A�*#

LossjT�>

Adam/Loss/raw0�p@��\�1       ����	48ˀ��A�*#

Lossެ*?

Adam/Loss/raw�Q>�Z��1       ����	ÿ9ˀ��A�*#

Loss�0?

Adam/Loss/raw+ψ>�I1       ����	��:ˀ��A�*#

Loss�O?

Adam/Loss/raw��=EOq�1       ����	"5@ˀ��A�*#

LossO�?

Adam/Loss/raw ֘=�z�;1       ����	YEˀ��A�*#

Loss���>

Adam/Loss/raw�1�=�Gy1       ����	.Jˀ��A�*#

Loss{C�>

Adam/Loss/raw���=21       ����	Oˀ��A�*#

Loss���>

Adam/Loss/raw���=���1       ����	Tˀ��A�*#

Loss���>

Adam/Loss/raw�X�=0�Ґ1       ����	Yˀ��A�*#

LossTƲ>

Adam/Loss/raw��I>K_Y1       ����	��]ˀ��A�*#

Lossm��>

Adam/Loss/rawNق@���1       ����	�bˀ��A�*#

Loss�5?

Adam/Loss/raw��>��߬1       ����	+Pdˀ��A�*#

Loss�&?

Adam/Loss/raw �>>Ƕ�1       ����	��eˀ��A�*#

Losso?

Adam/Loss/raw��M>K���1       ����	i�jˀ��A�*#

Loss��?

Adam/Loss/raw]ݼ=�s[�1       ����	a�oˀ��A�*#

LossfQ?

Adam/Loss/rawy�=��Y^1       ����	{Jtˀ��A�*#

Lossi^�>

Adam/Loss/raw�SG=L	q1       ����	�.yˀ��A�*#

Loss�R�>

Adam/Loss/raw�;�=��D�1       ����	V~ˀ��A�*#

Lossm2�>

Adam/Loss/raw}��<���1       ����	��ˀ��A�*#

Loss��>

Adam/Loss/rawz�i>k!�Q1       ����	��ˀ��A�*#

Loss�&�>

Adam/Loss/raw��@v8��1       ����	$ьˀ��A�*#

Loss�@?

Adam/Loss/raw՛�=��1       ����	Y�ˀ��A�*#

Lossby/?

Adam/Loss/raw�,�=>d1       ����	���ˀ��A�*#

Loss ! ?

Adam/Loss/rawKB�<~��`1       ����	m��ˀ��A�*#

Loss��?

Adam/Loss/rawX�Z=���L1       ����	��ˀ��A�*#

Lossb�?

Adam/Loss/raw�^�<v]p�1       ����	<��ˀ��A�*#

Loss{��>

Adam/Loss/raw�=G�E�1       ����	}�ˀ��A�*#

Loss���>

Adam/Loss/rawL8j=$�C[1       ����	�Ũˀ��A�*#

Loss
��>

Adam/Loss/raw�G =}�X1       ����	���ˀ��A�*#

Loss"��>

Adam/Loss/raw�{W=R1       ����	���ˀ��A�*#

Loss�x�>

Adam/Loss/raw��@���1       ����	<��ˀ��A�*#

LossKJ?

Adam/Loss/raw�0=0�H�1       ����	�$�ˀ��A�*#

LossL�6?

Adam/Loss/raw��<S|qg1       ����	uv�ˀ��A�*#

Loss{8%?

Adam/Loss/raws�<N�� 1       ����	ע�ˀ��A�*#

Lossd ?

Adam/Loss/raw�1=��xq1       ����	ͭ�ˀ��A�*#

Loss�R?

Adam/Loss/raw���=G�o�1       ����	8��ˀ��A�*#

LossSF�>

Adam/Loss/rawr,==���,1       ����	�r�ˀ��A�*#

Loss?��>

Adam/Loss/raw���=m�jL1       ����	�1�ˀ��A�*#

LossO�>

Adam/Loss/raw޸�=�j�1       ����	�o�ˀ��A�*#

Loss��>

Adam/Loss/raw� =J�1       ����	���ˀ��A�*#

Lossͺ�>

Adam/Loss/raw޴�@���1       ����	$��ˀ��A�*#

Loss�1G?

Adam/Loss/rawvF�=���1       ����	��ˀ��A�*#

Loss� 5?

Adam/Loss/raw�`�<.@=t1       ����	�:�ˀ��A�*#

Loss�#?

Adam/Loss/raw���<��Z1       ����	�i�ˀ��A�*#

Loss@�?

Adam/Loss/raw8�d=n��Q1       ����	�t�ˀ��A�*#

Lossx�?

Adam/Loss/raw�i>!���1       ����	YL�ˀ��A�*#

Loss��>

Adam/Loss/raw�w�=j��1       ����	��ˀ��A�*#

Loss��>

Adam/Loss/raw��=���C1       ����	7:�ˀ��A�*#

Loss���>

Adam/Loss/raw��G=��1       ����	HP̀��A�*#

Loss�+�>

Adam/Loss/raw۬�=�N��1       ����	A̀��A�*#

LossMů>

Adam/Loss/raw��@h��H1       ����	,)̀��A�*#

Loss�dB?

Adam/Loss/raw�9�=Ċ�1       ����	��̀��A�*#

Lossf+1?

Adam/Loss/raw�Q>�Q�1       ����	��̀��A�*#

Loss�$?

Adam/Loss/raw�-�<Y	�1       ����	�3̀��A�*#

Loss@ ?

Adam/Loss/raw�J	=#�51       ����	�;̀��A�*#

Loss~�?

Adam/Loss/raw���=����1       ����	�"̀��A�*#

Loss�0�>

Adam/Loss/raw��=B ��1       ����	�u$̀��A�*#

Lossr��>

Adam/Loss/raw�P>��;1       ����	��)̀��A�*#

LossX��>

Adam/Loss/rawۣ=�y0�1       ����	��.̀��A�*#

Loss|��>

Adam/Loss/raw&?�<i~��1       ����	<�3̀��A�*#

Loss>

Adam/Loss/raw�;@�g��1       ����	$�8̀��A�*#

Loss��?

Adam/Loss/raw4
�=D��1       ����	�A:̀��A�*#

Loss4! ?

Adam/Loss/raw�
5>�>��1       ����	�;̀��A�*#

Loss���>

Adam/Loss/raws��=�4VL1       ����	��@̀��A�*#

Loss��>

Adam/Loss/rawi�=  �1       ����	��È��A�*#

Loss��>

Adam/Loss/raw�^=�Ce�1       ����	{�J̀��A�*#

Loss�P�>

Adam/Loss/raw��(=��	41       ����	ՕÒ��A�*#

Loss���>

Adam/Loss/raw�_={q��1       ����	T�T̀��A�*#

Loss���>

Adam/Loss/rawl�l=~�61       ����	��Ỳ��A�*#

Loss���>

Adam/Loss/raw>�+�&1       ����	;�^̀��A�*#

Loss+׆>

Adam/Loss/raw�@'ЅD1       ����	��d̀��A�*#

Loss�5?

Adam/Loss/rawb�.=�}�1       ����	Kf̀��A�*#

Loss<u$?

Adam/Loss/raw��B=�o�n1       ����	ףg̀��A�*#

Loss�:?

Adam/Loss/raw�_>=NWF1       ����	!�l̀��A�*#

Loss#?

Adam/Loss/raw�.=IYd*1       ����	u�q̀��A�*#

Loss��>

Adam/Loss/raw���=@隹1       ����	�v̀��A�*#

Loss=I�>

Adam/Loss/raw%;=��TL1       ����	�{̀��A�*#

Lossڵ�>

Adam/Loss/rawT�=�ܨ1       ����	�Ԁ̀��A�*#

Loss&�>

Adam/Loss/raw<2�<��Y1       ����	�ʅ̀��A�*#

Loss`ߨ>

Adam/Loss/raw0�D=�y��1       ����	g��̀��A�*#

Loss�q�>

Adam/Loss/rawr�p@�'�1       ����	$��̀��A�*#

Loss��%?

Adam/Loss/raw9�=�1'�1       ����	`;�̀��A�*#

Lossy�?

Adam/Loss/raw�Η=wnFm1       ����	���̀��A�*#

LossO�	?

Adam/Loss/rawډ`<z�