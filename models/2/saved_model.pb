Ќд
нЅ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
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
$
DisableCopyOnRead
resourceИ
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Т
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
∞
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.15.02v2.15.0-0-g6887368d6d48†ґ
є
random_rotation_1/StateVarVarHandleOp*
_output_shapes
: *+

debug_namerandom_rotation_1/StateVar/*
dtype0	*
shape:*+
shared_namerandom_rotation_1/StateVar
Е
.random_rotation_1/StateVar/Read/ReadVariableOpReadVariableOprandom_rotation_1/StateVar*
_output_shapes
:*
dtype0	
≠
random_flip_1/StateVarVarHandleOp*
_output_shapes
: *'

debug_namerandom_flip_1/StateVar/*
dtype0	*
shape:*'
shared_namerandom_flip_1/StateVar
}
*random_flip_1/StateVar/Read/ReadVariableOpReadVariableOprandom_flip_1/StateVar*
_output_shapes
:*
dtype0	
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
І
Adam/v/dense_26/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_26/bias/*
dtype0*
shape:*%
shared_nameAdam/v/dense_26/bias
y
(Adam/v/dense_26/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_26/bias*
_output_shapes
:*
dtype0
І
Adam/m/dense_26/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_26/bias/*
dtype0*
shape:*%
shared_nameAdam/m/dense_26/bias
y
(Adam/m/dense_26/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_26/bias*
_output_shapes
:*
dtype0
±
Adam/v/dense_26/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_26/kernel/*
dtype0*
shape
:H*'
shared_nameAdam/v/dense_26/kernel
Б
*Adam/v/dense_26/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_26/kernel*
_output_shapes

:H*
dtype0
±
Adam/m/dense_26/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_26/kernel/*
dtype0*
shape
:H*'
shared_nameAdam/m/dense_26/kernel
Б
*Adam/m/dense_26/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_26/kernel*
_output_shapes

:H*
dtype0
І
Adam/v/dense_25/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_25/bias/*
dtype0*
shape:H*%
shared_nameAdam/v/dense_25/bias
y
(Adam/v/dense_25/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_25/bias*
_output_shapes
:H*
dtype0
І
Adam/m/dense_25/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_25/bias/*
dtype0*
shape:H*%
shared_nameAdam/m/dense_25/bias
y
(Adam/m/dense_25/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_25/bias*
_output_shapes
:H*
dtype0
±
Adam/v/dense_25/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_25/kernel/*
dtype0*
shape
:HH*'
shared_nameAdam/v/dense_25/kernel
Б
*Adam/v/dense_25/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_25/kernel*
_output_shapes

:HH*
dtype0
±
Adam/m/dense_25/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_25/kernel/*
dtype0*
shape
:HH*'
shared_nameAdam/m/dense_25/kernel
Б
*Adam/m/dense_25/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_25/kernel*
_output_shapes

:HH*
dtype0
І
Adam/v/dense_24/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_24/bias/*
dtype0*
shape:H*%
shared_nameAdam/v/dense_24/bias
y
(Adam/v/dense_24/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_24/bias*
_output_shapes
:H*
dtype0
І
Adam/m/dense_24/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_24/bias/*
dtype0*
shape:H*%
shared_nameAdam/m/dense_24/bias
y
(Adam/m/dense_24/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_24/bias*
_output_shapes
:H*
dtype0
≤
Adam/v/dense_24/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_24/kernel/*
dtype0*
shape:	†H*'
shared_nameAdam/v/dense_24/kernel
В
*Adam/v/dense_24/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_24/kernel*
_output_shapes
:	†H*
dtype0
≤
Adam/m/dense_24/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_24/kernel/*
dtype0*
shape:	†H*'
shared_nameAdam/m/dense_24/kernel
В
*Adam/m/dense_24/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_24/kernel*
_output_shapes
:	†H*
dtype0
™
Adam/v/conv2d_55/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv2d_55/bias/*
dtype0*
shape:H*&
shared_nameAdam/v/conv2d_55/bias
{
)Adam/v/conv2d_55/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_55/bias*
_output_shapes
:H*
dtype0
™
Adam/m/conv2d_55/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv2d_55/bias/*
dtype0*
shape:H*&
shared_nameAdam/m/conv2d_55/bias
{
)Adam/m/conv2d_55/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_55/bias*
_output_shapes
:H*
dtype0
Љ
Adam/v/conv2d_55/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv2d_55/kernel/*
dtype0*
shape:HH*(
shared_nameAdam/v/conv2d_55/kernel
Л
+Adam/v/conv2d_55/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_55/kernel*&
_output_shapes
:HH*
dtype0
Љ
Adam/m/conv2d_55/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv2d_55/kernel/*
dtype0*
shape:HH*(
shared_nameAdam/m/conv2d_55/kernel
Л
+Adam/m/conv2d_55/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_55/kernel*&
_output_shapes
:HH*
dtype0
™
Adam/v/conv2d_54/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv2d_54/bias/*
dtype0*
shape:H*&
shared_nameAdam/v/conv2d_54/bias
{
)Adam/v/conv2d_54/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_54/bias*
_output_shapes
:H*
dtype0
™
Adam/m/conv2d_54/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv2d_54/bias/*
dtype0*
shape:H*&
shared_nameAdam/m/conv2d_54/bias
{
)Adam/m/conv2d_54/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_54/bias*
_output_shapes
:H*
dtype0
Љ
Adam/v/conv2d_54/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv2d_54/kernel/*
dtype0*
shape:HH*(
shared_nameAdam/v/conv2d_54/kernel
Л
+Adam/v/conv2d_54/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_54/kernel*&
_output_shapes
:HH*
dtype0
Љ
Adam/m/conv2d_54/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv2d_54/kernel/*
dtype0*
shape:HH*(
shared_nameAdam/m/conv2d_54/kernel
Л
+Adam/m/conv2d_54/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_54/kernel*&
_output_shapes
:HH*
dtype0
™
Adam/v/conv2d_53/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv2d_53/bias/*
dtype0*
shape:H*&
shared_nameAdam/v/conv2d_53/bias
{
)Adam/v/conv2d_53/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_53/bias*
_output_shapes
:H*
dtype0
™
Adam/m/conv2d_53/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv2d_53/bias/*
dtype0*
shape:H*&
shared_nameAdam/m/conv2d_53/bias
{
)Adam/m/conv2d_53/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_53/bias*
_output_shapes
:H*
dtype0
Љ
Adam/v/conv2d_53/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv2d_53/kernel/*
dtype0*
shape:HH*(
shared_nameAdam/v/conv2d_53/kernel
Л
+Adam/v/conv2d_53/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_53/kernel*&
_output_shapes
:HH*
dtype0
Љ
Adam/m/conv2d_53/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv2d_53/kernel/*
dtype0*
shape:HH*(
shared_nameAdam/m/conv2d_53/kernel
Л
+Adam/m/conv2d_53/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_53/kernel*&
_output_shapes
:HH*
dtype0
™
Adam/v/conv2d_52/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv2d_52/bias/*
dtype0*
shape:H*&
shared_nameAdam/v/conv2d_52/bias
{
)Adam/v/conv2d_52/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_52/bias*
_output_shapes
:H*
dtype0
™
Adam/m/conv2d_52/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv2d_52/bias/*
dtype0*
shape:H*&
shared_nameAdam/m/conv2d_52/bias
{
)Adam/m/conv2d_52/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_52/bias*
_output_shapes
:H*
dtype0
Љ
Adam/v/conv2d_52/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv2d_52/kernel/*
dtype0*
shape:HH*(
shared_nameAdam/v/conv2d_52/kernel
Л
+Adam/v/conv2d_52/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_52/kernel*&
_output_shapes
:HH*
dtype0
Љ
Adam/m/conv2d_52/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv2d_52/kernel/*
dtype0*
shape:HH*(
shared_nameAdam/m/conv2d_52/kernel
Л
+Adam/m/conv2d_52/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_52/kernel*&
_output_shapes
:HH*
dtype0
™
Adam/v/conv2d_51/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv2d_51/bias/*
dtype0*
shape:H*&
shared_nameAdam/v/conv2d_51/bias
{
)Adam/v/conv2d_51/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_51/bias*
_output_shapes
:H*
dtype0
™
Adam/m/conv2d_51/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv2d_51/bias/*
dtype0*
shape:H*&
shared_nameAdam/m/conv2d_51/bias
{
)Adam/m/conv2d_51/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_51/bias*
_output_shapes
:H*
dtype0
Љ
Adam/v/conv2d_51/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv2d_51/kernel/*
dtype0*
shape: H*(
shared_nameAdam/v/conv2d_51/kernel
Л
+Adam/v/conv2d_51/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_51/kernel*&
_output_shapes
: H*
dtype0
Љ
Adam/m/conv2d_51/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv2d_51/kernel/*
dtype0*
shape: H*(
shared_nameAdam/m/conv2d_51/kernel
Л
+Adam/m/conv2d_51/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_51/kernel*&
_output_shapes
: H*
dtype0
™
Adam/v/conv2d_50/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv2d_50/bias/*
dtype0*
shape: *&
shared_nameAdam/v/conv2d_50/bias
{
)Adam/v/conv2d_50/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_50/bias*
_output_shapes
: *
dtype0
™
Adam/m/conv2d_50/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv2d_50/bias/*
dtype0*
shape: *&
shared_nameAdam/m/conv2d_50/bias
{
)Adam/m/conv2d_50/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_50/bias*
_output_shapes
: *
dtype0
Љ
Adam/v/conv2d_50/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv2d_50/kernel/*
dtype0*
shape: *(
shared_nameAdam/v/conv2d_50/kernel
Л
+Adam/v/conv2d_50/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_50/kernel*&
_output_shapes
: *
dtype0
Љ
Adam/m/conv2d_50/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv2d_50/kernel/*
dtype0*
shape: *(
shared_nameAdam/m/conv2d_50/kernel
Л
+Adam/m/conv2d_50/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_50/kernel*&
_output_shapes
: *
dtype0
О
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
В
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
Т
dense_26/biasVarHandleOp*
_output_shapes
: *

debug_namedense_26/bias/*
dtype0*
shape:*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:*
dtype0
Ь
dense_26/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_26/kernel/*
dtype0*
shape
:H* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:H*
dtype0
Т
dense_25/biasVarHandleOp*
_output_shapes
: *

debug_namedense_25/bias/*
dtype0*
shape:H*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:H*
dtype0
Ь
dense_25/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_25/kernel/*
dtype0*
shape
:HH* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:HH*
dtype0
Т
dense_24/biasVarHandleOp*
_output_shapes
: *

debug_namedense_24/bias/*
dtype0*
shape:H*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:H*
dtype0
Э
dense_24/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_24/kernel/*
dtype0*
shape:	†H* 
shared_namedense_24/kernel
t
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes
:	†H*
dtype0
Х
conv2d_55/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_55/bias/*
dtype0*
shape:H*
shared_nameconv2d_55/bias
m
"conv2d_55/bias/Read/ReadVariableOpReadVariableOpconv2d_55/bias*
_output_shapes
:H*
dtype0
І
conv2d_55/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_55/kernel/*
dtype0*
shape:HH*!
shared_nameconv2d_55/kernel
}
$conv2d_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_55/kernel*&
_output_shapes
:HH*
dtype0
Х
conv2d_54/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_54/bias/*
dtype0*
shape:H*
shared_nameconv2d_54/bias
m
"conv2d_54/bias/Read/ReadVariableOpReadVariableOpconv2d_54/bias*
_output_shapes
:H*
dtype0
І
conv2d_54/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_54/kernel/*
dtype0*
shape:HH*!
shared_nameconv2d_54/kernel
}
$conv2d_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_54/kernel*&
_output_shapes
:HH*
dtype0
Х
conv2d_53/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_53/bias/*
dtype0*
shape:H*
shared_nameconv2d_53/bias
m
"conv2d_53/bias/Read/ReadVariableOpReadVariableOpconv2d_53/bias*
_output_shapes
:H*
dtype0
І
conv2d_53/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_53/kernel/*
dtype0*
shape:HH*!
shared_nameconv2d_53/kernel
}
$conv2d_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_53/kernel*&
_output_shapes
:HH*
dtype0
Х
conv2d_52/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_52/bias/*
dtype0*
shape:H*
shared_nameconv2d_52/bias
m
"conv2d_52/bias/Read/ReadVariableOpReadVariableOpconv2d_52/bias*
_output_shapes
:H*
dtype0
І
conv2d_52/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_52/kernel/*
dtype0*
shape:HH*!
shared_nameconv2d_52/kernel
}
$conv2d_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_52/kernel*&
_output_shapes
:HH*
dtype0
Х
conv2d_51/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_51/bias/*
dtype0*
shape:H*
shared_nameconv2d_51/bias
m
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
_output_shapes
:H*
dtype0
І
conv2d_51/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_51/kernel/*
dtype0*
shape: H*!
shared_nameconv2d_51/kernel
}
$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*&
_output_shapes
: H*
dtype0
Х
conv2d_50/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_50/bias/*
dtype0*
shape: *
shared_nameconv2d_50/bias
m
"conv2d_50/bias/Read/ReadVariableOpReadVariableOpconv2d_50/bias*
_output_shapes
: *
dtype0
І
conv2d_50/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_50/kernel/*
dtype0*
shape: *!
shared_nameconv2d_50/kernel
}
$conv2d_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_50/kernel*&
_output_shapes
: *
dtype0
Щ
"serving_default_sequential_6_inputPlaceholder*1
_output_shapes
:€€€€€€€€€АА*
dtype0*&
shape:€€€€€€€€€АА
П
StatefulPartitionedCallStatefulPartitionedCall"serving_default_sequential_6_inputconv2d_50/kernelconv2d_50/biasconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasconv2d_54/kernelconv2d_54/biasconv2d_55/kernelconv2d_55/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_39524

NoOpNoOp
оƒ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*®ƒ
valueЭƒBЩƒ BСƒ
К
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer_with_weights-5
layer-18
layer-19
layer-20
layer-21
layer_with_weights-6
layer-22
layer_with_weights-7
layer-23
layer_with_weights-8
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures*
®
#layer-0
$layer-1
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses* 
™
+layer-0
,layer-1
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses*
О
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
»
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias
 A_jit_compiled_convolution_op*
О
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
О
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
»
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias
 V_jit_compiled_convolution_op*
О
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses* 
О
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
»
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias
 k_jit_compiled_convolution_op*
О
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
О
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses* 
…
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias
!А_jit_compiled_convolution_op*
Ф
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses* 
Ф
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses* 
—
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
Уkernel
	Фbias
!Х_jit_compiled_convolution_op*
Ф
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses* 
Ф
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
†__call__
+°&call_and_return_all_conditional_losses* 
—
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses
®kernel
	©bias
!™_jit_compiled_convolution_op*
Ф
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses* 
Ф
±	variables
≤trainable_variables
≥regularization_losses
і	keras_api
µ__call__
+ґ&call_and_return_all_conditional_losses* 
Ф
Ј	variables
Єtrainable_variables
єregularization_losses
Ї	keras_api
ї__call__
+Љ&call_and_return_all_conditional_losses* 
Ѓ
љ	variables
Њtrainable_variables
њregularization_losses
ј	keras_api
Ѕ__call__
+¬&call_and_return_all_conditional_losses
√kernel
	ƒbias*
Ѓ
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses
Ћkernel
	ћbias*
Ѓ
Ќ	variables
ќtrainable_variables
ѕregularization_losses
–	keras_api
—__call__
+“&call_and_return_all_conditional_losses
”kernel
	‘bias*
Ф
?0
@1
T2
U3
i4
j5
~6
7
У8
Ф9
®10
©11
√12
ƒ13
Ћ14
ћ15
”16
‘17*
Ф
?0
@1
T2
U3
i4
j5
~6
7
У8
Ф9
®10
©11
√12
ƒ13
Ћ14
ћ15
”16
‘17*
* 
µ
’non_trainable_variables
÷layers
„metrics
 Ўlayer_regularization_losses
ўlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Џtrace_0
џtrace_1* 

№trace_0
Ёtrace_1* 
* 
И
ё
_variables
я_iterations
а_learning_rate
б_index_dict
в
_momentums
г_velocities
д_update_step_xla*

еserving_default* 
Ф
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses* 
Ф
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
р__call__
+с&call_and_return_all_conditional_losses* 
* 
* 
* 
Ц
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

чtrace_0
шtrace_1* 

щtrace_0
ъtrace_1* 
Ѓ
ы	variables
ьtrainable_variables
эregularization_losses
ю	keras_api
€__call__
+А&call_and_return_all_conditional_losses
Б_random_generator*
Ѓ
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses
И_random_generator*
* 
* 
* 
Ш
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

Оtrace_0
Пtrace_1* 

Рtrace_0
Сtrace_1* 
* 
* 
* 
Ц
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

Чtrace_0* 

Шtrace_0* 

?0
@1*

?0
@1*
* 
Ш
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

Юtrace_0* 

Яtrace_0* 
`Z
VARIABLE_VALUEconv2d_50/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_50/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

•trace_0* 

¶trace_0* 
* 
* 
* 
Ц
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

ђtrace_0* 

≠trace_0* 

T0
U1*

T0
U1*
* 
Ш
Ѓnon_trainable_variables
ѓlayers
∞metrics
 ±layer_regularization_losses
≤layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

≥trace_0* 

іtrace_0* 
`Z
VARIABLE_VALUEconv2d_51/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_51/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
µnon_trainable_variables
ґlayers
Јmetrics
 Єlayer_regularization_losses
єlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 

Їtrace_0* 

їtrace_0* 
* 
* 
* 
Ц
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

Ѕtrace_0* 

¬trace_0* 

i0
j1*

i0
j1*
* 
Ш
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

»trace_0* 

…trace_0* 
`Z
VARIABLE_VALUEconv2d_52/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_52/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
 non_trainable_variables
Ћlayers
ћmetrics
 Ќlayer_regularization_losses
ќlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 

ѕtrace_0* 

–trace_0* 
* 
* 
* 
Ц
—non_trainable_variables
“layers
”metrics
 ‘layer_regularization_losses
’layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 

÷trace_0* 

„trace_0* 

~0
1*

~0
1*
* 
Ш
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

Ёtrace_0* 

ёtrace_0* 
`Z
VARIABLE_VALUEconv2d_53/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_53/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ь
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses* 

дtrace_0* 

еtrace_0* 
* 
* 
* 
Ь
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses* 

лtrace_0* 

мtrace_0* 

У0
Ф1*

У0
Ф1*
* 
Ю
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses*

тtrace_0* 

уtrace_0* 
`Z
VARIABLE_VALUEconv2d_54/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_54/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ь
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses* 

щtrace_0* 

ъtrace_0* 
* 
* 
* 
Ь
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
€layer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
†__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses* 

Аtrace_0* 

Бtrace_0* 

®0
©1*

®0
©1*
* 
Ю
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses*

Зtrace_0* 

Иtrace_0* 
`Z
VARIABLE_VALUEconv2d_55/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_55/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ь
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
Ђ	variables
ђtrainable_variables
≠regularization_losses
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses* 

Оtrace_0* 

Пtrace_0* 
* 
* 
* 
Ь
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
±	variables
≤trainable_variables
≥regularization_losses
µ__call__
+ґ&call_and_return_all_conditional_losses
'ґ"call_and_return_conditional_losses* 

Хtrace_0* 

Цtrace_0* 
* 
* 
* 
Ь
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
Ј	variables
Єtrainable_variables
єregularization_losses
ї__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses* 

Ьtrace_0* 

Эtrace_0* 

√0
ƒ1*

√0
ƒ1*
* 
Ю
Юnon_trainable_variables
Яlayers
†metrics
 °layer_regularization_losses
Ґlayer_metrics
љ	variables
Њtrainable_variables
њregularization_losses
Ѕ__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*

£trace_0* 

§trace_0* 
_Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_24/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ћ0
ћ1*

Ћ0
ћ1*
* 
Ю
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*

™trace_0* 

Ђtrace_0* 
_Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_25/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

”0
‘1*

”0
‘1*
* 
Ю
ђnon_trainable_variables
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
∞layer_metrics
Ќ	variables
ќtrainable_variables
ѕregularization_losses
—__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses*

±trace_0* 

≤trace_0* 
_Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_26/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
¬
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24*

≥0
і1*
* 
* 
* 
* 
* 
* 
«
я0
µ1
ґ2
Ј3
Є4
є5
Ї6
ї7
Љ8
љ9
Њ10
њ11
ј12
Ѕ13
¬14
√15
ƒ16
≈17
∆18
«19
»20
…21
 22
Ћ23
ћ24
Ќ25
ќ26
ѕ27
–28
—29
“30
”31
‘32
’33
÷34
„35
Ў36*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
Ь
µ0
Ј1
є2
ї3
љ4
њ5
Ѕ6
√7
≈8
«9
…10
Ћ11
Ќ12
ѕ13
—14
”15
’16
„17*
Ь
ґ0
Є1
Ї2
Љ3
Њ4
ј5
¬6
ƒ7
∆8
»9
 10
ћ11
ќ12
–13
“14
‘15
÷16
Ў17*
Ж
ўtrace_0
Џtrace_1
џtrace_2
№trace_3
Ёtrace_4
ёtrace_5
яtrace_6
аtrace_7
бtrace_8
вtrace_9
гtrace_10
дtrace_11
еtrace_12
жtrace_13
зtrace_14
иtrace_15
йtrace_16
кtrace_17* 
* 
* 
* 
* 
Ь
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses* 

рtrace_0* 

сtrace_0* 
* 
* 
* 
Ь
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
м	variables
нtrainable_variables
оregularization_losses
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses* 

чtrace_0* 

шtrace_0* 
* 

#0
$1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
ы	variables
ьtrainable_variables
эregularization_losses
€__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses* 

юtrace_0
€trace_1* 

Аtrace_0
Бtrace_1* 

В
_generator*
* 
* 
* 
Ь
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses* 

Иtrace_0
Йtrace_1* 

Кtrace_0
Лtrace_1* 

М
_generator*
* 

+0
,1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Н	variables
О	keras_api

Пtotal

Рcount*
M
С	variables
Т	keras_api

Уtotal

Фcount
Х
_fn_kwargs*
b\
VARIABLE_VALUEAdam/m/conv2d_50/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_50/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_50/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_50/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_51/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_51/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_51/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_51/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_52/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_52/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_52/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_52/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_53/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_53/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_53/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_53/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_54/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_54/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_54/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_54/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_55/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_55/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_55/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_55/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_24/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_24/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_24/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_24/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_25/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_25/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_25/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_25/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_26/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_26/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_26/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_26/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ц
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ч
_state_var*

П0
Р1*

Н	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

У0
Ф1*

С	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
В|
VARIABLE_VALUErandom_flip_1/StateVarRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUErandom_rotation_1/StateVarRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
…
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_50/kernelconv2d_50/biasconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasconv2d_54/kernelconv2d_54/biasconv2d_55/kernelconv2d_55/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/bias	iterationlearning_rateAdam/m/conv2d_50/kernelAdam/v/conv2d_50/kernelAdam/m/conv2d_50/biasAdam/v/conv2d_50/biasAdam/m/conv2d_51/kernelAdam/v/conv2d_51/kernelAdam/m/conv2d_51/biasAdam/v/conv2d_51/biasAdam/m/conv2d_52/kernelAdam/v/conv2d_52/kernelAdam/m/conv2d_52/biasAdam/v/conv2d_52/biasAdam/m/conv2d_53/kernelAdam/v/conv2d_53/kernelAdam/m/conv2d_53/biasAdam/v/conv2d_53/biasAdam/m/conv2d_54/kernelAdam/v/conv2d_54/kernelAdam/m/conv2d_54/biasAdam/v/conv2d_54/biasAdam/m/conv2d_55/kernelAdam/v/conv2d_55/kernelAdam/m/conv2d_55/biasAdam/v/conv2d_55/biasAdam/m/dense_24/kernelAdam/v/dense_24/kernelAdam/m/dense_24/biasAdam/v/dense_24/biasAdam/m/dense_25/kernelAdam/v/dense_25/kernelAdam/m/dense_25/biasAdam/v/dense_25/biasAdam/m/dense_26/kernelAdam/v/dense_26/kernelAdam/m/dense_26/biasAdam/v/dense_26/biastotal_1count_1totalcountrandom_flip_1/StateVarrandom_rotation_1/StateVarConst*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_40529
ƒ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_50/kernelconv2d_50/biasconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasconv2d_54/kernelconv2d_54/biasconv2d_55/kernelconv2d_55/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/bias	iterationlearning_rateAdam/m/conv2d_50/kernelAdam/v/conv2d_50/kernelAdam/m/conv2d_50/biasAdam/v/conv2d_50/biasAdam/m/conv2d_51/kernelAdam/v/conv2d_51/kernelAdam/m/conv2d_51/biasAdam/v/conv2d_51/biasAdam/m/conv2d_52/kernelAdam/v/conv2d_52/kernelAdam/m/conv2d_52/biasAdam/v/conv2d_52/biasAdam/m/conv2d_53/kernelAdam/v/conv2d_53/kernelAdam/m/conv2d_53/biasAdam/v/conv2d_53/biasAdam/m/conv2d_54/kernelAdam/v/conv2d_54/kernelAdam/m/conv2d_54/biasAdam/v/conv2d_54/biasAdam/m/conv2d_55/kernelAdam/v/conv2d_55/kernelAdam/m/conv2d_55/biasAdam/v/conv2d_55/biasAdam/m/dense_24/kernelAdam/v/dense_24/kernelAdam/m/dense_24/biasAdam/v/dense_24/biasAdam/m/dense_25/kernelAdam/v/dense_25/kernelAdam/m/dense_25/biasAdam/v/dense_25/biasAdam/m/dense_26/kernelAdam/v/dense_26/kernelAdam/m/dense_26/biasAdam/v/dense_26/biastotal_1count_1totalcountrandom_flip_1/StateVarrandom_rotation_1/StateVar*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_40724Ѓ√
з
_
C__inference_re_lu_53_layer_call_and_return_conditional_losses_39110

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
Ы
Ю
)__inference_conv2d_51_layer_call_fn_39583

inputs!
unknown: H
	unknown_0:H
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_39056w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€H<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ББ : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ББ 
 
_user_specified_nameinputs:%!

_user_specified_name39577:%!

_user_specified_name39579
ќ

х
C__inference_dense_24_layer_call_and_return_conditional_losses_39800

inputs1
matmul_readvariableop_resource:	†H-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	†H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€HP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ha
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
≤
E
)__inference_flatten_4_layer_call_fn_39774

inputs
identity≥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_39162a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€†"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
Ў

э
D__inference_conv2d_55_layer_call_and_return_conditional_losses_39749

inputs8
conv2d_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ц
d
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38898

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ў

э
D__inference_conv2d_53_layer_call_and_return_conditional_losses_39100

inputs8
conv2d_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ч
Ю
)__inference_conv2d_55_layer_call_fn_39739

inputs!
unknown:HH
	unknown_0:H
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_55_layer_call_and_return_conditional_losses_39144w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€H<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€H: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:%!

_user_specified_name39733:%!

_user_specified_name39735
У
g
K__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_39000

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_39691

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Уf
Ь	
H__inference_sequential_12_layer_call_and_return_conditional_losses_39213
sequential_6_input 
sequential_7_39019:	 
sequential_7_39021:	)
conv2d_50_39035: 
conv2d_50_39037: )
conv2d_51_39057: H
conv2d_51_39059:H)
conv2d_52_39079:HH
conv2d_52_39081:H)
conv2d_53_39101:HH
conv2d_53_39103:H)
conv2d_54_39123:HH
conv2d_54_39125:H)
conv2d_55_39145:HH
conv2d_55_39147:H!
dense_24_39175:	†H
dense_24_39177:H 
dense_25_39191:HH
dense_25_39193:H 
dense_26_39207:H
dense_26_39209:
identityИҐ!conv2d_50/StatefulPartitionedCallҐ!conv2d_51/StatefulPartitionedCallҐ!conv2d_52/StatefulPartitionedCallҐ!conv2d_53/StatefulPartitionedCallҐ!conv2d_54/StatefulPartitionedCallҐ!conv2d_55/StatefulPartitionedCallҐ dense_24/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallҐ$sequential_7/StatefulPartitionedCallЎ
sequential_6/PartitionedCallPartitionedCallsequential_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_38627•
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall%sequential_6/PartitionedCall:output:0sequential_7_39019sequential_7_39021*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_38892ы
 zero_padding2d_7/PartitionedCallPartitionedCall-sequential_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ДД* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_38950°
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_7/PartitionedCall:output:0conv2d_50_39035conv2d_50_39037*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ВВ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_39034и
re_lu_50/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ВВ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_50_layer_call_and_return_conditional_losses_39044п
 max_pooling2d_50/PartitionedCallPartitionedCall!re_lu_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ББ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_38960Я
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0conv2d_51_39057conv2d_51_39059*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_39056ж
re_lu_51/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_51_layer_call_and_return_conditional_losses_39066н
 max_pooling2d_51/PartitionedCallPartitionedCall!re_lu_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€??H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_38970Я
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0conv2d_52_39079conv2d_52_39081*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€==H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_52_layer_call_and_return_conditional_losses_39078ж
re_lu_52/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€==H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_52_layer_call_and_return_conditional_losses_39088н
 max_pooling2d_52/PartitionedCallPartitionedCall!re_lu_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_38980Я
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0conv2d_53_39101conv2d_53_39103*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_39100ж
re_lu_53/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_53_layer_call_and_return_conditional_losses_39110н
 max_pooling2d_53/PartitionedCallPartitionedCall!re_lu_53/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_38990Я
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_53/PartitionedCall:output:0conv2d_54_39123conv2d_54_39125*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_54_layer_call_and_return_conditional_losses_39122ж
re_lu_54/PartitionedCallPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_54_layer_call_and_return_conditional_losses_39132н
 max_pooling2d_54/PartitionedCallPartitionedCall!re_lu_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_39000Я
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_54/PartitionedCall:output:0conv2d_55_39145conv2d_55_39147*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_55_layer_call_and_return_conditional_losses_39144ж
re_lu_55/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_55_layer_call_and_return_conditional_losses_39154н
 max_pooling2d_55/PartitionedCallPartitionedCall!re_lu_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_39010а
flatten_4/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_39162М
 dense_24/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_24_39175dense_24_39177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_39174У
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_39191dense_25_39193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_39190У
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_39207dense_26_39209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_39206x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€К
NoOpNoOp"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : 2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_6_input:%!

_user_specified_name39019:%!

_user_specified_name39021:%!

_user_specified_name39035:%!

_user_specified_name39037:%!

_user_specified_name39057:%!

_user_specified_name39059:%!

_user_specified_name39079:%!

_user_specified_name39081:%	!

_user_specified_name39101:%
!

_user_specified_name39103:%!

_user_specified_name39123:%!

_user_specified_name39125:%!

_user_specified_name39145:%!

_user_specified_name39147:%!

_user_specified_name39175:%!

_user_specified_name39177:%!

_user_specified_name39191:%!

_user_specified_name39193:%!

_user_specified_name39207:%!

_user_specified_name39209
Њ
D
(__inference_re_lu_54_layer_call_fn_39715

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_54_layer_call_and_return_conditional_losses_39132h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
ї
L
0__inference_max_pooling2d_55_layer_call_fn_39764

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_39010Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_37059
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
У
g
K__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_39010

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
з
_
C__inference_re_lu_55_layer_call_and_return_conditional_losses_39154

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
Я
Ю
)__inference_conv2d_50_layer_call_fn_39544

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ВВ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_39034y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ВВ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ДД: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ДД
 
_user_specified_nameinputs:%!

_user_specified_name39538:%!

_user_specified_name39540
й
g
K__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_38950

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ш
b
F__inference_rescaling_1_layer_call_and_return_conditional_losses_39864

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_37019
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:H: *
	_noinline(:D @

_output_shapes
:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ

ф
C__inference_dense_26_layer_call_and_return_conditional_losses_39206

inputs0
matmul_readvariableop_resource:H-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
з
_
C__inference_re_lu_54_layer_call_and_return_conditional_losses_39132

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
 
F
*__inference_resizing_1_layer_call_fn_39845

inputs
identityљ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_resizing_1_layer_call_and_return_conditional_losses_38615j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_36999
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:H: *
	_noinline(:D @

_output_shapes
:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
«А
≈
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40121

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityИҐstateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†њY
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ґ
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ґ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Б
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Л
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: Ч
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€А
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€Z
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @П
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:€€€€€€€€€А
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€g
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
::нѕm
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ў
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :£
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€S
transform/ShapeShapeinputs*
T0*
_output_shapes
::нѕg
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    °
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARТ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААD
NoOpNoOp ^stateful_uniform/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource
∆
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_39780

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€†Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€†"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
•Ч
А
H__inference_random_flip_1_layer_call_and_return_conditional_losses_39987

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityИҐ(stateful_uniform_full_int/RngReadAndSkipҐ*stateful_uniform_full_int_1/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ы
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Б
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Џ
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskП
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:У
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :С
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ч
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskЮ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:€€€€€€€€€АА†
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::нѕ~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЃ
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:В
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    В
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?∞
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::Ш
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :ю
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€х
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Т
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€ы
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ќ
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:е
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Ь
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:л
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА»
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААk
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?¬
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€—
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААњ
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€ААk
!stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:k
!stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: °
 stateful_uniform_full_int_1/ProdProd*stateful_uniform_full_int_1/shape:output:0*stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: d
"stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Е
"stateful_uniform_full_int_1/Cast_1Cast)stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Л
*stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource+stateful_uniform_full_int_1/Cast/x:output:0&stateful_uniform_full_int_1/Cast_1:y:0)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:y
/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
)stateful_uniform_full_int_1/strided_sliceStridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:08stateful_uniform_full_int_1/strided_slice/stack:output:0:stateful_uniform_full_int_1/strided_slice/stack_1:output:0:stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskУ
#stateful_uniform_full_int_1/BitcastBitcast2stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0{
1stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
+stateful_uniform_full_int_1/strided_slice_1StridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:0:stateful_uniform_full_int_1/strided_slice_1/stack:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ч
%stateful_uniform_full_int_1/Bitcast_1Bitcast4stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0a
stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :Ы
stateful_uniform_full_int_1StatelessRandomUniformFullIntV2*stateful_uniform_full_int_1/shape:output:0.stateful_uniform_full_int_1/Bitcast_1:output:0,stateful_uniform_full_int_1/Bitcast:output:0(stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	V
zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R ~
stack_1Pack$stateful_uniform_full_int_1:output:0zeros_like_1:output:0*
N*
T0	*
_output_shapes

:f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Б
strided_slice_1StridedSlicestack_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskџ
0stateless_random_flip_up_down/control_dependencyIdentity(stateless_random_flip_left_right/add:z:0*
T0*7
_class-
+)loc:@stateless_random_flip_left_right/add*1
_output_shapes
:€€€€€€€€€ААЪ
#stateless_random_flip_up_down/ShapeShape9stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
::нѕ{
1stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+stateless_random_flip_up_down/strided_sliceStridedSlice,stateless_random_flip_up_down/Shape:output:0:stateless_random_flip_up_down/strided_slice/stack:output:0<stateless_random_flip_up_down/strided_slice/stack_1:output:0<stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
<stateless_random_flip_up_down/stateless_random_uniform/shapePack4stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:
:stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
:stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ѓ
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice_1:output:0* 
_output_shapes
::Х
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :п
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Estateless_random_flip_up_down/stateless_random_uniform/shape:output:0Ystateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0]stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0\stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€м
:stateless_random_flip_up_down/stateless_random_uniform/subSubCstateless_random_flip_up_down/stateless_random_uniform/max:output:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Й
:stateless_random_flip_up_down/stateless_random_uniform/mulMulXstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0>stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€т
6stateless_random_flip_up_down/stateless_random_uniformAddV2>stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€o
-stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :њ
+stateless_random_flip_up_down/Reshape/shapePack4stateless_random_flip_up_down/strided_slice:output:06stateless_random_flip_up_down/Reshape/shape/1:output:06stateless_random_flip_up_down/Reshape/shape/2:output:06stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:№
%stateless_random_flip_up_down/ReshapeReshape:stateless_random_flip_up_down/stateless_random_uniform:z:04stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Ц
#stateless_random_flip_up_down/RoundRound.stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€v
,stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:в
'stateless_random_flip_up_down/ReverseV2	ReverseV29stateless_random_flip_up_down/control_dependency:output:05stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААњ
!stateless_random_flip_up_down/mulMul'stateless_random_flip_up_down/Round:y:00stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААh
#stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?є
!stateless_random_flip_up_down/subSub,stateless_random_flip_up_down/sub/x:output:0'stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€»
#stateless_random_flip_up_down/mul_1Mul%stateless_random_flip_up_down/sub:z:09stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААґ
!stateless_random_flip_up_down/addAddV2%stateless_random_flip_up_down/mul:z:0'stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА~
IdentityIdentity%stateless_random_flip_up_down/add:z:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААz
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip+^stateful_uniform_full_int_1/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2X
*stateful_uniform_full_int_1/RngReadAndSkip*stateful_uniform_full_int_1/RngReadAndSkip:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource
У
g
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_38970

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_36989
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:H: *
	_noinline(:D @

_output_shapes
:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
з
_
C__inference_re_lu_53_layer_call_and_return_conditional_losses_39681

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
Љ
Ґ
,__inference_sequential_7_layer_call_fn_38915
random_flip_1_input
unknown:	
	unknown_0:	
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_38892y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:€€€€€€€€€АА
-
_user_specified_namerandom_flip_1_input:%!

_user_specified_name38909:%!

_user_specified_name38911
–
I
-__inference_random_flip_1_layer_call_fn_39876

inputs
identityј
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38898j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ў

э
D__inference_conv2d_52_layer_call_and_return_conditional_losses_39632

inputs8
conv2d_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€==H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€==Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€==HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€??H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€??H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
«А
≈
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38887

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityИҐstateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†њY
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ґ
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ґ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Б
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Л
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: Ч
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€А
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€Z
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @П
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:€€€€€€€€€А
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€g
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
::нѕm
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€v
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ў
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :£
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€S
transform/ShapeShapeinputs*
T0*
_output_shapes
::нѕg
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    °
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARТ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААD
NoOpNoOp ^stateful_uniform/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource
Њ
D
(__inference_re_lu_55_layer_call_fn_39754

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_55_layer_call_and_return_conditional_losses_39154h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
ћ
G
+__inference_rescaling_1_layer_call_fn_39856

inputs
identityЊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_rescaling_1_layer_call_and_return_conditional_losses_38624j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
з
_
C__inference_re_lu_52_layer_call_and_return_conditional_losses_39088

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€==Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€==H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€==H:W S
/
_output_shapes
:€€€€€€€€€==H
 
_user_specified_nameinputs
ѕ
V
"__inference__update_step_xla_36984
gradient"
variable: H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: H: *
	_noinline(:P L
&
_output_shapes
: H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
У
g
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_38980

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
з
_
C__inference_re_lu_52_layer_call_and_return_conditional_losses_39642

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€==Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€==H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€==H:W S
/
_output_shapes
:€€€€€€€€€==H
 
_user_specified_nameinputs
ї
L
0__inference_zero_padding2d_7_layer_call_fn_39529

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_38950Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
•Ч
А
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38766

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityИҐ(stateful_uniform_full_int/RngReadAndSkipҐ*stateful_uniform_full_int_1/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ы
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Б
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Џ
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskП
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:У
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :С
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ч
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskЮ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:€€€€€€€€€АА†
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::нѕ~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЃ
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:В
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    В
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?∞
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::Ш
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :ю
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€х
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Т
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€ы
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ќ
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:е
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Ь
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:л
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА»
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААk
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?¬
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€—
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААњ
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€ААk
!stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:k
!stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: °
 stateful_uniform_full_int_1/ProdProd*stateful_uniform_full_int_1/shape:output:0*stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: d
"stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Е
"stateful_uniform_full_int_1/Cast_1Cast)stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Л
*stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource+stateful_uniform_full_int_1/Cast/x:output:0&stateful_uniform_full_int_1/Cast_1:y:0)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:y
/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
)stateful_uniform_full_int_1/strided_sliceStridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:08stateful_uniform_full_int_1/strided_slice/stack:output:0:stateful_uniform_full_int_1/strided_slice/stack_1:output:0:stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskУ
#stateful_uniform_full_int_1/BitcastBitcast2stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0{
1stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
+stateful_uniform_full_int_1/strided_slice_1StridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:0:stateful_uniform_full_int_1/strided_slice_1/stack:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ч
%stateful_uniform_full_int_1/Bitcast_1Bitcast4stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0a
stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :Ы
stateful_uniform_full_int_1StatelessRandomUniformFullIntV2*stateful_uniform_full_int_1/shape:output:0.stateful_uniform_full_int_1/Bitcast_1:output:0,stateful_uniform_full_int_1/Bitcast:output:0(stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	V
zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R ~
stack_1Pack$stateful_uniform_full_int_1:output:0zeros_like_1:output:0*
N*
T0	*
_output_shapes

:f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Б
strided_slice_1StridedSlicestack_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskџ
0stateless_random_flip_up_down/control_dependencyIdentity(stateless_random_flip_left_right/add:z:0*
T0*7
_class-
+)loc:@stateless_random_flip_left_right/add*1
_output_shapes
:€€€€€€€€€ААЪ
#stateless_random_flip_up_down/ShapeShape9stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
::нѕ{
1stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+stateless_random_flip_up_down/strided_sliceStridedSlice,stateless_random_flip_up_down/Shape:output:0:stateless_random_flip_up_down/strided_slice/stack:output:0<stateless_random_flip_up_down/strided_slice/stack_1:output:0<stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
<stateless_random_flip_up_down/stateless_random_uniform/shapePack4stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:
:stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
:stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ѓ
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice_1:output:0* 
_output_shapes
::Х
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :п
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Estateless_random_flip_up_down/stateless_random_uniform/shape:output:0Ystateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0]stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0\stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€м
:stateless_random_flip_up_down/stateless_random_uniform/subSubCstateless_random_flip_up_down/stateless_random_uniform/max:output:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Й
:stateless_random_flip_up_down/stateless_random_uniform/mulMulXstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0>stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€т
6stateless_random_flip_up_down/stateless_random_uniformAddV2>stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€o
-stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :њ
+stateless_random_flip_up_down/Reshape/shapePack4stateless_random_flip_up_down/strided_slice:output:06stateless_random_flip_up_down/Reshape/shape/1:output:06stateless_random_flip_up_down/Reshape/shape/2:output:06stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:№
%stateless_random_flip_up_down/ReshapeReshape:stateless_random_flip_up_down/stateless_random_uniform:z:04stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Ц
#stateless_random_flip_up_down/RoundRound.stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€v
,stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:в
'stateless_random_flip_up_down/ReverseV2	ReverseV29stateless_random_flip_up_down/control_dependency:output:05stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААњ
!stateless_random_flip_up_down/mulMul'stateless_random_flip_up_down/Round:y:00stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААh
#stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?є
!stateless_random_flip_up_down/subSub,stateless_random_flip_up_down/sub/x:output:0'stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€»
#stateless_random_flip_up_down/mul_1Mul%stateless_random_flip_up_down/sub:z:09stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААґ
!stateless_random_flip_up_down/addAddV2%stateless_random_flip_up_down/mul:z:0'stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА~
IdentityIdentity%stateless_random_flip_up_down/add:z:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААz
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip+^stateful_uniform_full_int_1/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2X
*stateful_uniform_full_int_1/RngReadAndSkip*stateful_uniform_full_int_1/RngReadAndSkip:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource
Ў

э
D__inference_conv2d_55_layer_call_and_return_conditional_losses_39144

inputs8
conv2d_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
У
g
K__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_39769

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м
R
,__inference_sequential_6_layer_call_fn_38638
resizing_1_input
identity…
PartitionedCallPartitionedCallresizing_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_38627j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:c _
1
_output_shapes
:€€€€€€€€€АА
*
_user_specified_nameresizing_1_input
Ў
M
1__inference_random_rotation_1_layer_call_fn_40003

inputs
identityƒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38903j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_39574

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ц
d
H__inference_random_flip_1_layer_call_and_return_conditional_losses_39991

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
ѕ
V
"__inference__update_step_xla_36994
gradient"
variable:HH*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:HH: *
	_noinline(:P L
&
_output_shapes
:HH
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ў

э
D__inference_conv2d_53_layer_call_and_return_conditional_losses_39671

inputs8
conv2d_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
з
_
C__inference_re_lu_55_layer_call_and_return_conditional_losses_39759

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
з
_
C__inference_re_lu_51_layer_call_and_return_conditional_losses_39603

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
™	
p
G__inference_sequential_7_layer_call_and_return_conditional_losses_38906
random_flip_1_input
identityџ
random_flip_1/PartitionedCallPartitionedCallrandom_flip_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38898ц
!random_rotation_1/PartitionedCallPartitionedCall&random_flip_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38903|
IdentityIdentity*random_rotation_1/PartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:f b
1
_output_shapes
:€€€€€€€€€АА
-
_user_specified_namerandom_flip_1_input
ї
L
0__inference_max_pooling2d_51_layer_call_fn_39608

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_38970Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѕ
V
"__inference__update_step_xla_36974
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
»
}
-__inference_random_flip_1_layer_call_fn_39871

inputs
unknown:	
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38766y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs:%!

_user_specified_name39867
й
g
K__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_39535

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
з
_
C__inference_re_lu_51_layer_call_and_return_conditional_losses_39066

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_37029
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:H: *
	_noinline(:D @

_output_shapes
:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
a
E__inference_resizing_1_layer_call_and_return_conditional_losses_38615

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      Ы
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
н
Х
(__inference_dense_26_layer_call_fn_39829

inputs
unknown:H
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_39206o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€H: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:%!

_user_specified_name39823:%!

_user_specified_name39825
Ј
N
"__inference__update_step_xla_37054
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:H: *
	_noinline(:H D

_output_shapes

:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
У
g
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_39652

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ч
Ю
)__inference_conv2d_54_layer_call_fn_39700

inputs!
unknown:HH
	unknown_0:H
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_54_layer_call_and_return_conditional_losses_39122w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€H<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€H: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:%!

_user_specified_name39694:%!

_user_specified_name39696
¶
С
G__inference_sequential_7_layer_call_and_return_conditional_losses_38892
random_flip_1_input!
random_flip_1_38767:	%
random_rotation_1_38888:	
identityИҐ%random_flip_1/StatefulPartitionedCallҐ)random_rotation_1/StatefulPartitionedCallБ
%random_flip_1/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_1_inputrandom_flip_1_38767*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38766®
)random_rotation_1/StatefulPartitionedCallStatefulPartitionedCall.random_flip_1/StatefulPartitionedCall:output:0random_rotation_1_38888*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38887Л
IdentityIdentity2random_rotation_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААv
NoOpNoOp&^random_flip_1/StatefulPartitionedCall*^random_rotation_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА: : 2N
%random_flip_1/StatefulPartitionedCall%random_flip_1/StatefulPartitionedCall2V
)random_rotation_1/StatefulPartitionedCall)random_rotation_1/StatefulPartitionedCall:f b
1
_output_shapes
:€€€€€€€€€АА
-
_user_specified_namerandom_flip_1_input:%!

_user_specified_name38767:%!

_user_specified_name38888
№

э
D__inference_conv2d_51_layer_call_and_return_conditional_losses_39056

inputs8
conv2d_readvariableop_resource: H-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: H*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ББ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€ББ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_37049
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:H: *
	_noinline(:D @

_output_shapes
:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ј
N
"__inference__update_step_xla_37044
gradient
variable:HH*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:HH: *
	_noinline(:H D

_output_shapes

:HH
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
з
_
C__inference_re_lu_54_layer_call_and_return_conditional_losses_39720

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€Hb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
ї
L
0__inference_max_pooling2d_53_layer_call_fn_39686

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_38990Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ў

э
D__inference_conv2d_54_layer_call_and_return_conditional_losses_39710

inputs8
conv2d_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
«Е
т
 __inference__wrapped_model_38607
sequential_6_inputP
6sequential_12_conv2d_50_conv2d_readvariableop_resource: E
7sequential_12_conv2d_50_biasadd_readvariableop_resource: P
6sequential_12_conv2d_51_conv2d_readvariableop_resource: HE
7sequential_12_conv2d_51_biasadd_readvariableop_resource:HP
6sequential_12_conv2d_52_conv2d_readvariableop_resource:HHE
7sequential_12_conv2d_52_biasadd_readvariableop_resource:HP
6sequential_12_conv2d_53_conv2d_readvariableop_resource:HHE
7sequential_12_conv2d_53_biasadd_readvariableop_resource:HP
6sequential_12_conv2d_54_conv2d_readvariableop_resource:HHE
7sequential_12_conv2d_54_biasadd_readvariableop_resource:HP
6sequential_12_conv2d_55_conv2d_readvariableop_resource:HHE
7sequential_12_conv2d_55_biasadd_readvariableop_resource:HH
5sequential_12_dense_24_matmul_readvariableop_resource:	†HD
6sequential_12_dense_24_biasadd_readvariableop_resource:HG
5sequential_12_dense_25_matmul_readvariableop_resource:HHD
6sequential_12_dense_25_biasadd_readvariableop_resource:HG
5sequential_12_dense_26_matmul_readvariableop_resource:HD
6sequential_12_dense_26_biasadd_readvariableop_resource:
identityИҐ.sequential_12/conv2d_50/BiasAdd/ReadVariableOpҐ-sequential_12/conv2d_50/Conv2D/ReadVariableOpҐ.sequential_12/conv2d_51/BiasAdd/ReadVariableOpҐ-sequential_12/conv2d_51/Conv2D/ReadVariableOpҐ.sequential_12/conv2d_52/BiasAdd/ReadVariableOpҐ-sequential_12/conv2d_52/Conv2D/ReadVariableOpҐ.sequential_12/conv2d_53/BiasAdd/ReadVariableOpҐ-sequential_12/conv2d_53/Conv2D/ReadVariableOpҐ.sequential_12/conv2d_54/BiasAdd/ReadVariableOpҐ-sequential_12/conv2d_54/Conv2D/ReadVariableOpҐ.sequential_12/conv2d_55/BiasAdd/ReadVariableOpҐ-sequential_12/conv2d_55/Conv2D/ReadVariableOpҐ-sequential_12/dense_24/BiasAdd/ReadVariableOpҐ,sequential_12/dense_24/MatMul/ReadVariableOpҐ-sequential_12/dense_25/BiasAdd/ReadVariableOpҐ,sequential_12/dense_25/MatMul/ReadVariableOpҐ-sequential_12/dense_26/BiasAdd/ReadVariableOpҐ,sequential_12/dense_26/MatMul/ReadVariableOpВ
1sequential_12/sequential_6/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      у
;sequential_12/sequential_6/resizing_1/resize/ResizeBilinearResizeBilinearsequential_6_input:sequential_12/sequential_6/resizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(r
-sequential_12/sequential_6/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;t
/sequential_12/sequential_6/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    у
*sequential_12/sequential_6/rescaling_1/mulMulLsequential_12/sequential_6/resizing_1/resize/ResizeBilinear:resized_images:06sequential_12/sequential_6/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААў
*sequential_12/sequential_6/rescaling_1/addAddV2.sequential_12/sequential_6/rescaling_1/mul:z:08sequential_12/sequential_6/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААЬ
+sequential_12/zero_padding2d_7/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             Ћ
"sequential_12/zero_padding2d_7/PadPad.sequential_12/sequential_6/rescaling_1/add:z:04sequential_12/zero_padding2d_7/Pad/paddings:output:0*
T0*1
_output_shapes
:€€€€€€€€€ДДђ
-sequential_12/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0с
sequential_12/conv2d_50/Conv2DConv2D+sequential_12/zero_padding2d_7/Pad:output:05sequential_12/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ *
paddingVALID*
strides
Ґ
.sequential_12/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0«
sequential_12/conv2d_50/BiasAddBiasAdd'sequential_12/conv2d_50/Conv2D:output:06sequential_12/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ Й
sequential_12/re_lu_50/ReluRelu(sequential_12/conv2d_50/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ Ћ
&sequential_12/max_pooling2d_50/MaxPoolMaxPool)sequential_12/re_lu_50/Relu:activations:0*1
_output_shapes
:€€€€€€€€€ББ *
ksize
*
paddingVALID*
strides
ђ
-sequential_12/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: H*
dtype0у
sequential_12/conv2d_51/Conv2DConv2D/sequential_12/max_pooling2d_50/MaxPool:output:05sequential_12/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
Ґ
.sequential_12/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0≈
sequential_12/conv2d_51/BiasAddBiasAdd'sequential_12/conv2d_51/Conv2D:output:06sequential_12/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€HЗ
sequential_12/re_lu_51/ReluRelu(sequential_12/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€H…
&sequential_12/max_pooling2d_51/MaxPoolMaxPool)sequential_12/re_lu_51/Relu:activations:0*/
_output_shapes
:€€€€€€€€€??H*
ksize
*
paddingVALID*
strides
ђ
-sequential_12/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0у
sequential_12/conv2d_52/Conv2DConv2D/sequential_12/max_pooling2d_51/MaxPool:output:05sequential_12/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€==H*
paddingVALID*
strides
Ґ
.sequential_12/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_52_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0≈
sequential_12/conv2d_52/BiasAddBiasAdd'sequential_12/conv2d_52/Conv2D:output:06sequential_12/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€==HЗ
sequential_12/re_lu_52/ReluRelu(sequential_12/conv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€==H…
&sequential_12/max_pooling2d_52/MaxPoolMaxPool)sequential_12/re_lu_52/Relu:activations:0*/
_output_shapes
:€€€€€€€€€H*
ksize
*
paddingVALID*
strides
ђ
-sequential_12/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0у
sequential_12/conv2d_53/Conv2DConv2D/sequential_12/max_pooling2d_52/MaxPool:output:05sequential_12/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
Ґ
.sequential_12/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_53_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0≈
sequential_12/conv2d_53/BiasAddBiasAdd'sequential_12/conv2d_53/Conv2D:output:06sequential_12/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€HЗ
sequential_12/re_lu_53/ReluRelu(sequential_12/conv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€H…
&sequential_12/max_pooling2d_53/MaxPoolMaxPool)sequential_12/re_lu_53/Relu:activations:0*/
_output_shapes
:€€€€€€€€€H*
ksize
*
paddingVALID*
strides
ђ
-sequential_12/conv2d_54/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0у
sequential_12/conv2d_54/Conv2DConv2D/sequential_12/max_pooling2d_53/MaxPool:output:05sequential_12/conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
Ґ
.sequential_12/conv2d_54/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0≈
sequential_12/conv2d_54/BiasAddBiasAdd'sequential_12/conv2d_54/Conv2D:output:06sequential_12/conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€HЗ
sequential_12/re_lu_54/ReluRelu(sequential_12/conv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€H…
&sequential_12/max_pooling2d_54/MaxPoolMaxPool)sequential_12/re_lu_54/Relu:activations:0*/
_output_shapes
:€€€€€€€€€H*
ksize
*
paddingVALID*
strides
ђ
-sequential_12/conv2d_55/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0у
sequential_12/conv2d_55/Conv2DConv2D/sequential_12/max_pooling2d_54/MaxPool:output:05sequential_12/conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
Ґ
.sequential_12/conv2d_55/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0≈
sequential_12/conv2d_55/BiasAddBiasAdd'sequential_12/conv2d_55/Conv2D:output:06sequential_12/conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€HЗ
sequential_12/re_lu_55/ReluRelu(sequential_12/conv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€H…
&sequential_12/max_pooling2d_55/MaxPoolMaxPool)sequential_12/re_lu_55/Relu:activations:0*/
_output_shapes
:€€€€€€€€€H*
ksize
*
paddingVALID*
strides
n
sequential_12/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
sequential_12/flatten_4/ReshapeReshape/sequential_12/max_pooling2d_55/MaxPool:output:0&sequential_12/flatten_4/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€†£
,sequential_12/dense_24/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_24_matmul_readvariableop_resource*
_output_shapes
:	†H*
dtype0є
sequential_12/dense_24/MatMulMatMul(sequential_12/flatten_4/Reshape:output:04sequential_12/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€H†
-sequential_12/dense_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_24_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0ї
sequential_12/dense_24/BiasAddBiasAdd'sequential_12/dense_24/MatMul:product:05sequential_12/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€H~
sequential_12/dense_24/ReluRelu'sequential_12/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€HҐ
,sequential_12/dense_25/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_25_matmul_readvariableop_resource*
_output_shapes

:HH*
dtype0Ї
sequential_12/dense_25/MatMulMatMul)sequential_12/dense_24/Relu:activations:04sequential_12/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€H†
-sequential_12/dense_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_25_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0ї
sequential_12/dense_25/BiasAddBiasAdd'sequential_12/dense_25/MatMul:product:05sequential_12/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€H~
sequential_12/dense_25/ReluRelu'sequential_12/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€HҐ
,sequential_12/dense_26/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_26_matmul_readvariableop_resource*
_output_shapes

:H*
dtype0Ї
sequential_12/dense_26/MatMulMatMul)sequential_12/dense_25/Relu:activations:04sequential_12/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
-sequential_12/dense_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ї
sequential_12/dense_26/BiasAddBiasAdd'sequential_12/dense_26/MatMul:product:05sequential_12/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
sequential_12/dense_26/SoftmaxSoftmax'sequential_12/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€w
IdentityIdentity(sequential_12/dense_26/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Е
NoOpNoOp/^sequential_12/conv2d_50/BiasAdd/ReadVariableOp.^sequential_12/conv2d_50/Conv2D/ReadVariableOp/^sequential_12/conv2d_51/BiasAdd/ReadVariableOp.^sequential_12/conv2d_51/Conv2D/ReadVariableOp/^sequential_12/conv2d_52/BiasAdd/ReadVariableOp.^sequential_12/conv2d_52/Conv2D/ReadVariableOp/^sequential_12/conv2d_53/BiasAdd/ReadVariableOp.^sequential_12/conv2d_53/Conv2D/ReadVariableOp/^sequential_12/conv2d_54/BiasAdd/ReadVariableOp.^sequential_12/conv2d_54/Conv2D/ReadVariableOp/^sequential_12/conv2d_55/BiasAdd/ReadVariableOp.^sequential_12/conv2d_55/Conv2D/ReadVariableOp.^sequential_12/dense_24/BiasAdd/ReadVariableOp-^sequential_12/dense_24/MatMul/ReadVariableOp.^sequential_12/dense_25/BiasAdd/ReadVariableOp-^sequential_12/dense_25/MatMul/ReadVariableOp.^sequential_12/dense_26/BiasAdd/ReadVariableOp-^sequential_12/dense_26/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : 2`
.sequential_12/conv2d_50/BiasAdd/ReadVariableOp.sequential_12/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_50/Conv2D/ReadVariableOp-sequential_12/conv2d_50/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_51/BiasAdd/ReadVariableOp.sequential_12/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_51/Conv2D/ReadVariableOp-sequential_12/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_52/BiasAdd/ReadVariableOp.sequential_12/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_52/Conv2D/ReadVariableOp-sequential_12/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_53/BiasAdd/ReadVariableOp.sequential_12/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_53/Conv2D/ReadVariableOp-sequential_12/conv2d_53/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_54/BiasAdd/ReadVariableOp.sequential_12/conv2d_54/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_54/Conv2D/ReadVariableOp-sequential_12/conv2d_54/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_55/BiasAdd/ReadVariableOp.sequential_12/conv2d_55/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_55/Conv2D/ReadVariableOp-sequential_12/conv2d_55/Conv2D/ReadVariableOp2^
-sequential_12/dense_24/BiasAdd/ReadVariableOp-sequential_12/dense_24/BiasAdd/ReadVariableOp2\
,sequential_12/dense_24/MatMul/ReadVariableOp,sequential_12/dense_24/MatMul/ReadVariableOp2^
-sequential_12/dense_25/BiasAdd/ReadVariableOp-sequential_12/dense_25/BiasAdd/ReadVariableOp2\
,sequential_12/dense_25/MatMul/ReadVariableOp,sequential_12/dense_25/MatMul/ReadVariableOp2^
-sequential_12/dense_26/BiasAdd/ReadVariableOp-sequential_12/dense_26/BiasAdd/ReadVariableOp2\
,sequential_12/dense_26/MatMul/ReadVariableOp,sequential_12/dense_26/MatMul/ReadVariableOp:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_6_input:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Њ
D
(__inference_re_lu_52_layer_call_fn_39637

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€==H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_52_layer_call_and_return_conditional_losses_39088h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€==H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€==H:W S
/
_output_shapes
:€€€€€€€€€==H
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_36979
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
в

э
D__inference_conv2d_50_layer_call_and_return_conditional_losses_39554

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ь
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ВВ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ДД: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€ДД
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
п
_
C__inference_re_lu_50_layer_call_and_return_conditional_losses_39044

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€ВВ d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ВВ :Y U
1
_output_shapes
:€€€€€€€€€ВВ 
 
_user_specified_nameinputs
ї
L
0__inference_max_pooling2d_50_layer_call_fn_39569

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_38960Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_39613

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ш
b
F__inference_rescaling_1_layer_call_and_return_conditional_losses_38624

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Яc
±
H__inference_sequential_12_layer_call_and_return_conditional_losses_39278
sequential_6_input)
conv2d_50_39219: 
conv2d_50_39221: )
conv2d_51_39226: H
conv2d_51_39228:H)
conv2d_52_39233:HH
conv2d_52_39235:H)
conv2d_53_39240:HH
conv2d_53_39242:H)
conv2d_54_39247:HH
conv2d_54_39249:H)
conv2d_55_39254:HH
conv2d_55_39256:H!
dense_24_39262:	†H
dense_24_39264:H 
dense_25_39267:HH
dense_25_39269:H 
dense_26_39272:H
dense_26_39274:
identityИҐ!conv2d_50/StatefulPartitionedCallҐ!conv2d_51/StatefulPartitionedCallҐ!conv2d_52/StatefulPartitionedCallҐ!conv2d_53/StatefulPartitionedCallҐ!conv2d_54/StatefulPartitionedCallҐ!conv2d_55/StatefulPartitionedCallҐ dense_24/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallЎ
sequential_6/PartitionedCallPartitionedCallsequential_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_38633л
sequential_7/PartitionedCallPartitionedCall%sequential_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_38906у
 zero_padding2d_7/PartitionedCallPartitionedCall%sequential_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ДД* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_38950°
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_7/PartitionedCall:output:0conv2d_50_39219conv2d_50_39221*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ВВ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_39034и
re_lu_50/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ВВ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_50_layer_call_and_return_conditional_losses_39044п
 max_pooling2d_50/PartitionedCallPartitionedCall!re_lu_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ББ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_38960Я
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0conv2d_51_39226conv2d_51_39228*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_39056ж
re_lu_51/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_51_layer_call_and_return_conditional_losses_39066н
 max_pooling2d_51/PartitionedCallPartitionedCall!re_lu_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€??H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_38970Я
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0conv2d_52_39233conv2d_52_39235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€==H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_52_layer_call_and_return_conditional_losses_39078ж
re_lu_52/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€==H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_52_layer_call_and_return_conditional_losses_39088н
 max_pooling2d_52/PartitionedCallPartitionedCall!re_lu_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_38980Я
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0conv2d_53_39240conv2d_53_39242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_39100ж
re_lu_53/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_53_layer_call_and_return_conditional_losses_39110н
 max_pooling2d_53/PartitionedCallPartitionedCall!re_lu_53/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_38990Я
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_53/PartitionedCall:output:0conv2d_54_39247conv2d_54_39249*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_54_layer_call_and_return_conditional_losses_39122ж
re_lu_54/PartitionedCallPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_54_layer_call_and_return_conditional_losses_39132н
 max_pooling2d_54/PartitionedCallPartitionedCall!re_lu_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_39000Я
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_54/PartitionedCall:output:0conv2d_55_39254conv2d_55_39256*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_55_layer_call_and_return_conditional_losses_39144ж
re_lu_55/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_55_layer_call_and_return_conditional_losses_39154н
 max_pooling2d_55/PartitionedCallPartitionedCall!re_lu_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_39010а
flatten_4/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_39162М
 dense_24/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_24_39262dense_24_39264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_39174У
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_39267dense_25_39269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_39190У
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_39272dense_26_39274*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_39206x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€г
NoOpNoOp"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : 2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_6_input:%!

_user_specified_name39219:%!

_user_specified_name39221:%!

_user_specified_name39226:%!

_user_specified_name39228:%!

_user_specified_name39233:%!

_user_specified_name39235:%!

_user_specified_name39240:%!

_user_specified_name39242:%	!

_user_specified_name39247:%
!

_user_specified_name39249:%!

_user_specified_name39254:%!

_user_specified_name39256:%!

_user_specified_name39262:%!

_user_specified_name39264:%!

_user_specified_name39267:%!

_user_specified_name39269:%!

_user_specified_name39272:%!

_user_specified_name39274
Ю
¬
-__inference_sequential_12_layer_call_fn_39323
sequential_6_input
unknown:	
	unknown_0:	#
	unknown_1: 
	unknown_2: #
	unknown_3: H
	unknown_4:H#
	unknown_5:HH
	unknown_6:H#
	unknown_7:HH
	unknown_8:H#
	unknown_9:HH

unknown_10:H$

unknown_11:HH

unknown_12:H

unknown_13:	†H

unknown_14:H

unknown_15:HH

unknown_16:H

unknown_17:H

unknown_18:
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallsequential_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_39213o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_6_input:%!

_user_specified_name39281:%!

_user_specified_name39283:%!

_user_specified_name39285:%!

_user_specified_name39287:%!

_user_specified_name39289:%!

_user_specified_name39291:%!

_user_specified_name39293:%!

_user_specified_name39295:%	!

_user_specified_name39297:%
!

_user_specified_name39299:%!

_user_specified_name39301:%!

_user_specified_name39303:%!

_user_specified_name39305:%!

_user_specified_name39307:%!

_user_specified_name39309:%!

_user_specified_name39311:%!

_user_specified_name39313:%!

_user_specified_name39315:%!

_user_specified_name39317:%!

_user_specified_name39319
∆
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_39162

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€†Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€†"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_37039
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:H: *
	_noinline(:D @

_output_shapes
:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
р
Ц
(__inference_dense_24_layer_call_fn_39789

inputs
unknown:	†H
	unknown_0:H
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_39174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€H<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs:%!

_user_specified_name39783:%!

_user_specified_name39785
Ў

э
D__inference_conv2d_54_layer_call_and_return_conditional_losses_39122

inputs8
conv2d_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ђ’
§9
__inference__traced_save_40529
file_prefixA
'read_disablecopyonread_conv2d_50_kernel: 5
'read_1_disablecopyonread_conv2d_50_bias: C
)read_2_disablecopyonread_conv2d_51_kernel: H5
'read_3_disablecopyonread_conv2d_51_bias:HC
)read_4_disablecopyonread_conv2d_52_kernel:HH5
'read_5_disablecopyonread_conv2d_52_bias:HC
)read_6_disablecopyonread_conv2d_53_kernel:HH5
'read_7_disablecopyonread_conv2d_53_bias:HC
)read_8_disablecopyonread_conv2d_54_kernel:HH5
'read_9_disablecopyonread_conv2d_54_bias:HD
*read_10_disablecopyonread_conv2d_55_kernel:HH6
(read_11_disablecopyonread_conv2d_55_bias:H<
)read_12_disablecopyonread_dense_24_kernel:	†H5
'read_13_disablecopyonread_dense_24_bias:H;
)read_14_disablecopyonread_dense_25_kernel:HH5
'read_15_disablecopyonread_dense_25_bias:H;
)read_16_disablecopyonread_dense_26_kernel:H5
'read_17_disablecopyonread_dense_26_bias:-
#read_18_disablecopyonread_iteration:	 1
'read_19_disablecopyonread_learning_rate: K
1read_20_disablecopyonread_adam_m_conv2d_50_kernel: K
1read_21_disablecopyonread_adam_v_conv2d_50_kernel: =
/read_22_disablecopyonread_adam_m_conv2d_50_bias: =
/read_23_disablecopyonread_adam_v_conv2d_50_bias: K
1read_24_disablecopyonread_adam_m_conv2d_51_kernel: HK
1read_25_disablecopyonread_adam_v_conv2d_51_kernel: H=
/read_26_disablecopyonread_adam_m_conv2d_51_bias:H=
/read_27_disablecopyonread_adam_v_conv2d_51_bias:HK
1read_28_disablecopyonread_adam_m_conv2d_52_kernel:HHK
1read_29_disablecopyonread_adam_v_conv2d_52_kernel:HH=
/read_30_disablecopyonread_adam_m_conv2d_52_bias:H=
/read_31_disablecopyonread_adam_v_conv2d_52_bias:HK
1read_32_disablecopyonread_adam_m_conv2d_53_kernel:HHK
1read_33_disablecopyonread_adam_v_conv2d_53_kernel:HH=
/read_34_disablecopyonread_adam_m_conv2d_53_bias:H=
/read_35_disablecopyonread_adam_v_conv2d_53_bias:HK
1read_36_disablecopyonread_adam_m_conv2d_54_kernel:HHK
1read_37_disablecopyonread_adam_v_conv2d_54_kernel:HH=
/read_38_disablecopyonread_adam_m_conv2d_54_bias:H=
/read_39_disablecopyonread_adam_v_conv2d_54_bias:HK
1read_40_disablecopyonread_adam_m_conv2d_55_kernel:HHK
1read_41_disablecopyonread_adam_v_conv2d_55_kernel:HH=
/read_42_disablecopyonread_adam_m_conv2d_55_bias:H=
/read_43_disablecopyonread_adam_v_conv2d_55_bias:HC
0read_44_disablecopyonread_adam_m_dense_24_kernel:	†HC
0read_45_disablecopyonread_adam_v_dense_24_kernel:	†H<
.read_46_disablecopyonread_adam_m_dense_24_bias:H<
.read_47_disablecopyonread_adam_v_dense_24_bias:HB
0read_48_disablecopyonread_adam_m_dense_25_kernel:HHB
0read_49_disablecopyonread_adam_v_dense_25_kernel:HH<
.read_50_disablecopyonread_adam_m_dense_25_bias:H<
.read_51_disablecopyonread_adam_v_dense_25_bias:HB
0read_52_disablecopyonread_adam_m_dense_26_kernel:HB
0read_53_disablecopyonread_adam_v_dense_26_kernel:H<
.read_54_disablecopyonread_adam_m_dense_26_bias:<
.read_55_disablecopyonread_adam_v_dense_26_bias:+
!read_56_disablecopyonread_total_1: +
!read_57_disablecopyonread_count_1: )
read_58_disablecopyonread_total: )
read_59_disablecopyonread_count: >
0read_60_disablecopyonread_random_flip_1_statevar:	B
4read_61_disablecopyonread_random_rotation_1_statevar:	
savev2_const
identity_125ИҐMergeV2CheckpointsҐRead/DisableCopyOnReadҐRead/ReadVariableOpҐRead_1/DisableCopyOnReadҐRead_1/ReadVariableOpҐRead_10/DisableCopyOnReadҐRead_10/ReadVariableOpҐRead_11/DisableCopyOnReadҐRead_11/ReadVariableOpҐRead_12/DisableCopyOnReadҐRead_12/ReadVariableOpҐRead_13/DisableCopyOnReadҐRead_13/ReadVariableOpҐRead_14/DisableCopyOnReadҐRead_14/ReadVariableOpҐRead_15/DisableCopyOnReadҐRead_15/ReadVariableOpҐRead_16/DisableCopyOnReadҐRead_16/ReadVariableOpҐRead_17/DisableCopyOnReadҐRead_17/ReadVariableOpҐRead_18/DisableCopyOnReadҐRead_18/ReadVariableOpҐRead_19/DisableCopyOnReadҐRead_19/ReadVariableOpҐRead_2/DisableCopyOnReadҐRead_2/ReadVariableOpҐRead_20/DisableCopyOnReadҐRead_20/ReadVariableOpҐRead_21/DisableCopyOnReadҐRead_21/ReadVariableOpҐRead_22/DisableCopyOnReadҐRead_22/ReadVariableOpҐRead_23/DisableCopyOnReadҐRead_23/ReadVariableOpҐRead_24/DisableCopyOnReadҐRead_24/ReadVariableOpҐRead_25/DisableCopyOnReadҐRead_25/ReadVariableOpҐRead_26/DisableCopyOnReadҐRead_26/ReadVariableOpҐRead_27/DisableCopyOnReadҐRead_27/ReadVariableOpҐRead_28/DisableCopyOnReadҐRead_28/ReadVariableOpҐRead_29/DisableCopyOnReadҐRead_29/ReadVariableOpҐRead_3/DisableCopyOnReadҐRead_3/ReadVariableOpҐRead_30/DisableCopyOnReadҐRead_30/ReadVariableOpҐRead_31/DisableCopyOnReadҐRead_31/ReadVariableOpҐRead_32/DisableCopyOnReadҐRead_32/ReadVariableOpҐRead_33/DisableCopyOnReadҐRead_33/ReadVariableOpҐRead_34/DisableCopyOnReadҐRead_34/ReadVariableOpҐRead_35/DisableCopyOnReadҐRead_35/ReadVariableOpҐRead_36/DisableCopyOnReadҐRead_36/ReadVariableOpҐRead_37/DisableCopyOnReadҐRead_37/ReadVariableOpҐRead_38/DisableCopyOnReadҐRead_38/ReadVariableOpҐRead_39/DisableCopyOnReadҐRead_39/ReadVariableOpҐRead_4/DisableCopyOnReadҐRead_4/ReadVariableOpҐRead_40/DisableCopyOnReadҐRead_40/ReadVariableOpҐRead_41/DisableCopyOnReadҐRead_41/ReadVariableOpҐRead_42/DisableCopyOnReadҐRead_42/ReadVariableOpҐRead_43/DisableCopyOnReadҐRead_43/ReadVariableOpҐRead_44/DisableCopyOnReadҐRead_44/ReadVariableOpҐRead_45/DisableCopyOnReadҐRead_45/ReadVariableOpҐRead_46/DisableCopyOnReadҐRead_46/ReadVariableOpҐRead_47/DisableCopyOnReadҐRead_47/ReadVariableOpҐRead_48/DisableCopyOnReadҐRead_48/ReadVariableOpҐRead_49/DisableCopyOnReadҐRead_49/ReadVariableOpҐRead_5/DisableCopyOnReadҐRead_5/ReadVariableOpҐRead_50/DisableCopyOnReadҐRead_50/ReadVariableOpҐRead_51/DisableCopyOnReadҐRead_51/ReadVariableOpҐRead_52/DisableCopyOnReadҐRead_52/ReadVariableOpҐRead_53/DisableCopyOnReadҐRead_53/ReadVariableOpҐRead_54/DisableCopyOnReadҐRead_54/ReadVariableOpҐRead_55/DisableCopyOnReadҐRead_55/ReadVariableOpҐRead_56/DisableCopyOnReadҐRead_56/ReadVariableOpҐRead_57/DisableCopyOnReadҐRead_57/ReadVariableOpҐRead_58/DisableCopyOnReadҐRead_58/ReadVariableOpҐRead_59/DisableCopyOnReadҐRead_59/ReadVariableOpҐRead_6/DisableCopyOnReadҐRead_6/ReadVariableOpҐRead_60/DisableCopyOnReadҐRead_60/ReadVariableOpҐRead_61/DisableCopyOnReadҐRead_61/ReadVariableOpҐRead_7/DisableCopyOnReadҐRead_7/ReadVariableOpҐRead_8/DisableCopyOnReadҐRead_8/ReadVariableOpҐRead_9/DisableCopyOnReadҐRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_conv2d_50_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_conv2d_50_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: {
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_conv2d_50_bias"/device:CPU:0*
_output_shapes
 £
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_conv2d_50_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_conv2d_51_kernel"/device:CPU:0*
_output_shapes
 ±
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_conv2d_51_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: H*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: Hk

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: H{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_conv2d_51_bias"/device:CPU:0*
_output_shapes
 £
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_conv2d_51_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:H_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:H}
Read_4/DisableCopyOnReadDisableCopyOnRead)read_4_disablecopyonread_conv2d_52_kernel"/device:CPU:0*
_output_shapes
 ±
Read_4/ReadVariableOpReadVariableOp)read_4_disablecopyonread_conv2d_52_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHk

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:HH{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_conv2d_52_bias"/device:CPU:0*
_output_shapes
 £
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_conv2d_52_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:H}
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_conv2d_53_kernel"/device:CPU:0*
_output_shapes
 ±
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_conv2d_53_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:HH{
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_conv2d_53_bias"/device:CPU:0*
_output_shapes
 £
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_conv2d_53_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:H}
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_conv2d_54_kernel"/device:CPU:0*
_output_shapes
 ±
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_conv2d_54_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:HH{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_conv2d_54_bias"/device:CPU:0*
_output_shapes
 £
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_conv2d_54_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:H
Read_10/DisableCopyOnReadDisableCopyOnRead*read_10_disablecopyonread_conv2d_55_kernel"/device:CPU:0*
_output_shapes
 і
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_conv2d_55_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:HH}
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_conv2d_55_bias"/device:CPU:0*
_output_shapes
 ¶
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_conv2d_55_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:H~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_dense_24_kernel"/device:CPU:0*
_output_shapes
 ђ
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_dense_24_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	†H*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	†Hf
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	†H|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_dense_24_bias"/device:CPU:0*
_output_shapes
 •
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_dense_24_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:H~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_dense_25_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_dense_25_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:HH*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:HHe
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:HH|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_dense_25_bias"/device:CPU:0*
_output_shapes
 •
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_dense_25_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:H~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_dense_26_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_dense_26_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:H*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:He
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:H|
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_dense_26_bias"/device:CPU:0*
_output_shapes
 •
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_dense_26_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_18/DisableCopyOnReadDisableCopyOnRead#read_18_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Э
Read_18/ReadVariableOpReadVariableOp#read_18_disablecopyonread_iteration^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 °
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_learning_rate^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: Ж
Read_20/DisableCopyOnReadDisableCopyOnRead1read_20_disablecopyonread_adam_m_conv2d_50_kernel"/device:CPU:0*
_output_shapes
 ї
Read_20/ReadVariableOpReadVariableOp1read_20_disablecopyonread_adam_m_conv2d_50_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
: Ж
Read_21/DisableCopyOnReadDisableCopyOnRead1read_21_disablecopyonread_adam_v_conv2d_50_kernel"/device:CPU:0*
_output_shapes
 ї
Read_21/ReadVariableOpReadVariableOp1read_21_disablecopyonread_adam_v_conv2d_50_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*&
_output_shapes
: Д
Read_22/DisableCopyOnReadDisableCopyOnRead/read_22_disablecopyonread_adam_m_conv2d_50_bias"/device:CPU:0*
_output_shapes
 ≠
Read_22/ReadVariableOpReadVariableOp/read_22_disablecopyonread_adam_m_conv2d_50_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: Д
Read_23/DisableCopyOnReadDisableCopyOnRead/read_23_disablecopyonread_adam_v_conv2d_50_bias"/device:CPU:0*
_output_shapes
 ≠
Read_23/ReadVariableOpReadVariableOp/read_23_disablecopyonread_adam_v_conv2d_50_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: Ж
Read_24/DisableCopyOnReadDisableCopyOnRead1read_24_disablecopyonread_adam_m_conv2d_51_kernel"/device:CPU:0*
_output_shapes
 ї
Read_24/ReadVariableOpReadVariableOp1read_24_disablecopyonread_adam_m_conv2d_51_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: H*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: Hm
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
: HЖ
Read_25/DisableCopyOnReadDisableCopyOnRead1read_25_disablecopyonread_adam_v_conv2d_51_kernel"/device:CPU:0*
_output_shapes
 ї
Read_25/ReadVariableOpReadVariableOp1read_25_disablecopyonread_adam_v_conv2d_51_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: H*
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: Hm
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
: HД
Read_26/DisableCopyOnReadDisableCopyOnRead/read_26_disablecopyonread_adam_m_conv2d_51_bias"/device:CPU:0*
_output_shapes
 ≠
Read_26/ReadVariableOpReadVariableOp/read_26_disablecopyonread_adam_m_conv2d_51_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:HД
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_adam_v_conv2d_51_bias"/device:CPU:0*
_output_shapes
 ≠
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_adam_v_conv2d_51_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:HЖ
Read_28/DisableCopyOnReadDisableCopyOnRead1read_28_disablecopyonread_adam_m_conv2d_52_kernel"/device:CPU:0*
_output_shapes
 ї
Read_28/ReadVariableOpReadVariableOp1read_28_disablecopyonread_adam_m_conv2d_52_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0w
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*&
_output_shapes
:HHЖ
Read_29/DisableCopyOnReadDisableCopyOnRead1read_29_disablecopyonread_adam_v_conv2d_52_kernel"/device:CPU:0*
_output_shapes
 ї
Read_29/ReadVariableOpReadVariableOp1read_29_disablecopyonread_adam_v_conv2d_52_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0w
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*&
_output_shapes
:HHД
Read_30/DisableCopyOnReadDisableCopyOnRead/read_30_disablecopyonread_adam_m_conv2d_52_bias"/device:CPU:0*
_output_shapes
 ≠
Read_30/ReadVariableOpReadVariableOp/read_30_disablecopyonread_adam_m_conv2d_52_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:HД
Read_31/DisableCopyOnReadDisableCopyOnRead/read_31_disablecopyonread_adam_v_conv2d_52_bias"/device:CPU:0*
_output_shapes
 ≠
Read_31/ReadVariableOpReadVariableOp/read_31_disablecopyonread_adam_v_conv2d_52_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:HЖ
Read_32/DisableCopyOnReadDisableCopyOnRead1read_32_disablecopyonread_adam_m_conv2d_53_kernel"/device:CPU:0*
_output_shapes
 ї
Read_32/ReadVariableOpReadVariableOp1read_32_disablecopyonread_adam_m_conv2d_53_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
:HHЖ
Read_33/DisableCopyOnReadDisableCopyOnRead1read_33_disablecopyonread_adam_v_conv2d_53_kernel"/device:CPU:0*
_output_shapes
 ї
Read_33/ReadVariableOpReadVariableOp1read_33_disablecopyonread_adam_v_conv2d_53_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0w
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*&
_output_shapes
:HHД
Read_34/DisableCopyOnReadDisableCopyOnRead/read_34_disablecopyonread_adam_m_conv2d_53_bias"/device:CPU:0*
_output_shapes
 ≠
Read_34/ReadVariableOpReadVariableOp/read_34_disablecopyonread_adam_m_conv2d_53_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:HД
Read_35/DisableCopyOnReadDisableCopyOnRead/read_35_disablecopyonread_adam_v_conv2d_53_bias"/device:CPU:0*
_output_shapes
 ≠
Read_35/ReadVariableOpReadVariableOp/read_35_disablecopyonread_adam_v_conv2d_53_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:HЖ
Read_36/DisableCopyOnReadDisableCopyOnRead1read_36_disablecopyonread_adam_m_conv2d_54_kernel"/device:CPU:0*
_output_shapes
 ї
Read_36/ReadVariableOpReadVariableOp1read_36_disablecopyonread_adam_m_conv2d_54_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
:HHЖ
Read_37/DisableCopyOnReadDisableCopyOnRead1read_37_disablecopyonread_adam_v_conv2d_54_kernel"/device:CPU:0*
_output_shapes
 ї
Read_37/ReadVariableOpReadVariableOp1read_37_disablecopyonread_adam_v_conv2d_54_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0w
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*&
_output_shapes
:HHД
Read_38/DisableCopyOnReadDisableCopyOnRead/read_38_disablecopyonread_adam_m_conv2d_54_bias"/device:CPU:0*
_output_shapes
 ≠
Read_38/ReadVariableOpReadVariableOp/read_38_disablecopyonread_adam_m_conv2d_54_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:HД
Read_39/DisableCopyOnReadDisableCopyOnRead/read_39_disablecopyonread_adam_v_conv2d_54_bias"/device:CPU:0*
_output_shapes
 ≠
Read_39/ReadVariableOpReadVariableOp/read_39_disablecopyonread_adam_v_conv2d_54_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:HЖ
Read_40/DisableCopyOnReadDisableCopyOnRead1read_40_disablecopyonread_adam_m_conv2d_55_kernel"/device:CPU:0*
_output_shapes
 ї
Read_40/ReadVariableOpReadVariableOp1read_40_disablecopyonread_adam_m_conv2d_55_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0w
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*&
_output_shapes
:HHЖ
Read_41/DisableCopyOnReadDisableCopyOnRead1read_41_disablecopyonread_adam_v_conv2d_55_kernel"/device:CPU:0*
_output_shapes
 ї
Read_41/ReadVariableOpReadVariableOp1read_41_disablecopyonread_adam_v_conv2d_55_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:HH*
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:HHm
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
:HHД
Read_42/DisableCopyOnReadDisableCopyOnRead/read_42_disablecopyonread_adam_m_conv2d_55_bias"/device:CPU:0*
_output_shapes
 ≠
Read_42/ReadVariableOpReadVariableOp/read_42_disablecopyonread_adam_m_conv2d_55_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:HД
Read_43/DisableCopyOnReadDisableCopyOnRead/read_43_disablecopyonread_adam_v_conv2d_55_bias"/device:CPU:0*
_output_shapes
 ≠
Read_43/ReadVariableOpReadVariableOp/read_43_disablecopyonread_adam_v_conv2d_55_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:HЕ
Read_44/DisableCopyOnReadDisableCopyOnRead0read_44_disablecopyonread_adam_m_dense_24_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_44/ReadVariableOpReadVariableOp0read_44_disablecopyonread_adam_m_dense_24_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	†H*
dtype0p
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	†Hf
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:	†HЕ
Read_45/DisableCopyOnReadDisableCopyOnRead0read_45_disablecopyonread_adam_v_dense_24_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_45/ReadVariableOpReadVariableOp0read_45_disablecopyonread_adam_v_dense_24_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	†H*
dtype0p
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	†Hf
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:	†HГ
Read_46/DisableCopyOnReadDisableCopyOnRead.read_46_disablecopyonread_adam_m_dense_24_bias"/device:CPU:0*
_output_shapes
 ђ
Read_46/ReadVariableOpReadVariableOp.read_46_disablecopyonread_adam_m_dense_24_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:HГ
Read_47/DisableCopyOnReadDisableCopyOnRead.read_47_disablecopyonread_adam_v_dense_24_bias"/device:CPU:0*
_output_shapes
 ђ
Read_47/ReadVariableOpReadVariableOp.read_47_disablecopyonread_adam_v_dense_24_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ha
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:HЕ
Read_48/DisableCopyOnReadDisableCopyOnRead0read_48_disablecopyonread_adam_m_dense_25_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_48/ReadVariableOpReadVariableOp0read_48_disablecopyonread_adam_m_dense_25_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:HH*
dtype0o
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:HHe
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes

:HHЕ
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_v_dense_25_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_v_dense_25_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:HH*
dtype0o
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:HHe
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes

:HHГ
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_adam_m_dense_25_bias"/device:CPU:0*
_output_shapes
 ђ
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_adam_m_dense_25_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Hc
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:HГ
Read_51/DisableCopyOnReadDisableCopyOnRead.read_51_disablecopyonread_adam_v_dense_25_bias"/device:CPU:0*
_output_shapes
 ђ
Read_51/ReadVariableOpReadVariableOp.read_51_disablecopyonread_adam_v_dense_25_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:H*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Hc
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:HЕ
Read_52/DisableCopyOnReadDisableCopyOnRead0read_52_disablecopyonread_adam_m_dense_26_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_52/ReadVariableOpReadVariableOp0read_52_disablecopyonread_adam_m_dense_26_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:H*
dtype0p
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Hg
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes

:HЕ
Read_53/DisableCopyOnReadDisableCopyOnRead0read_53_disablecopyonread_adam_v_dense_26_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_53/ReadVariableOpReadVariableOp0read_53_disablecopyonread_adam_v_dense_26_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:H*
dtype0p
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Hg
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes

:HГ
Read_54/DisableCopyOnReadDisableCopyOnRead.read_54_disablecopyonread_adam_m_dense_26_bias"/device:CPU:0*
_output_shapes
 ђ
Read_54/ReadVariableOpReadVariableOp.read_54_disablecopyonread_adam_m_dense_26_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:Г
Read_55/DisableCopyOnReadDisableCopyOnRead.read_55_disablecopyonread_adam_v_dense_26_bias"/device:CPU:0*
_output_shapes
 ђ
Read_55/ReadVariableOpReadVariableOp.read_55_disablecopyonread_adam_v_dense_26_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_56/DisableCopyOnReadDisableCopyOnRead!read_56_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_56/ReadVariableOpReadVariableOp!read_56_disablecopyonread_total_1^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_57/DisableCopyOnReadDisableCopyOnRead!read_57_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_57/ReadVariableOpReadVariableOp!read_57_disablecopyonread_count_1^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_58/DisableCopyOnReadDisableCopyOnReadread_58_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_58/ReadVariableOpReadVariableOpread_58_disablecopyonread_total^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_59/DisableCopyOnReadDisableCopyOnReadread_59_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_59/ReadVariableOpReadVariableOpread_59_disablecopyonread_count^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_60/DisableCopyOnReadDisableCopyOnRead0read_60_disablecopyonread_random_flip_1_statevar"/device:CPU:0*
_output_shapes
 Ѓ
Read_60/ReadVariableOpReadVariableOp0read_60_disablecopyonread_random_flip_1_statevar^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0	*
_output_shapes
:Й
Read_61/DisableCopyOnReadDisableCopyOnRead4read_61_disablecopyonread_random_rotation_1_statevar"/device:CPU:0*
_output_shapes
 ≤
Read_61/ReadVariableOpReadVariableOp4read_61_disablecopyonread_random_rotation_1_statevar^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0	*
_output_shapes
:Ц
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*њ
valueµB≤?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHо
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*У
valueЙBЖ?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B с
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *M
dtypesC
A2?			Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_124Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_125IdentityIdentity_124:output:0^NoOp*
T0*
_output_shapes
: х
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_125Identity_125:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_user_specified_nameconv2d_50/kernel:.*
(
_user_specified_nameconv2d_50/bias:0,
*
_user_specified_nameconv2d_51/kernel:.*
(
_user_specified_nameconv2d_51/bias:0,
*
_user_specified_nameconv2d_52/kernel:.*
(
_user_specified_nameconv2d_52/bias:0,
*
_user_specified_nameconv2d_53/kernel:.*
(
_user_specified_nameconv2d_53/bias:0	,
*
_user_specified_nameconv2d_54/kernel:.
*
(
_user_specified_nameconv2d_54/bias:0,
*
_user_specified_nameconv2d_55/kernel:.*
(
_user_specified_nameconv2d_55/bias:/+
)
_user_specified_namedense_24/kernel:-)
'
_user_specified_namedense_24/bias:/+
)
_user_specified_namedense_25/kernel:-)
'
_user_specified_namedense_25/bias:/+
)
_user_specified_namedense_26/kernel:-)
'
_user_specified_namedense_26/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:73
1
_user_specified_nameAdam/m/conv2d_50/kernel:73
1
_user_specified_nameAdam/v/conv2d_50/kernel:51
/
_user_specified_nameAdam/m/conv2d_50/bias:51
/
_user_specified_nameAdam/v/conv2d_50/bias:73
1
_user_specified_nameAdam/m/conv2d_51/kernel:73
1
_user_specified_nameAdam/v/conv2d_51/kernel:51
/
_user_specified_nameAdam/m/conv2d_51/bias:51
/
_user_specified_nameAdam/v/conv2d_51/bias:73
1
_user_specified_nameAdam/m/conv2d_52/kernel:73
1
_user_specified_nameAdam/v/conv2d_52/kernel:51
/
_user_specified_nameAdam/m/conv2d_52/bias:5 1
/
_user_specified_nameAdam/v/conv2d_52/bias:7!3
1
_user_specified_nameAdam/m/conv2d_53/kernel:7"3
1
_user_specified_nameAdam/v/conv2d_53/kernel:5#1
/
_user_specified_nameAdam/m/conv2d_53/bias:5$1
/
_user_specified_nameAdam/v/conv2d_53/bias:7%3
1
_user_specified_nameAdam/m/conv2d_54/kernel:7&3
1
_user_specified_nameAdam/v/conv2d_54/kernel:5'1
/
_user_specified_nameAdam/m/conv2d_54/bias:5(1
/
_user_specified_nameAdam/v/conv2d_54/bias:7)3
1
_user_specified_nameAdam/m/conv2d_55/kernel:7*3
1
_user_specified_nameAdam/v/conv2d_55/kernel:5+1
/
_user_specified_nameAdam/m/conv2d_55/bias:5,1
/
_user_specified_nameAdam/v/conv2d_55/bias:6-2
0
_user_specified_nameAdam/m/dense_24/kernel:6.2
0
_user_specified_nameAdam/v/dense_24/kernel:4/0
.
_user_specified_nameAdam/m/dense_24/bias:400
.
_user_specified_nameAdam/v/dense_24/bias:612
0
_user_specified_nameAdam/m/dense_25/kernel:622
0
_user_specified_nameAdam/v/dense_25/kernel:430
.
_user_specified_nameAdam/m/dense_25/bias:440
.
_user_specified_nameAdam/v/dense_25/bias:652
0
_user_specified_nameAdam/m/dense_26/kernel:662
0
_user_specified_nameAdam/v/dense_26/kernel:470
.
_user_specified_nameAdam/m/dense_26/bias:480
.
_user_specified_nameAdam/v/dense_26/bias:'9#
!
_user_specified_name	total_1:':#
!
_user_specified_name	count_1:%;!

_user_specified_nametotal:%<!

_user_specified_namecount:6=2
0
_user_specified_namerandom_flip_1/StateVar::>6
4
_user_specified_namerandom_rotation_1/StateVar:=?9

_output_shapes
: 

_user_specified_nameConst
Ї
O
"__inference__update_step_xla_37034
gradient
variable:	†H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	†H: *
	_noinline(:I E

_output_shapes
:	†H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ю
О
-__inference_sequential_12_layer_call_fn_39364
sequential_6_input!
unknown: 
	unknown_0: #
	unknown_1: H
	unknown_2:H#
	unknown_3:HH
	unknown_4:H#
	unknown_5:HH
	unknown_6:H#
	unknown_7:HH
	unknown_8:H#
	unknown_9:HH

unknown_10:H

unknown_11:	†H

unknown_12:H

unknown_13:HH

unknown_14:H

unknown_15:H

unknown_16:
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallsequential_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_39278o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_6_input:%!

_user_specified_name39326:%!

_user_specified_name39328:%!

_user_specified_name39330:%!

_user_specified_name39332:%!

_user_specified_name39334:%!

_user_specified_name39336:%!

_user_specified_name39338:%!

_user_specified_name39340:%	!

_user_specified_name39342:%
!

_user_specified_name39344:%!

_user_specified_name39346:%!

_user_specified_name39348:%!

_user_specified_name39350:%!

_user_specified_name39352:%!

_user_specified_name39354:%!

_user_specified_name39356:%!

_user_specified_name39358:%!

_user_specified_name39360
м
R
,__inference_sequential_6_layer_call_fn_38643
resizing_1_input
identity…
PartitionedCallPartitionedCallresizing_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_38633j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:c _
1
_output_shapes
:€€€€€€€€€АА
*
_user_specified_nameresizing_1_input
в

э
D__inference_conv2d_50_layer_call_and_return_conditional_losses_39034

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ь
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ВВ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ДД: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€ДД
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
 

ф
C__inference_dense_25_layer_call_and_return_conditional_losses_39820

inputs0
matmul_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:HH*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€HP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ha
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
V
"__inference__update_step_xla_37004
gradient"
variable:HH*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:HH: *
	_noinline(:P L
&
_output_shapes
:HH
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
—
Б
1__inference_random_rotation_1_layer_call_fn_39998

inputs
unknown:	
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38887y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs:%!

_user_specified_name39994
У
g
K__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_39730

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
х
U
,__inference_sequential_7_layer_call_fn_38920
random_flip_1_input
identityћ
PartitionedCallPartitionedCallrandom_flip_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_38906j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:f b
1
_output_shapes
:€€€€€€€€€АА
-
_user_specified_namerandom_flip_1_input
н
Х
(__inference_dense_25_layer_call_fn_39809

inputs
unknown:HH
	unknown_0:H
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_39190o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€H<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€H: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:%!

_user_specified_name39803:%!

_user_specified_name39805
ї
L
0__inference_max_pooling2d_54_layer_call_fn_39725

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_39000Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѕ

ф
C__inference_dense_26_layer_call_and_return_conditional_losses_39840

inputs0
matmul_readvariableop_resource:H-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ћ
Д
#__inference_signature_wrapper_39524
sequential_6_input!
unknown: 
	unknown_0: #
	unknown_1: H
	unknown_2:H#
	unknown_3:HH
	unknown_4:H#
	unknown_5:HH
	unknown_6:H#
	unknown_7:HH
	unknown_8:H#
	unknown_9:HH

unknown_10:H

unknown_11:	†H

unknown_12:H

unknown_13:HH

unknown_14:H

unknown_15:H

unknown_16:
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallsequential_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_38607o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_6_input:%!

_user_specified_name39486:%!

_user_specified_name39488:%!

_user_specified_name39490:%!

_user_specified_name39492:%!

_user_specified_name39494:%!

_user_specified_name39496:%!

_user_specified_name39498:%!

_user_specified_name39500:%	!

_user_specified_name39502:%
!

_user_specified_name39504:%!

_user_specified_name39506:%!

_user_specified_name39508:%!

_user_specified_name39510:%!

_user_specified_name39512:%!

_user_specified_name39514:%!

_user_specified_name39516:%!

_user_specified_name39518:%!

_user_specified_name39520
ќ

х
C__inference_dense_24_layer_call_and_return_conditional_losses_39174

inputs1
matmul_readvariableop_resource:	†H-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	†H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€HP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ha
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ч
Ю
)__inference_conv2d_53_layer_call_fn_39661

inputs!
unknown:HH
	unknown_0:H
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_39100w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€H<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€H: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:%!

_user_specified_name39655:%!

_user_specified_name39657
Ъ
h
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38903

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ж	
m
G__inference_sequential_6_layer_call_and_return_conditional_losses_38633
resizing_1_input
identity“
resizing_1/PartitionedCallPartitionedCallresizing_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_resizing_1_layer_call_and_return_conditional_losses_38615з
rescaling_1/PartitionedCallPartitionedCall#resizing_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_rescaling_1_layer_call_and_return_conditional_losses_38624v
IdentityIdentity$rescaling_1/PartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:c _
1
_output_shapes
:€€€€€€€€€АА
*
_user_specified_nameresizing_1_input
Ж	
m
G__inference_sequential_6_layer_call_and_return_conditional_losses_38627
resizing_1_input
identity“
resizing_1/PartitionedCallPartitionedCallresizing_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_resizing_1_layer_call_and_return_conditional_losses_38615з
rescaling_1/PartitionedCallPartitionedCall#resizing_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_rescaling_1_layer_call_and_return_conditional_losses_38624v
IdentityIdentity$rescaling_1/PartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:c _
1
_output_shapes
:€€€€€€€€€АА
*
_user_specified_nameresizing_1_input
Ъ
h
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40125

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
ѓ
a
E__inference_resizing_1_layer_call_and_return_conditional_losses_39851

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      Ы
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_38990

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ў

э
D__inference_conv2d_52_layer_call_and_return_conditional_losses_39078

inputs8
conv2d_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:HH*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€==H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€==Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€==HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€??H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€??H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ї
L
0__inference_max_pooling2d_52_layer_call_fn_39647

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_38980Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Њ
D
(__inference_re_lu_51_layer_call_fn_39598

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_51_layer_call_and_return_conditional_losses_39066h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
п
_
C__inference_re_lu_50_layer_call_and_return_conditional_losses_39564

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€ВВ d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ВВ :Y U
1
_output_shapes
:€€€€€€€€€ВВ 
 
_user_specified_nameinputs
ѕ
V
"__inference__update_step_xla_37024
gradient"
variable:HH*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:HH: *
	_noinline(:P L
&
_output_shapes
:HH
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
 

ф
C__inference_dense_25_layer_call_and_return_conditional_losses_39190

inputs0
matmul_readvariableop_resource:HH-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:HH*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€HP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ha
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
V
"__inference__update_step_xla_37014
gradient"
variable:HH*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:HH: *
	_noinline(:P L
&
_output_shapes
:HH
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Њ
D
(__inference_re_lu_53_layer_call_fn_39676

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_53_layer_call_and_return_conditional_losses_39110h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€H:W S
/
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs
∆
D
(__inference_re_lu_50_layer_call_fn_39559

inputs
identityї
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ВВ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_50_layer_call_and_return_conditional_losses_39044j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€ВВ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ВВ :Y U
1
_output_shapes
:€€€€€€€€€ВВ 
 
_user_specified_nameinputs
НЭ
Х'
!__inference__traced_restore_40724
file_prefix;
!assignvariableop_conv2d_50_kernel: /
!assignvariableop_1_conv2d_50_bias: =
#assignvariableop_2_conv2d_51_kernel: H/
!assignvariableop_3_conv2d_51_bias:H=
#assignvariableop_4_conv2d_52_kernel:HH/
!assignvariableop_5_conv2d_52_bias:H=
#assignvariableop_6_conv2d_53_kernel:HH/
!assignvariableop_7_conv2d_53_bias:H=
#assignvariableop_8_conv2d_54_kernel:HH/
!assignvariableop_9_conv2d_54_bias:H>
$assignvariableop_10_conv2d_55_kernel:HH0
"assignvariableop_11_conv2d_55_bias:H6
#assignvariableop_12_dense_24_kernel:	†H/
!assignvariableop_13_dense_24_bias:H5
#assignvariableop_14_dense_25_kernel:HH/
!assignvariableop_15_dense_25_bias:H5
#assignvariableop_16_dense_26_kernel:H/
!assignvariableop_17_dense_26_bias:'
assignvariableop_18_iteration:	 +
!assignvariableop_19_learning_rate: E
+assignvariableop_20_adam_m_conv2d_50_kernel: E
+assignvariableop_21_adam_v_conv2d_50_kernel: 7
)assignvariableop_22_adam_m_conv2d_50_bias: 7
)assignvariableop_23_adam_v_conv2d_50_bias: E
+assignvariableop_24_adam_m_conv2d_51_kernel: HE
+assignvariableop_25_adam_v_conv2d_51_kernel: H7
)assignvariableop_26_adam_m_conv2d_51_bias:H7
)assignvariableop_27_adam_v_conv2d_51_bias:HE
+assignvariableop_28_adam_m_conv2d_52_kernel:HHE
+assignvariableop_29_adam_v_conv2d_52_kernel:HH7
)assignvariableop_30_adam_m_conv2d_52_bias:H7
)assignvariableop_31_adam_v_conv2d_52_bias:HE
+assignvariableop_32_adam_m_conv2d_53_kernel:HHE
+assignvariableop_33_adam_v_conv2d_53_kernel:HH7
)assignvariableop_34_adam_m_conv2d_53_bias:H7
)assignvariableop_35_adam_v_conv2d_53_bias:HE
+assignvariableop_36_adam_m_conv2d_54_kernel:HHE
+assignvariableop_37_adam_v_conv2d_54_kernel:HH7
)assignvariableop_38_adam_m_conv2d_54_bias:H7
)assignvariableop_39_adam_v_conv2d_54_bias:HE
+assignvariableop_40_adam_m_conv2d_55_kernel:HHE
+assignvariableop_41_adam_v_conv2d_55_kernel:HH7
)assignvariableop_42_adam_m_conv2d_55_bias:H7
)assignvariableop_43_adam_v_conv2d_55_bias:H=
*assignvariableop_44_adam_m_dense_24_kernel:	†H=
*assignvariableop_45_adam_v_dense_24_kernel:	†H6
(assignvariableop_46_adam_m_dense_24_bias:H6
(assignvariableop_47_adam_v_dense_24_bias:H<
*assignvariableop_48_adam_m_dense_25_kernel:HH<
*assignvariableop_49_adam_v_dense_25_kernel:HH6
(assignvariableop_50_adam_m_dense_25_bias:H6
(assignvariableop_51_adam_v_dense_25_bias:H<
*assignvariableop_52_adam_m_dense_26_kernel:H<
*assignvariableop_53_adam_v_dense_26_kernel:H6
(assignvariableop_54_adam_m_dense_26_bias:6
(assignvariableop_55_adam_v_dense_26_bias:%
assignvariableop_56_total_1: %
assignvariableop_57_count_1: #
assignvariableop_58_total: #
assignvariableop_59_count: 8
*assignvariableop_60_random_flip_1_statevar:	<
.assignvariableop_61_random_rotation_1_statevar:	
identity_63ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Щ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*њ
valueµB≤?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHс
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*У
valueЙBЖ?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B №
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Т
_output_shapes€
ь:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?			[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_50_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_50_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_51_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_51_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_52_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_52_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_53_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_53_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_54_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_54_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_55_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_55_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_24_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_24_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_25_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_25_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_26_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_26_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_18AssignVariableOpassignvariableop_18_iterationIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_m_conv2d_50_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_v_conv2d_50_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_m_conv2d_50_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_v_conv2d_50_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_m_conv2d_51_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_v_conv2d_51_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_m_conv2d_51_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_v_conv2d_51_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_m_conv2d_52_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_v_conv2d_52_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_m_conv2d_52_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_v_conv2d_52_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_m_conv2d_53_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_v_conv2d_53_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_m_conv2d_53_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_v_conv2d_53_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_m_conv2d_54_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_v_conv2d_54_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_m_conv2d_54_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_v_conv2d_54_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_m_conv2d_55_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_v_conv2d_55_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_m_conv2d_55_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_v_conv2d_55_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_m_dense_24_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_v_dense_24_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_m_dense_24_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_v_dense_24_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_dense_25_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_dense_25_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_m_dense_25_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_v_dense_25_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_m_dense_26_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_v_dense_26_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_m_dense_26_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_v_dense_26_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_56AssignVariableOpassignvariableop_56_total_1Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_57AssignVariableOpassignvariableop_57_count_1Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_58AssignVariableOpassignvariableop_58_totalIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_59AssignVariableOpassignvariableop_59_countIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0	*
_output_shapes
:√
AssignVariableOp_60AssignVariableOp*assignvariableop_60_random_flip_1_statevarIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0	*
_output_shapes
:«
AssignVariableOp_61AssignVariableOp.assignvariableop_61_random_rotation_1_statevarIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 £
Identity_62Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_63IdentityIdentity_62:output:0^NoOp_1*
T0*
_output_shapes
: м

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_63Identity_63:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_user_specified_nameconv2d_50/kernel:.*
(
_user_specified_nameconv2d_50/bias:0,
*
_user_specified_nameconv2d_51/kernel:.*
(
_user_specified_nameconv2d_51/bias:0,
*
_user_specified_nameconv2d_52/kernel:.*
(
_user_specified_nameconv2d_52/bias:0,
*
_user_specified_nameconv2d_53/kernel:.*
(
_user_specified_nameconv2d_53/bias:0	,
*
_user_specified_nameconv2d_54/kernel:.
*
(
_user_specified_nameconv2d_54/bias:0,
*
_user_specified_nameconv2d_55/kernel:.*
(
_user_specified_nameconv2d_55/bias:/+
)
_user_specified_namedense_24/kernel:-)
'
_user_specified_namedense_24/bias:/+
)
_user_specified_namedense_25/kernel:-)
'
_user_specified_namedense_25/bias:/+
)
_user_specified_namedense_26/kernel:-)
'
_user_specified_namedense_26/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:73
1
_user_specified_nameAdam/m/conv2d_50/kernel:73
1
_user_specified_nameAdam/v/conv2d_50/kernel:51
/
_user_specified_nameAdam/m/conv2d_50/bias:51
/
_user_specified_nameAdam/v/conv2d_50/bias:73
1
_user_specified_nameAdam/m/conv2d_51/kernel:73
1
_user_specified_nameAdam/v/conv2d_51/kernel:51
/
_user_specified_nameAdam/m/conv2d_51/bias:51
/
_user_specified_nameAdam/v/conv2d_51/bias:73
1
_user_specified_nameAdam/m/conv2d_52/kernel:73
1
_user_specified_nameAdam/v/conv2d_52/kernel:51
/
_user_specified_nameAdam/m/conv2d_52/bias:5 1
/
_user_specified_nameAdam/v/conv2d_52/bias:7!3
1
_user_specified_nameAdam/m/conv2d_53/kernel:7"3
1
_user_specified_nameAdam/v/conv2d_53/kernel:5#1
/
_user_specified_nameAdam/m/conv2d_53/bias:5$1
/
_user_specified_nameAdam/v/conv2d_53/bias:7%3
1
_user_specified_nameAdam/m/conv2d_54/kernel:7&3
1
_user_specified_nameAdam/v/conv2d_54/kernel:5'1
/
_user_specified_nameAdam/m/conv2d_54/bias:5(1
/
_user_specified_nameAdam/v/conv2d_54/bias:7)3
1
_user_specified_nameAdam/m/conv2d_55/kernel:7*3
1
_user_specified_nameAdam/v/conv2d_55/kernel:5+1
/
_user_specified_nameAdam/m/conv2d_55/bias:5,1
/
_user_specified_nameAdam/v/conv2d_55/bias:6-2
0
_user_specified_nameAdam/m/dense_24/kernel:6.2
0
_user_specified_nameAdam/v/dense_24/kernel:4/0
.
_user_specified_nameAdam/m/dense_24/bias:400
.
_user_specified_nameAdam/v/dense_24/bias:612
0
_user_specified_nameAdam/m/dense_25/kernel:622
0
_user_specified_nameAdam/v/dense_25/kernel:430
.
_user_specified_nameAdam/m/dense_25/bias:440
.
_user_specified_nameAdam/v/dense_25/bias:652
0
_user_specified_nameAdam/m/dense_26/kernel:662
0
_user_specified_nameAdam/v/dense_26/kernel:470
.
_user_specified_nameAdam/m/dense_26/bias:480
.
_user_specified_nameAdam/v/dense_26/bias:'9#
!
_user_specified_name	total_1:':#
!
_user_specified_name	count_1:%;!

_user_specified_nametotal:%<!

_user_specified_namecount:6=2
0
_user_specified_namerandom_flip_1/StateVar::>6
4
_user_specified_namerandom_rotation_1/StateVar
Ђ
J
"__inference__update_step_xla_37009
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:H: *
	_noinline(:D @

_output_shapes
:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
№

э
D__inference_conv2d_51_layer_call_and_return_conditional_losses_39593

inputs8
conv2d_readvariableop_resource: H-
biasadd_readvariableop_resource:H
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: H*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€H*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Hg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€HS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€ББ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€ББ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ч
Ю
)__inference_conv2d_52_layer_call_fn_39622

inputs!
unknown:HH
	unknown_0:H
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€==H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_52_layer_call_and_return_conditional_losses_39078w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€==H<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€??H: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€??H
 
_user_specified_nameinputs:%!

_user_specified_name39616:%!

_user_specified_name39618
У
g
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_38960

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs"нL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_defaultЈ
[
sequential_6_inputE
$serving_default_sequential_6_input:0€€€€€€€€€АА<
dense_260
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ЄФ
§
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer_with_weights-5
layer-18
layer-19
layer-20
layer-21
layer_with_weights-6
layer-22
layer_with_weights-7
layer-23
layer_with_weights-8
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures"
_tf_keras_sequential
ƒ
#layer-0
$layer-1
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_sequential
ƒ
+layer-0
,layer-1
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_sequential
•
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
Ё
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias
 A_jit_compiled_convolution_op"
_tf_keras_layer
•
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
•
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
Ё
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias
 V_jit_compiled_convolution_op"
_tf_keras_layer
•
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
•
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
Ё
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias
 k_jit_compiled_convolution_op"
_tf_keras_layer
•
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
•
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias
!А_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
Уkernel
	Фbias
!Х_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
†__call__
+°&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses
®kernel
	©bias
!™_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
±	variables
≤trainable_variables
≥regularization_losses
і	keras_api
µ__call__
+ґ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ј	variables
Єtrainable_variables
єregularization_losses
Ї	keras_api
ї__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
√
љ	variables
Њtrainable_variables
њregularization_losses
ј	keras_api
Ѕ__call__
+¬&call_and_return_all_conditional_losses
√kernel
	ƒbias"
_tf_keras_layer
√
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses
Ћkernel
	ћbias"
_tf_keras_layer
√
Ќ	variables
ќtrainable_variables
ѕregularization_losses
–	keras_api
—__call__
+“&call_and_return_all_conditional_losses
”kernel
	‘bias"
_tf_keras_layer
∞
?0
@1
T2
U3
i4
j5
~6
7
У8
Ф9
®10
©11
√12
ƒ13
Ћ14
ћ15
”16
‘17"
trackable_list_wrapper
∞
?0
@1
T2
U3
i4
j5
~6
7
У8
Ф9
®10
©11
√12
ƒ13
Ћ14
ћ15
”16
‘17"
trackable_list_wrapper
 "
trackable_list_wrapper
ѕ
’non_trainable_variables
÷layers
„metrics
 Ўlayer_regularization_losses
ўlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
—
Џtrace_0
џtrace_12Ц
-__inference_sequential_12_layer_call_fn_39323
-__inference_sequential_12_layer_call_fn_39364µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЏtrace_0zџtrace_1
З
№trace_0
Ёtrace_12ћ
H__inference_sequential_12_layer_call_and_return_conditional_losses_39213
H__inference_sequential_12_layer_call_and_return_conditional_losses_39278µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z№trace_0zЁtrace_1
÷B”
 __inference__wrapped_model_38607sequential_6_input"Ш
С≤Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
£
ё
_variables
я_iterations
а_learning_rate
б_index_dict
в
_momentums
г_velocities
д_update_step_xla"
experimentalOptimizer
-
еserving_default"
signature_map
Ђ
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
р__call__
+с&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ѕ
чtrace_0
шtrace_12Ф
,__inference_sequential_6_layer_call_fn_38638
,__inference_sequential_6_layer_call_fn_38643µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zчtrace_0zшtrace_1
Е
щtrace_0
ъtrace_12 
G__inference_sequential_6_layer_call_and_return_conditional_losses_38627
G__inference_sequential_6_layer_call_and_return_conditional_losses_38633µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zщtrace_0zъtrace_1
√
ы	variables
ьtrainable_variables
эregularization_losses
ю	keras_api
€__call__
+А&call_and_return_all_conditional_losses
Б_random_generator"
_tf_keras_layer
√
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses
И_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
ѕ
Оtrace_0
Пtrace_12Ф
,__inference_sequential_7_layer_call_fn_38915
,__inference_sequential_7_layer_call_fn_38920µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zОtrace_0zПtrace_1
Е
Рtrace_0
Сtrace_12 
G__inference_sequential_7_layer_call_and_return_conditional_losses_38892
G__inference_sequential_7_layer_call_and_return_conditional_losses_38906µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zРtrace_0zСtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
м
Чtrace_02Ќ
0__inference_zero_padding2d_7_layer_call_fn_39529Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЧtrace_0
З
Шtrace_02и
K__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_39535Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zШtrace_0
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
е
Юtrace_02∆
)__inference_conv2d_50_layer_call_fn_39544Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЮtrace_0
А
Яtrace_02б
D__inference_conv2d_50_layer_call_and_return_conditional_losses_39554Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЯtrace_0
*:( 2conv2d_50/kernel
: 2conv2d_50/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
д
•trace_02≈
(__inference_re_lu_50_layer_call_fn_39559Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z•trace_0
€
¶trace_02а
C__inference_re_lu_50_layer_call_and_return_conditional_losses_39564Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z¶trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
м
ђtrace_02Ќ
0__inference_max_pooling2d_50_layer_call_fn_39569Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zђtrace_0
З
≠trace_02и
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_39574Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≠trace_0
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ѓnon_trainable_variables
ѓlayers
∞metrics
 ±layer_regularization_losses
≤layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
е
≥trace_02∆
)__inference_conv2d_51_layer_call_fn_39583Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≥trace_0
А
іtrace_02б
D__inference_conv2d_51_layer_call_and_return_conditional_losses_39593Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zіtrace_0
*:( H2conv2d_51/kernel
:H2conv2d_51/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
µnon_trainable_variables
ґlayers
Јmetrics
 Єlayer_regularization_losses
єlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
д
Їtrace_02≈
(__inference_re_lu_51_layer_call_fn_39598Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЇtrace_0
€
їtrace_02а
C__inference_re_lu_51_layer_call_and_return_conditional_losses_39603Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zїtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
м
Ѕtrace_02Ќ
0__inference_max_pooling2d_51_layer_call_fn_39608Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЅtrace_0
З
¬trace_02и
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_39613Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z¬trace_0
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
е
»trace_02∆
)__inference_conv2d_52_layer_call_fn_39622Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z»trace_0
А
…trace_02б
D__inference_conv2d_52_layer_call_and_return_conditional_losses_39632Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z…trace_0
*:(HH2conv2d_52/kernel
:H2conv2d_52/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
 non_trainable_variables
Ћlayers
ћmetrics
 Ќlayer_regularization_losses
ќlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
д
ѕtrace_02≈
(__inference_re_lu_52_layer_call_fn_39637Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zѕtrace_0
€
–trace_02а
C__inference_re_lu_52_layer_call_and_return_conditional_losses_39642Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z–trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
—non_trainable_variables
“layers
”metrics
 ‘layer_regularization_losses
’layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
м
÷trace_02Ќ
0__inference_max_pooling2d_52_layer_call_fn_39647Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z÷trace_0
З
„trace_02и
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_39652Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z„trace_0
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
е
Ёtrace_02∆
)__inference_conv2d_53_layer_call_fn_39661Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЁtrace_0
А
ёtrace_02б
D__inference_conv2d_53_layer_call_and_return_conditional_losses_39671Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zёtrace_0
*:(HH2conv2d_53/kernel
:H2conv2d_53/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
д
дtrace_02≈
(__inference_re_lu_53_layer_call_fn_39676Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zдtrace_0
€
еtrace_02а
C__inference_re_lu_53_layer_call_and_return_conditional_losses_39681Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zеtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
м
лtrace_02Ќ
0__inference_max_pooling2d_53_layer_call_fn_39686Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zлtrace_0
З
мtrace_02и
K__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_39691Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zмtrace_0
0
У0
Ф1"
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
е
тtrace_02∆
)__inference_conv2d_54_layer_call_fn_39700Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zтtrace_0
А
уtrace_02б
D__inference_conv2d_54_layer_call_and_return_conditional_losses_39710Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zуtrace_0
*:(HH2conv2d_54/kernel
:H2conv2d_54/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
д
щtrace_02≈
(__inference_re_lu_54_layer_call_fn_39715Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zщtrace_0
€
ъtrace_02а
C__inference_re_lu_54_layer_call_and_return_conditional_losses_39720Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zъtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
€layer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
†__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
м
Аtrace_02Ќ
0__inference_max_pooling2d_54_layer_call_fn_39725Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zАtrace_0
З
Бtrace_02и
K__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_39730Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zБtrace_0
0
®0
©1"
trackable_list_wrapper
0
®0
©1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
е
Зtrace_02∆
)__inference_conv2d_55_layer_call_fn_39739Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЗtrace_0
А
Иtrace_02б
D__inference_conv2d_55_layer_call_and_return_conditional_losses_39749Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zИtrace_0
*:(HH2conv2d_55/kernel
:H2conv2d_55/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
Ђ	variables
ђtrainable_variables
≠regularization_losses
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
д
Оtrace_02≈
(__inference_re_lu_55_layer_call_fn_39754Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zОtrace_0
€
Пtrace_02а
C__inference_re_lu_55_layer_call_and_return_conditional_losses_39759Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zПtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
±	variables
≤trainable_variables
≥regularization_losses
µ__call__
+ґ&call_and_return_all_conditional_losses
'ґ"call_and_return_conditional_losses"
_generic_user_object
м
Хtrace_02Ќ
0__inference_max_pooling2d_55_layer_call_fn_39764Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zХtrace_0
З
Цtrace_02и
K__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_39769Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЦtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
Ј	variables
Єtrainable_variables
єregularization_losses
ї__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
е
Ьtrace_02∆
)__inference_flatten_4_layer_call_fn_39774Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЬtrace_0
А
Эtrace_02б
D__inference_flatten_4_layer_call_and_return_conditional_losses_39780Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЭtrace_0
0
√0
ƒ1"
trackable_list_wrapper
0
√0
ƒ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Юnon_trainable_variables
Яlayers
†metrics
 °layer_regularization_losses
Ґlayer_metrics
љ	variables
Њtrainable_variables
њregularization_losses
Ѕ__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
д
£trace_02≈
(__inference_dense_24_layer_call_fn_39789Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z£trace_0
€
§trace_02а
C__inference_dense_24_layer_call_and_return_conditional_losses_39800Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z§trace_0
": 	†H2dense_24/kernel
:H2dense_24/bias
0
Ћ0
ћ1"
trackable_list_wrapper
0
Ћ0
ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
д
™trace_02≈
(__inference_dense_25_layer_call_fn_39809Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z™trace_0
€
Ђtrace_02а
C__inference_dense_25_layer_call_and_return_conditional_losses_39820Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЂtrace_0
!:HH2dense_25/kernel
:H2dense_25/bias
0
”0
‘1"
trackable_list_wrapper
0
”0
‘1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ђnon_trainable_variables
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
∞layer_metrics
Ќ	variables
ќtrainable_variables
ѕregularization_losses
—__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object
д
±trace_02≈
(__inference_dense_26_layer_call_fn_39829Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z±trace_0
€
≤trace_02а
C__inference_dense_26_layer_call_and_return_conditional_losses_39840Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≤trace_0
!:H2dense_26/kernel
:2dense_26/bias
 "
trackable_list_wrapper
ё
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
0
≥0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
-__inference_sequential_12_layer_call_fn_39323sequential_6_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
-__inference_sequential_12_layer_call_fn_39364sequential_6_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ТBП
H__inference_sequential_12_layer_call_and_return_conditional_losses_39213sequential_6_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ТBП
H__inference_sequential_12_layer_call_and_return_conditional_losses_39278sequential_6_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
г
я0
µ1
ґ2
Ј3
Є4
є5
Ї6
ї7
Љ8
љ9
Њ10
њ11
ј12
Ѕ13
¬14
√15
ƒ16
≈17
∆18
«19
»20
…21
 22
Ћ23
ћ24
Ќ25
ќ26
ѕ27
–28
—29
“30
”31
‘32
’33
÷34
„35
Ў36"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
Є
µ0
Ј1
є2
ї3
љ4
њ5
Ѕ6
√7
≈8
«9
…10
Ћ11
Ќ12
ѕ13
—14
”15
’16
„17"
trackable_list_wrapper
Є
ґ0
Є1
Ї2
Љ3
Њ4
ј5
¬6
ƒ7
∆8
»9
 10
ћ11
ќ12
–13
“14
‘15
÷16
Ў17"
trackable_list_wrapper
≈

ўtrace_0
Џtrace_1
џtrace_2
№trace_3
Ёtrace_4
ёtrace_5
яtrace_6
аtrace_7
бtrace_8
вtrace_9
гtrace_10
дtrace_11
еtrace_12
жtrace_13
зtrace_14
иtrace_15
йtrace_16
кtrace_172Ї
"__inference__update_step_xla_36974
"__inference__update_step_xla_36979
"__inference__update_step_xla_36984
"__inference__update_step_xla_36989
"__inference__update_step_xla_36994
"__inference__update_step_xla_36999
"__inference__update_step_xla_37004
"__inference__update_step_xla_37009
"__inference__update_step_xla_37014
"__inference__update_step_xla_37019
"__inference__update_step_xla_37024
"__inference__update_step_xla_37029
"__inference__update_step_xla_37034
"__inference__update_step_xla_37039
"__inference__update_step_xla_37044
"__inference__update_step_xla_37049
"__inference__update_step_xla_37054
"__inference__update_step_xla_37059ѓ
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0zўtrace_0zЏtrace_1zџtrace_2z№trace_3zЁtrace_4zёtrace_5zяtrace_6zаtrace_7zбtrace_8zвtrace_9zгtrace_10zдtrace_11zеtrace_12zжtrace_13zзtrace_14zиtrace_15zйtrace_16zкtrace_17
еBв
#__inference_signature_wrapper_39524sequential_6_input"§
Э≤Щ
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 '

kwonlyargsЪ
jsequential_6_input
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
ж
рtrace_02«
*__inference_resizing_1_layer_call_fn_39845Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zрtrace_0
Б
сtrace_02в
E__inference_resizing_1_layer_call_and_return_conditional_losses_39851Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zсtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
м	variables
нtrainable_variables
оregularization_losses
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
з
чtrace_02»
+__inference_rescaling_1_layer_call_fn_39856Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zчtrace_0
В
шtrace_02г
F__inference_rescaling_1_layer_call_and_return_conditional_losses_39864Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zшtrace_0
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
,__inference_sequential_6_layer_call_fn_38638resizing_1_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
фBс
,__inference_sequential_6_layer_call_fn_38643resizing_1_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
G__inference_sequential_6_layer_call_and_return_conditional_losses_38627resizing_1_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
G__inference_sequential_6_layer_call_and_return_conditional_losses_38633resizing_1_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
ы	variables
ьtrainable_variables
эregularization_losses
€__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
≈
юtrace_0
€trace_12К
-__inference_random_flip_1_layer_call_fn_39871
-__inference_random_flip_1_layer_call_fn_39876©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zюtrace_0z€trace_1
ы
Аtrace_0
Бtrace_12ј
H__inference_random_flip_1_layer_call_and_return_conditional_losses_39987
H__inference_random_flip_1_layer_call_and_return_conditional_losses_39991©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zАtrace_0zБtrace_1
/
В
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
Ќ
Иtrace_0
Йtrace_12Т
1__inference_random_rotation_1_layer_call_fn_39998
1__inference_random_rotation_1_layer_call_fn_40003©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zИtrace_0zЙtrace_1
Г
Кtrace_0
Лtrace_12»
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40121
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40125©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zКtrace_0zЛtrace_1
/
М
_generator"
_generic_user_object
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
,__inference_sequential_7_layer_call_fn_38915random_flip_1_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
,__inference_sequential_7_layer_call_fn_38920random_flip_1_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ТBП
G__inference_sequential_7_layer_call_and_return_conditional_losses_38892random_flip_1_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ТBП
G__inference_sequential_7_layer_call_and_return_conditional_losses_38906random_flip_1_input"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_zero_padding2d_7_layer_call_fn_39529inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_39535inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_conv2d_50_layer_call_fn_39544inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv2d_50_layer_call_and_return_conditional_losses_39554inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_re_lu_50_layer_call_fn_39559inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_re_lu_50_layer_call_and_return_conditional_losses_39564inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_max_pooling2d_50_layer_call_fn_39569inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_39574inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_conv2d_51_layer_call_fn_39583inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv2d_51_layer_call_and_return_conditional_losses_39593inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_re_lu_51_layer_call_fn_39598inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_re_lu_51_layer_call_and_return_conditional_losses_39603inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_max_pooling2d_51_layer_call_fn_39608inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_39613inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_conv2d_52_layer_call_fn_39622inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv2d_52_layer_call_and_return_conditional_losses_39632inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_re_lu_52_layer_call_fn_39637inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_re_lu_52_layer_call_and_return_conditional_losses_39642inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_max_pooling2d_52_layer_call_fn_39647inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_39652inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_conv2d_53_layer_call_fn_39661inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv2d_53_layer_call_and_return_conditional_losses_39671inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_re_lu_53_layer_call_fn_39676inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_re_lu_53_layer_call_and_return_conditional_losses_39681inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_max_pooling2d_53_layer_call_fn_39686inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_39691inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_conv2d_54_layer_call_fn_39700inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv2d_54_layer_call_and_return_conditional_losses_39710inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_re_lu_54_layer_call_fn_39715inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_re_lu_54_layer_call_and_return_conditional_losses_39720inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_max_pooling2d_54_layer_call_fn_39725inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_39730inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_conv2d_55_layer_call_fn_39739inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv2d_55_layer_call_and_return_conditional_losses_39749inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_re_lu_55_layer_call_fn_39754inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_re_lu_55_layer_call_and_return_conditional_losses_39759inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_max_pooling2d_55_layer_call_fn_39764inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_39769inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_flatten_4_layer_call_fn_39774inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_flatten_4_layer_call_and_return_conditional_losses_39780inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_dense_24_layer_call_fn_39789inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_dense_24_layer_call_and_return_conditional_losses_39800inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_dense_25_layer_call_fn_39809inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_dense_25_layer_call_and_return_conditional_losses_39820inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_dense_26_layer_call_fn_39829inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_dense_26_layer_call_and_return_conditional_losses_39840inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R
Н	variables
О	keras_api

Пtotal

Рcount"
_tf_keras_metric
c
С	variables
Т	keras_api

Уtotal

Фcount
Х
_fn_kwargs"
_tf_keras_metric
/:- 2Adam/m/conv2d_50/kernel
/:- 2Adam/v/conv2d_50/kernel
!: 2Adam/m/conv2d_50/bias
!: 2Adam/v/conv2d_50/bias
/:- H2Adam/m/conv2d_51/kernel
/:- H2Adam/v/conv2d_51/kernel
!:H2Adam/m/conv2d_51/bias
!:H2Adam/v/conv2d_51/bias
/:-HH2Adam/m/conv2d_52/kernel
/:-HH2Adam/v/conv2d_52/kernel
!:H2Adam/m/conv2d_52/bias
!:H2Adam/v/conv2d_52/bias
/:-HH2Adam/m/conv2d_53/kernel
/:-HH2Adam/v/conv2d_53/kernel
!:H2Adam/m/conv2d_53/bias
!:H2Adam/v/conv2d_53/bias
/:-HH2Adam/m/conv2d_54/kernel
/:-HH2Adam/v/conv2d_54/kernel
!:H2Adam/m/conv2d_54/bias
!:H2Adam/v/conv2d_54/bias
/:-HH2Adam/m/conv2d_55/kernel
/:-HH2Adam/v/conv2d_55/kernel
!:H2Adam/m/conv2d_55/bias
!:H2Adam/v/conv2d_55/bias
':%	†H2Adam/m/dense_24/kernel
':%	†H2Adam/v/dense_24/kernel
 :H2Adam/m/dense_24/bias
 :H2Adam/v/dense_24/bias
&:$HH2Adam/m/dense_25/kernel
&:$HH2Adam/v/dense_25/kernel
 :H2Adam/m/dense_25/bias
 :H2Adam/v/dense_25/bias
&:$H2Adam/m/dense_26/kernel
&:$H2Adam/v/dense_26/kernel
 :2Adam/m/dense_26/bias
 :2Adam/v/dense_26/bias
нBк
"__inference__update_step_xla_36974gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_36979gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_36984gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_36989gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_36994gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_36999gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37004gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37009gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37014gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37019gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37024gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37029gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37034gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37039gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37044gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37049gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37054gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_37059gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
‘B—
*__inference_resizing_1_layer_call_fn_39845inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
E__inference_resizing_1_layer_call_and_return_conditional_losses_39851inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
’B“
+__inference_rescaling_1_layer_call_fn_39856inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
рBн
F__inference_rescaling_1_layer_call_and_return_conditional_losses_39864inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
-__inference_random_flip_1_layer_call_fn_39871inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
гBа
-__inference_random_flip_1_layer_call_fn_39876inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
H__inference_random_flip_1_layer_call_and_return_conditional_losses_39987inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
H__inference_random_flip_1_layer_call_and_return_conditional_losses_39991inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
/
Ц
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
1__inference_random_rotation_1_layer_call_fn_39998inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
зBд
1__inference_random_rotation_1_layer_call_fn_40003inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40121inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40125inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
/
Ч
_state_var"
_generic_user_object
0
П0
Р1"
trackable_list_wrapper
.
Н	variables"
_generic_user_object
:  (2total
:  (2count
0
У0
Ф1"
trackable_list_wrapper
.
С	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
": 	2random_flip_1/StateVar
&:$	2random_rotation_1/StateVar§
"__inference__update_step_xla_36974~xҐu
nҐk
!К
gradient 
<Т9	%Ґ"
ъ 
А
p
` VariableSpec 
`†ƒВ¬ЃЋ=
™ "
 М
"__inference__update_step_xla_36979f`Ґ]
VҐS
К
gradient 
0Т-	Ґ
ъ 
А
p
` VariableSpec 
`†ЇВ¬ЃЋ=
™ "
 §
"__inference__update_step_xla_36984~xҐu
nҐk
!К
gradient H
<Т9	%Ґ"
ъ H
А
p
` VariableSpec 
`ј¬Г¬ЃЋ=
™ "
 М
"__inference__update_step_xla_36989f`Ґ]
VҐS
К
gradientH
0Т-	Ґ
ъH
А
p
` VariableSpec 
`†≠Г¬ЃЋ=
™ "
 §
"__inference__update_step_xla_36994~xҐu
nҐk
!К
gradientHH
<Т9	%Ґ"
ъHH
А
p
` VariableSpec 
`А√ЮЅЃЋ=
™ "
 М
"__inference__update_step_xla_36999f`Ґ]
VҐS
К
gradientH
0Т-	Ґ
ъH
А
p
` VariableSpec 
`јђЮЅЃЋ=
™ "
 §
"__inference__update_step_xla_37004~xҐu
nҐk
!К
gradientHH
<Т9	%Ґ"
ъHH
А
p
` VariableSpec 
`аоЮЅЃЋ=
™ "
 М
"__inference__update_step_xla_37009f`Ґ]
VҐS
К
gradientH
0Т-	Ґ
ъH
А
p
` VariableSpec 
`АрЮЅЃЋ=
™ "
 §
"__inference__update_step_xla_37014~xҐu
nҐk
!К
gradientHH
<Т9	%Ґ"
ъHH
А
p
` VariableSpec 
`а»ЯЅЃЋ=
™ "
 М
"__inference__update_step_xla_37019f`Ґ]
VҐS
К
gradientH
0Т-	Ґ
ъH
А
p
` VariableSpec 
`аяЮЅЃЋ=
™ "
 §
"__inference__update_step_xla_37024~xҐu
nҐk
!К
gradientHH
<Т9	%Ґ"
ъHH
А
p
` VariableSpec 
`АлКє£Ћ=
™ "
 М
"__inference__update_step_xla_37029f`Ґ]
VҐS
К
gradientH
0Т-	Ґ
ъH
А
p
` VariableSpec 
`†шЯЅЃЋ=
™ "
 Ц
"__inference__update_step_xla_37034pjҐg
`Ґ]
К
gradient	†H
5Т2	Ґ
ъ	†H
А
p
` VariableSpec 
`аПцЅЃЋ=
™ "
 М
"__inference__update_step_xla_37039f`Ґ]
VҐS
К
gradientH
0Т-	Ґ
ъH
А
p
` VariableSpec 
`а√ЩЖ†Ћ=
™ "
 Ф
"__inference__update_step_xla_37044nhҐe
^Ґ[
К
gradientHH
4Т1	Ґ
ъHH
А
p
` VariableSpec 
`А‘Лє£Ћ=
™ "
 М
"__inference__update_step_xla_37049f`Ґ]
VҐS
К
gradientH
0Т-	Ґ
ъH
А
p
` VariableSpec 
`јЯЛє£Ћ=
™ "
 Ф
"__inference__update_step_xla_37054nhҐe
^Ґ[
К
gradientH
4Т1	Ґ
ъH
А
p
` VariableSpec 
`†сЮЅЃЋ=
™ "
 М
"__inference__update_step_xla_37059f`Ґ]
VҐS
К
gradient
0Т-	Ґ
ъ
А
p
` VariableSpec 
`аКЪД†Ћ=
™ "
 њ
 __inference__wrapped_model_38607Ъ?@TUij~УФ®©√ƒЋћ”‘EҐB
;Ґ8
6К3
sequential_6_input€€€€€€€€€АА
™ "3™0
.
dense_26"К
dense_26€€€€€€€€€њ
D__inference_conv2d_50_layer_call_and_return_conditional_losses_39554w?@9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ДД
™ "6Ґ3
,К)
tensor_0€€€€€€€€€ВВ 
Ъ Щ
)__inference_conv2d_50_layer_call_fn_39544l?@9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ДД
™ "+К(
unknown€€€€€€€€€ВВ љ
D__inference_conv2d_51_layer_call_and_return_conditional_losses_39593uTU9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ББ 
™ "4Ґ1
*К'
tensor_0€€€€€€€€€H
Ъ Ч
)__inference_conv2d_51_layer_call_fn_39583jTU9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ББ 
™ ")К&
unknown€€€€€€€€€Hї
D__inference_conv2d_52_layer_call_and_return_conditional_losses_39632sij7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€??H
™ "4Ґ1
*К'
tensor_0€€€€€€€€€==H
Ъ Х
)__inference_conv2d_52_layer_call_fn_39622hij7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€??H
™ ")К&
unknown€€€€€€€€€==Hї
D__inference_conv2d_53_layer_call_and_return_conditional_losses_39671s~7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ "4Ґ1
*К'
tensor_0€€€€€€€€€H
Ъ Х
)__inference_conv2d_53_layer_call_fn_39661h~7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ ")К&
unknown€€€€€€€€€Hљ
D__inference_conv2d_54_layer_call_and_return_conditional_losses_39710uУФ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ "4Ґ1
*К'
tensor_0€€€€€€€€€H
Ъ Ч
)__inference_conv2d_54_layer_call_fn_39700jУФ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ ")К&
unknown€€€€€€€€€Hљ
D__inference_conv2d_55_layer_call_and_return_conditional_losses_39749u®©7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ "4Ґ1
*К'
tensor_0€€€€€€€€€H
Ъ Ч
)__inference_conv2d_55_layer_call_fn_39739j®©7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ ")К&
unknown€€€€€€€€€H≠
C__inference_dense_24_layer_call_and_return_conditional_losses_39800f√ƒ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€†
™ ",Ґ)
"К
tensor_0€€€€€€€€€H
Ъ З
(__inference_dense_24_layer_call_fn_39789[√ƒ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€†
™ "!К
unknown€€€€€€€€€Hђ
C__inference_dense_25_layer_call_and_return_conditional_losses_39820eЋћ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€H
™ ",Ґ)
"К
tensor_0€€€€€€€€€H
Ъ Ж
(__inference_dense_25_layer_call_fn_39809ZЋћ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€H
™ "!К
unknown€€€€€€€€€Hђ
C__inference_dense_26_layer_call_and_return_conditional_losses_39840e”‘/Ґ,
%Ґ"
 К
inputs€€€€€€€€€H
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Ж
(__inference_dense_26_layer_call_fn_39829Z”‘/Ґ,
%Ґ"
 К
inputs€€€€€€€€€H
™ "!К
unknown€€€€€€€€€∞
D__inference_flatten_4_layer_call_and_return_conditional_losses_39780h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ "-Ґ*
#К 
tensor_0€€€€€€€€€†
Ъ К
)__inference_flatten_4_layer_call_fn_39774]7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ ""К
unknown€€€€€€€€€†х
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_39574•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѕ
0__inference_max_pooling2d_50_layer_call_fn_39569ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€х
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_39613•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѕ
0__inference_max_pooling2d_51_layer_call_fn_39608ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€х
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_39652•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѕ
0__inference_max_pooling2d_52_layer_call_fn_39647ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€х
K__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_39691•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѕ
0__inference_max_pooling2d_53_layer_call_fn_39686ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€х
K__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_39730•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѕ
0__inference_max_pooling2d_54_layer_call_fn_39725ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€х
K__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_39769•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѕ
0__inference_max_pooling2d_55_layer_call_fn_39764ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€«
H__inference_random_flip_1_layer_call_and_return_conditional_losses_39987{Ц=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ √
H__inference_random_flip_1_layer_call_and_return_conditional_losses_39991w=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p 
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ °
-__inference_random_flip_1_layer_call_fn_39871pЦ=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p
™ "+К(
unknown€€€€€€€€€ААЭ
-__inference_random_flip_1_layer_call_fn_39876l=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p 
™ "+К(
unknown€€€€€€€€€ААЋ
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40121{Ч=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ «
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40125w=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p 
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ •
1__inference_random_rotation_1_layer_call_fn_39998pЧ=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p
™ "+К(
unknown€€€€€€€€€АА°
1__inference_random_rotation_1_layer_call_fn_40003l=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p 
™ "+К(
unknown€€€€€€€€€ААЇ
C__inference_re_lu_50_layer_call_and_return_conditional_losses_39564s9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ВВ 
™ "6Ґ3
,К)
tensor_0€€€€€€€€€ВВ 
Ъ Ф
(__inference_re_lu_50_layer_call_fn_39559h9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ВВ 
™ "+К(
unknown€€€€€€€€€ВВ ґ
C__inference_re_lu_51_layer_call_and_return_conditional_losses_39603o7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ "4Ґ1
*К'
tensor_0€€€€€€€€€H
Ъ Р
(__inference_re_lu_51_layer_call_fn_39598d7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ ")К&
unknown€€€€€€€€€Hґ
C__inference_re_lu_52_layer_call_and_return_conditional_losses_39642o7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€==H
™ "4Ґ1
*К'
tensor_0€€€€€€€€€==H
Ъ Р
(__inference_re_lu_52_layer_call_fn_39637d7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€==H
™ ")К&
unknown€€€€€€€€€==Hґ
C__inference_re_lu_53_layer_call_and_return_conditional_losses_39681o7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ "4Ґ1
*К'
tensor_0€€€€€€€€€H
Ъ Р
(__inference_re_lu_53_layer_call_fn_39676d7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ ")К&
unknown€€€€€€€€€Hґ
C__inference_re_lu_54_layer_call_and_return_conditional_losses_39720o7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ "4Ґ1
*К'
tensor_0€€€€€€€€€H
Ъ Р
(__inference_re_lu_54_layer_call_fn_39715d7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ ")К&
unknown€€€€€€€€€Hґ
C__inference_re_lu_55_layer_call_and_return_conditional_losses_39759o7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ "4Ґ1
*К'
tensor_0€€€€€€€€€H
Ъ Р
(__inference_re_lu_55_layer_call_fn_39754d7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€H
™ ")К&
unknown€€€€€€€€€Hљ
F__inference_rescaling_1_layer_call_and_return_conditional_losses_39864s9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ Ч
+__inference_rescaling_1_layer_call_fn_39856h9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "+К(
unknown€€€€€€€€€ААЉ
E__inference_resizing_1_layer_call_and_return_conditional_losses_39851s9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ Ц
*__inference_resizing_1_layer_call_fn_39845h9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "+К(
unknown€€€€€€€€€ААм
H__inference_sequential_12_layer_call_and_return_conditional_losses_39213Я ЦЧ?@TUij~УФ®©√ƒЋћ”‘MҐJ
CҐ@
6К3
sequential_6_input€€€€€€€€€АА
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ и
H__inference_sequential_12_layer_call_and_return_conditional_losses_39278Ы?@TUij~УФ®©√ƒЋћ”‘MҐJ
CҐ@
6К3
sequential_6_input€€€€€€€€€АА
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ∆
-__inference_sequential_12_layer_call_fn_39323Ф ЦЧ?@TUij~УФ®©√ƒЋћ”‘MҐJ
CҐ@
6К3
sequential_6_input€€€€€€€€€АА
p

 
™ "!К
unknown€€€€€€€€€¬
-__inference_sequential_12_layer_call_fn_39364Р?@TUij~УФ®©√ƒЋћ”‘MҐJ
CҐ@
6К3
sequential_6_input€€€€€€€€€АА
p 

 
™ "!К
unknown€€€€€€€€€—
G__inference_sequential_6_layer_call_and_return_conditional_losses_38627ЕKҐH
AҐ>
4К1
resizing_1_input€€€€€€€€€АА
p

 
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ —
G__inference_sequential_6_layer_call_and_return_conditional_losses_38633ЕKҐH
AҐ>
4К1
resizing_1_input€€€€€€€€€АА
p 

 
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ ™
,__inference_sequential_6_layer_call_fn_38638zKҐH
AҐ>
4К1
resizing_1_input€€€€€€€€€АА
p

 
™ "+К(
unknown€€€€€€€€€АА™
,__inference_sequential_6_layer_call_fn_38643zKҐH
AҐ>
4К1
resizing_1_input€€€€€€€€€АА
p 

 
™ "+К(
unknown€€€€€€€€€ААЏ
G__inference_sequential_7_layer_call_and_return_conditional_losses_38892ОЦЧNҐK
DҐA
7К4
random_flip_1_input€€€€€€€€€АА
p

 
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ ‘
G__inference_sequential_7_layer_call_and_return_conditional_losses_38906ИNҐK
DҐA
7К4
random_flip_1_input€€€€€€€€€АА
p 

 
™ "6Ґ3
,К)
tensor_0€€€€€€€€€АА
Ъ і
,__inference_sequential_7_layer_call_fn_38915ГЦЧNҐK
DҐA
7К4
random_flip_1_input€€€€€€€€€АА
p

 
™ "+К(
unknown€€€€€€€€€АА≠
,__inference_sequential_7_layer_call_fn_38920}NҐK
DҐA
7К4
random_flip_1_input€€€€€€€€€АА
p 

 
™ "+К(
unknown€€€€€€€€€ААЎ
#__inference_signature_wrapper_39524∞?@TUij~УФ®©√ƒЋћ”‘[ҐX
Ґ 
Q™N
L
sequential_6_input6К3
sequential_6_input€€€€€€€€€АА"3™0
.
dense_26"К
dense_26€€€€€€€€€х
K__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_39535•RҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "OҐL
EКB
tensor_04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѕ
0__inference_zero_padding2d_7_layer_call_fn_39529ЪRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "DКA
unknown4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€