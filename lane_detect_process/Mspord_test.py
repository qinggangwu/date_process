import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="CPU")#这里不同，改成GPU了，表示使用GPU。如果是用CPU，这里改成“CPU”即可
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))


import mindspore
mindspore.run_check()