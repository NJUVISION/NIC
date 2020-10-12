# Python 接口安装 (python2 可能会遇到问题)

需要文件：*AE.cpp*, *My_Range_Coder.h*, *My_Range_Encoder.cpp*, *My_Range_Decoder.cpp*.

**Step. 1**：  

将AE.cpp中的Python.h路径更改为本地主机上的路径。(如/usr/include/python3.6m/Python.h等）

**Step. 2**：

```
python setup.py build
python setup.py install
```
