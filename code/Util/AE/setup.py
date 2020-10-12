from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext


class BuildExt(build_ext, object):
    def build_extensions(self):
        self.compiler.compiler_so.remove("-Wstrict-prototypes")
        self.compiler.compiler_so.append("-std=c++11")
        self.compiler.compiler_so.append("-O2")
        # self.compiler.compiler_so.append("-march=native")
        super(BuildExt, self).build_extensions()

MOD = "AE"
setup(
    name=MOD,
    ext_modules=[
        Extension(
            MOD,
            sources=["./AE.cpp", "./My_Range_Encoder.cpp", "./My_Range_Decoder.cpp"],
        )
    ],
    command_options={"CFLAG": "-std=c++11 -O2"},
    cmdclass={"build_ext": BuildExt},
)

