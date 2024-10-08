import os
import sys
from os import listdir
from os.path import isfile, join
import makeshader
metalSourcePath=sys.argv[1]
renderPath = os.path.join(metalSourcePath, "render")
cppPath= os.path.join(metalSourcePath, "MetalOPRegister.mm")
cppRenderPath = os.path.join(renderPath, 'MetalRenderOpRegister.mm')
def genRegister():
    shaders=[]
    for file in os.listdir(metalSourcePath):
        if file.endswith('.mm'):
            shaders.append(os.path.join(metalSourcePath,file))
    with open(cppPath,"w") as f:
        f.write("// This file is generated by Shell for ops register\n")
        f.write("#import \"backend/metal/MetalDefine.h\"\n")
        f.write("   namespace MNN {\n")
        f.write("#if MNN_METAL_ENABLED\n")
        funcs=[]
        transformerFuncs = []
        for shapath in shaders:
            with open(shapath,"r") as sha:
                lines=sha.readlines()
                for l in lines:
                    if l.startswith("REGISTER_METAL_OP_CREATOR("):
                        x=l.replace("REGISTER_METAL_OP_CREATOR(","").replace(")","").replace(" ","").replace(";","").replace("\n","").split(",")
                        funcname="___"+x[0]+"__"+x[1]+"__();"
                        funcs.append(funcname)
                        f.write("  extern void "+funcname+"\n")
                    elif l.startswith('REGISTER_METAL_OP_TRANSFORMER_CREATOR('):
                        x=l.replace("REGISTER_METAL_OP_TRANSFORMER_CREATOR(","").replace(")","").replace(" ","").replace(";","").replace("\n","").split(",")
                        funcname="___"+x[0]+"__"+x[1]+"__();"
                        transformerFuncs.append(funcname)
                        f.write("#ifdef MNN_SUPPORT_TRANSFORMER_FUSE\n")
                        f.write("  extern void "+funcname+"\n")
                        f.write('#endif\n')

            pass
        f.write("void registerMetalOps() {\n")
        for func in funcs:
            f.write("   "+func+"\n")
        f.write('#ifdef MNN_SUPPORT_TRANSFORMER_FUSE\n')
        for func in transformerFuncs:
            f.write("   "+func+"\n")
        f.write('#endif\n')
        f.write("}\n#endif\n}")
    if os.path.isdir(renderPath):
        shaders=[]
        for file in os.listdir(renderPath):
            if file.endswith('.mm'):
                shaders.append(os.path.join(renderPath,file))
        with open(cppRenderPath,"w") as f:
            f.write("// This file is generated by Shell for ops register\n")
            f.write("#import \"backend/metal/MetalDefine.h\"\n")
            f.write("   namespace MNN {\n")
            f.write("#if MNN_METAL_ENABLED\n")
            funcs=[]
            for shapath in shaders:
                with open(shapath,"r") as sha:
                    lines=sha.readlines()
                    for l in lines:
                        if l.startswith("REGISTER_METAL_OP_CREATOR("):
                            x=l.replace("REGISTER_METAL_OP_CREATOR(","").replace(")","").replace(" ","").replace(";","").replace("\n","").split(",")
                            funcname="___"+x[0]+"__"+x[1]+"__();"
                            funcs.append(funcname)
                            f.write("  extern void "+funcname+"\n")
                pass
            f.write("void registerMetalRenderOps() {\n")
            for func in funcs:
                f.write("   "+func+"\n")
            f.write("}\n#endif\n}")

def genSchema():
    FLATC = metalSourcePath + "/../../../3rd_party/flatbuffers/tmp/flatc"
    sourceFile = metalSourcePath + "/schema/MetalCache.fbs"
    destFile = metalSourcePath + "/"
    cmd = FLATC + " -c " + sourceFile +" --gen-object-api" +" --reflect-names"
    print(cmd)
    print(os.popen(cmd).read())
    return

def genShader():
    if os.path.isdir(renderPath):
        print("Has Render")
        shaders = makeshader.findAllShader("render/shader")
        makeshader.generateFile(os.path.join(renderPath, "AllRenderShader.hpp"), os.path.join(renderPath, "AllRenderShader.cpp"), shaders)
    shaders = makeshader.findAllShader("shader")
    makeshader.generateFile(os.path.join(metalSourcePath, "AllShader.hpp"), os.path.join(metalSourcePath, "AllShader.cpp"), shaders)

if __name__ == '__main__':
    genRegister()
    genSchema()
    genShader()
