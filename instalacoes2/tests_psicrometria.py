from psicrometria import PsicrometryPoint
import numpy as np

P = 101325
TBS = 298.15
TBU= 291.033486810343788
UR = 0.499999993914168
h = 50423.450078302405018
w = 0.009925739173425
Pw = 1591.663388405871956
TPO = 287.016886461322144
V = 0.857788243260531

dataknown = {}
dataknown["P"] = P
dataknown["TBS"] = TBS
dataknown["TBU"] = TBU
dataknown["UR"] = UR
dataknown["h"] = h
dataknown["w"] = w
dataknown["Pw"] = Pw
dataknown["TPO"] = TPO
dataknown["V"] = V

datas = [PsicrometryPoint(TBS=TBS, TBU=TBU),
         PsicrometryPoint(TBS=TBS, UR=UR),
         PsicrometryPoint(TBS=TBS, h=h),
         PsicrometryPoint(TBS=TBS, w=w),
         PsicrometryPoint(TBS=TBS, Pw=Pw),
         PsicrometryPoint(TBS=TBS, TPO=TPO),
         PsicrometryPoint(TBS=TBS, V=V),
         PsicrometryPoint(TBU=TBU, UR=UR),
        #  PsicrometryPoint(TBU=TBU, h=h),  # Apenas com essas duas informacoes nao da pra achar
         PsicrometryPoint(TBU=TBU, w=w),
         PsicrometryPoint(TBU=TBU, Pw=Pw),  # Software nao acha solucao diretamente, necessaria conversao
         PsicrometryPoint(TBU=TBU, TPO=TPO),
        #  PsicrometryPoint(TBU=TBU, V=V),
         PsicrometryPoint(UR=UR, h=h),
         PsicrometryPoint(UR=UR, w=w),  # Software nao acha solucao diretamente, necessaria conversao
         PsicrometryPoint(UR=UR, Pw=Pw),
         PsicrometryPoint(UR=UR, TPO=TPO),
         PsicrometryPoint(UR=UR, V=V),  # Software nao acha solucao diretamente, necessaria conversao
         PsicrometryPoint(h=h, w=w),  # Software nao acha solucao diretamente, necessaria conversao
         PsicrometryPoint(h=h, Pw=Pw),  # Software nao acha solucao diretamente, necessaria conversao
         PsicrometryPoint(h=h, TPO=TPO),
        #  PsicrometryPoint(h=h, V=V),
        #  PsicrometryPoint(w=w, Pw=Pw),
        #  PsicrometryPoint(w=w, TPO=TPO),
         PsicrometryPoint(w=w, V=V),
        #  PsicrometryPoint(Pw=Pw, TPO=TPO),
        #  PsicrometryPoint(Pw=Pw, V=V),
         ]
        #  getPsicroData(w=w, Pw=Pw)  # Apenas com essas duas informacoes nao da pra achar
        #  getPsicroData(w=w, TO=TO)  # Apenas com essas duas informacoes nao da pra achar
        #  getPsicroData(Pw=Pw, TO=TO)  # Apenas com essas duas informacoes nao da pra achar


def relativeerror(a, b):
    threshold = 1e-6
    divisor = np.max([threshold, np.abs(a), np.abs(b)])
    return np.abs(a - b) / divisor


print("Dataknown = ")
print(dataknown)
print("Verificando os erros entre a referencia (dataknown) e pego por cada modo:")
keys = ["P", "TBS", "TBU", "UR", "h", "w", "Pw", "TPO"]
# keys = ["P", "TBS", "TBU"]
errorelativomaximo = 0
for i, point in enumerate(datas):
    for key in keys:
        item = point.get(key)
        try:
            errorela = relativeerror(dataknown[key], item)
            if errorela > errorelativomaximo:
                errorelativomaximo = errorela
            assert errorela < 1e-3
        except Exception as e:
            rerro = relativeerror(dataknown[key], item)
            print("Deu erro [%d] com [%s]: %.3f != %.3f. Relative error = %.3e" % (i, str(key), dataknown[key], item, rerro))


print("erro maximo = ")
print(errorelativomaximo)