import os
try:
    import CoolProp
except ModuleNotFoundError:
    os.system("pip install CoolProp")
    import CoolProp

import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle
cycol = cycle('bgrcmk')

HAPropsSI = CoolProp.HumidAirProp.HAPropsSI
PropsSI = CoolProp.CoolProp.PropsSI
Celsius0 = 273.15
RaRw = 18.01534 / 28.9645
mappingsiglas = {"P":"P",
                 "TBS": "T",
                 "TBU": "B",
                 "h": "H",
                 "UR": "R",
                 "w": "W",
                 "Pw": "P_w",
                 "TPO": "D",
                 "V": "V"}
import numpy as np

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

class PsicrometryPoint:
    keys = mappingsiglas.keys()
    @staticmethod
    def validinput(**kwargs):
        chaves = kwargs.keys()
        if "P" in chaves and len(chaves) != 3:
            raise Exception("Voce deve passar 2 valores(além da pressao P opcional)")
        elif "P" not in chaves and len(chaves) != 2:
            raise Exception("Voce deve passar 2 valores(além da pressao P opcional)")
        
        if "h" in chaves and "TBU" in chaves:
            raise Exception("Impossivel de encontrar valores apenas com (h, TBU)")
        if "w" in chaves and "Pw" in chaves:
            raise Exception("Impossivel de encontrar valores apenas com (w, Pw)")
        if "w" in chaves and "TPO" in chaves:
            raise Exception("Impossivel de encontrar valores apenas com (w, TPO)")
        if "Pw" in chaves and "TPO" in chaves:
            raise Exception("Impossivel de encontrar valores apenas com (Pw, TPO)")

        for chave in chaves:
            if chave not in PsicrometryPoint.keys:
                raise Exception("Recebido argumento '%s', os validos sao %s" % (chave, PsicrometryPoint.keys))

    def __init__(self, **kwargs):
        PsicrometryPoint.validinput(**kwargs)
        self.dataSI = {}
        self.basedata = ["P"]
        self.dataSI["P"] = 101325  # Pa
        for key, item in kwargs.items():
            if key not in self.basedata:
                self.basedata.append(key)
                self.dataSI[key] = item
                
    def __find_w(self, chave):
        P = self.dataSI["P"]
        valor = self.dataSI[chave]
        if chave == "Pw":
            w = RaRw * valor/(P-valor)
        elif chave == "TPO":
            w = HAPropsSI('W','P', P, 'R',1,'D',TPO)
        else:
            raise Exception("Not expected (chave, valor) = (%s, %.3f)" % (chave, valor))
        return w

    def __find_Pw(self, chave):
        P = self.dataSI["P"]
        valor = self.dataSI[chave]
        if chave == "w":
            Pw = P * valor / (valor+RaRw)
        elif chave == "TPO":
            Pw = HAPropsSI('P_w','P', P, 'R',1,'T',valor)
        else:
            raise Exception("Not expected (chave, valor) = (%s, %.3f)" % (chave, valor))
        return Pw

    def __find_TPO(self, chave):
        P = self.dataSI["P"]
        valor = self.dataSI[chave]
        if chave == "Pw":
            TPO = HAPropsSI('T','P', P, 'R',1,'P_w', valor)
        elif chave == "w":
            TPO = HAPropsSI('T','P', P, 'R',1,'W', valor)
        else:
            raise Exception("Not expected (chave, valor) = (%s, %.3f)" % (chave, valor))
        return TPO

    def __find_TBU(self, chave):
        P = self.dataSI["P"]
        valor = self.dataSI[chave]
        if chave == "h":
            TBU = HAPropsSI('B','P', P, 'R',1,'H',valor)
        else:
            raise Exception("Not expected (chave, valor) = (%s, %.3f)" % (chave, valor))
        return TBU
    
    def __find_h(self, chave):
        P = self.dataSI["P"]
        valor = self.dataSI[chave]
        if chave == "TBU":
            TBU = HAPropsSI('B','P', P, 'R',1,'TBS', valor)
        else:
            raise Exception("Not expected (chave, valor) = (%s, %.3f)" % (chave, valor))
        return TBU
        
    def __compute_dado(self, dado):
        # print("compute_dado: " + str(dado))
        if dado in self.dataSI:
            return
        P = self.dataSI["P"]
        
        chave1 = self.basedata[1]
        chave2 = self.basedata[2]
        tipo1 = mappingsiglas[chave1]
        tipo2 = mappingsiglas[chave2]
        val1 = self.dataSI[chave1]
        val2 = self.dataSI[chave2]
        tipodado = mappingsiglas[dado]
        
        try:
            result = HAPropsSI(tipodado,'P', P, tipo1, val1, tipo2, val2)
            self.dataSI[dado] = result
        except ValueError as e:
            print("Warning: for (%s, %s), conversion needed" % (chave1, chave2))
            print("Cause: ", e)
            # random = 1
            # self.dataSI[dado] = random
            self.treat_excessoes(chave1, chave2, dado)

    def treat_excessoes(self, chave1, chave2, dado):
        P = self.dataSI["P"]
        setchaves = set([chave1, chave2])
        oldchave, newchave = None, None
        if setchaves== set(["TBS", "h"]):
            TBU = self.__find_TBU("h")
            self.dataSI["TBU"] = TBU
            oldchave = "h"
            newchave = "TBU"
        if setchaves== set(["UR", "w"]):
            Pw = self.__find_Pw("w")
            self.dataSI["Pw"] = Pw
            oldchave = "w"
            newchave = "Pw"
        if setchaves == set(["UR", "TPO"]):
            Pw = self.__find_Pw("TPO")
            self.dataSI["Pw"] = Pw
            oldchave = "TPO"
            newchave = "Pw"
        if setchaves == set(["TBU", "Pw"]):
            w = self.__find_w("Pw")
            self.dataSI["w"] = w
            oldchave = "Pw"
            newchave = "w"
        if setchaves == set(["h", "w"]):
            TBU = self.__find_TBU("h")
            self.dataSI["TBU"] = TBU
            oldchave = "h"
            newchave = "TBU"
        if setchaves == set(["h", "Pw"]):
            w = self.__find_w("Pw")
            self.dataSI["w"] = w
            oldchave = "Pw"
            newchave = "w"
        if oldchave is None:
            raise Exception("Nao pude resolver com chaves (%s, %s)" % (chave1, chave2))
            
        index = self.basedata.index(oldchave)
        self.basedata[index] = newchave
        otherchave = chave1 if chave2 == oldchave else chave2
        tipo1 = mappingsiglas[newchave]
        val1 = self.dataSI[newchave]
        tipo2 = mappingsiglas[otherchave]
        val2 = self.dataSI[otherchave]
        tipodado = mappingsiglas[dado]
        result = HAPropsSI(tipodado,'P', P, tipo1, val1, tipo2, val2)
        self.dataSI[dado] = result 

    def __compute_all(self):
        for dado in PsicrometryPoint.keys:
            self.__compute_dado(dado)

    def get(self, dado):
        """
        Esta funcao retorna um valor requisitado no ponto.
        Exemplo de uso da funcao:
            
        """
        if dado not in PsicrometryPoint.keys:
            raise Exception("Dado requerido '%s' nao encontrado. Use help" % dado)
        if dado not in self.dataSI:
            self.__compute_dado(dado)
        return self.dataSI[dado]

    def __str__(self):
        return str(self.dataSI)


class PsicrometryChart:
    def __init__(self, points):
        self.figure = None
        self.axis = None
        self.points = points
        self.__verify_pression()

    def __verify_pression(self):
        self.P = self.points[0].get("P")
        for p in self.points:
            if self.P != p.get("P"):
                raise Exception("Para plotar o grafico, todas pressoes devem ser iguais!")
    
    def __getTmin(self):
        Tmin = 1e+3
        for p in self.points:
            if p.get("TBS") < Tmin:
                Tmin = p.get("TBS")
        return Tmin
    def __getTmax(self):
        Tmax = 1e-3
        for p in self.points:
            if p.get("TBS") > Tmax:
                Tmax = p.get("TBS")
        return Tmax

    def __getTBSmaxsat(self, UR=1):
        wmax = self.__getYmaxplot()
        Pwmax = self.P * wmax / (wmax+RaRw)
        TBSmaxsat = HAPropsSI("T","R",UR,"P",self.P,"P_w", Pwmax)
        TBSmaxsat = min(self.__getXmaxplot(), TBSmaxsat)
        return TBSmaxsat

    def __getwmin(self):
        return 0

    def __getwmax(self):
        wmax = 1e-3
        for p in self.points:
            wp = p.get("w")
            if wp > wmax:
                wmax = wp
        return wmax
    
    def __getXmaxplot(self):
        Tmax = self.__getTmax()
        Tmin = self.__getTmin()
        dT = Tmax-Tmin
        return Tmax + 0.2*dT

    def __getXminplot(self):
        Tmax = self.__getTmax()
        Tmin = self.__getTmin()
        dT = Tmax-Tmin
        return Tmin - 0.2*dT

    def __getYminplot(self):
        wmin = self.__getwmin()
        wmax = self.__getwmax()
        dw = wmax - wmin

        val = wmin - 0.2*dw
        return 0 if val < 0 else val

    def __getYmaxplot(self):
        wmax = self.__getwmax()
        wmin = self.__getwmin()
        dw = wmax - wmin
        return wmax + 0.2*dw

    def __getDeltaX(self):
        Xmax = self.__getXmaxplot()
        Xmin = self.__getXminplot()
        if Xmax-Xmin < 10:
            DeltaX = 1
        elif Xmax - Xmin < 20:
            DeltaX = 2
        elif Xmax - Xmin < 50:
            DeltaX = 5
        elif Xmax - Xmin < 100:
            DeltaX = 10
        return DeltaX

    def __getDeltaY(self):
        Ymin = self.__getYminplot()
        Ymax = self.__getYmaxplot()
        if Ymax-Ymin < 10:
            DeltaY = 1
        elif Ymax - Ymin < 20:
            DeltaY = 2
        elif Ymax - Ymin < 50:
            DeltaY = 5
        elif Ymax - Ymin < 100:
            DeltaY = 10
        return DeltaY
    
    def __getDeltaV(self):
        return 0.01


    def __prepare_plot(self):
        if self.figure is None:
            self.figure = plt.Figure()
        if self.axis is None:
            self.axis = plt.gca()

    def __plot_all_points(self):
        Xmax = self.__getXmaxplot()
        for i, p in enumerate(self.points):
            color = next(cycol)
            TBS = p.get("TBS")-Celsius0
            TBU = p.get("TBU")-Celsius0
            TPO = p.get("TPO")-Celsius0
            w = 1000* p.get("w")
            wU = 1000*HAPropsSI("W","R",1,"P",self.P,"B",TBU+Celsius0)
            self.axis.plot((TBS, TBS), (0, w), lw=0.5, ls="dotted",color=color)
            self.axis.plot((TBU, TBU, TBS), (0, wU, w), lw=0.5, ls="dotted",color=color)
            self.axis.plot((TPO, TPO, TBS), (0, w, w), lw=0.5, ls="dotted",color=color)
            self.axis.plot((Xmax-Celsius0, TBS), (w, w), lw=0.5, ls="dotted",color=color)
            self.axis.scatter(TBS, w, marker=".",color=color, label="%d"%(i+1))


    def __plot_saturation(self):
        TBSmaxsat = self.__getTBSmaxsat(UR=1)
        TBSplot = np.linspace(self.__getXminplot(), TBSmaxsat, 33)
        wplot = HAPropsSI("W","R",1,"P",self.P,"T",TBSplot)
        self.axis.plot(TBSplot-Celsius0, 1000*wplot, color='k', lw=1.5)

    def __plot_TBSconst(self):
        DeltaX = self.__getDeltaX()
        Xmin = self.__getXminplot()
        Xmax = self.__getXmaxplot()
        Ymin = self.__getYminplot()
        Ymax = self.__getYmaxplot()
        Xminval = np.ceil((Xmin-Celsius0)/DeltaX)*DeltaX + Celsius0
        TBSsat = self.__getTBSmaxsat(UR=1)
        X = Xminval
        
        while X < TBSsat:
            w = HAPropsSI("W","R",1,"P",self.P,"T",X)
            self.axis.plot((X-Celsius0, X-Celsius0), (1000*Ymin, 1000*w), color='k', lw = 0.5)
            X += DeltaX
        while X < Xmax:
            self.axis.plot((X-Celsius0, X-Celsius0), (1000*Ymin, 1000*Ymax), color='k', lw = 0.5)
            X += DeltaX

    def __plot_TBUconst(self):
        Xmin = self.__getXminplot()
        Xmax = self.__getXmaxplot()
        Ymax = self.__getYmaxplot()
        DeltaX = self.__getDeltaX()
        X = np.ceil((Xmin-Celsius0)/DeltaX)*DeltaX + Celsius0
        TBSmax = self.__getTBSmaxsat()
        while True:
            TBU = HAPropsSI("B","T",X,"P",self.P,"R",0)
            w = HAPropsSI("W","T",TBU,"P",self.P,"R",1)
            if TBU > TBSmax:
                break
            self.axis.plot((X-Celsius0, TBU-Celsius0), (0, 1000*w), color='k', lw = 0.5)
            X += DeltaX
        while True:
            TBU = HAPropsSI("B","T",X,"P",self.P,"W",0)
            TBS = HAPropsSI("T","B",TBU,"P",self.P,"W",Ymax)
            if TBS > Xmax:
                break
            self.axis.plot((X-Celsius0, TBS-Celsius0), (0, 1000*Ymax), color='k', lw = 0.5)
            X += DeltaX 

    def __plot_URconst(self):
        # Lines of constant relative humidity
        for RH in np.arange(0.1, 1, 0.1):
            TBSmax = self.__getTBSmaxsat(RH)
            Tminplot = self.__getXminplot()
            TBSplot = np.linspace(Tminplot, TBSmax, 33)
            wplot = HAPropsSI("W","R",RH,"P",self.P,"T",TBSplot)
            self.axis.plot(TBSplot-Celsius0, 1000*wplot, color='k', lw = 0.5)

    def __plot_Vconst(self):
        
        Xmin = self.__getXminplot()
        Xmax = self.__getXmaxplot()
        Vmin = HAPropsSI("Vda","R",0,"P",self.P,"T",Xmin)
        DeltaV = self.__getDeltaV()
        V = np.ceil(Vmin/DeltaV)*DeltaV
        while True:
            TBU = HAPropsSI("B","R",1,"P",self.P,"Vda",V)
            if TBU > Xmax:
                break
            TBS = HAPropsSI("T","R",0,"P",self.P,"Vda",V)
            wu = HAPropsSI("W","R",1,"P",self.P,"Vda",V)
            self.axis.plot((TBU-Celsius0, TBS-Celsius0), (1000*wu, 0), color='k', lw = 0.5)
            V += DeltaV

    def __plot_chart(self):
        Xmin = self.__getXminplot()
        Xmax = self.__getXmaxplot()
        Ymin = self.__getYminplot()
        Ymax = self.__getYmaxplot()
        self.axis.set_xlim(Xmin-Celsius0, Xmax-Celsius0)
        self.axis.set_ylim(1000*Ymin, 1000*Ymax)
        plt.grid()

        self.axis.set_xlabel(r'Dry bulb temperature $T_{\rm db}$ ($^{\circ}$ C)')
        self.axis.set_ylabel(r'Humidity Ratio $W$ (g/kg)')
        self.axis.yaxis.set_label_position("right")
        self.axis.yaxis.tick_right()
        self.axis.legend(loc="upper left")
        plt.savefig("CartaPsicrometrica.png", dpi=1600)
        plt.savefig("CartaPsicrometrica.pdf", dpi=1600, papertype="a4")
        

    def plot(self):
        self.__prepare_plot()
        self.__plot_saturation()
        self.__plot_URconst()
        # self.__plot_TBSconst()
        self.__plot_TBUconst()
        self.__plot_Vconst()
        self.__plot_all_points()
        self.__plot_chart()


        