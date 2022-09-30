# Vibrações

**Portuguese version bellow**
**Versão em português abaixo**

## Abstract

This folder containts codes made for vibrations course at UnB. 
Briefly, the dynamic ODE is given by

<div style="text-align: center;"><img style="text-align: center;" src="https://latex.codecogs.com/png.image?\inline&space;\dpi{150}\bg{white}m&space;\ddot{x}&space;&plus;&space;c&space;\dot{x}&space;&plus;&space;k&space;x&space;=&space;f" title="https://latex.codecogs.com/png.image?\inline \dpi{150}\bg{white}m \ddot{x} + c \dot{x} + k x = f" /></div>

Then

* Modeling:
    * Get analitic EDO from real problem
* Free vibration (```f = 0```):
    * Analitic solve this equation.
    * Implement numerical solution
* Harmonic vibration (```f = f0*exp(i*w*t)```):
    * Analitic solve the equation
    * Implement numerical solution
    * Frequency analisis of system
* Non-oscilatory vibration (```f``` transient):
    * Analitic solve using Laplace transformation
    * Implement numerical solution
* Multi Degrees of freedom system:
    * Modal decomposition
    * Dynamic vibration absorber
* Experimental: Estimate parameters
    * Free vibration experiment
    * Harmonic vibration experiment

Some documents are written in English (indicated with ```EN```) and others are in Portuguese (```BR```).

## Documents

* Homework
    * ```BR``` - ```Homework/1_VigaMassaEquivalente.pdf```: Given a beam with bending stiffness ```EI``` and linear density ```mu```, we estimate the equivalent mass ```m``` and spring constant ```k``` of the first modal frequency.
    * ```BR``` - ```Homework/2_EquacoesGovernantes.pdf```: Model a cilinder + spring to find system's ODE using the Lagrange mechanics differential formulation.
    * ```EN``` - ```Homework/3_DropMass.ipynb```: Model the colision of an free object into another conected into a spring-damper.
    * ```BR``` - ```Homework/4_MassaDesbalanceada.pdf```: Model a unbalanced helicopter propeller which rotates with angular speed ```w```.
    * ```EN``` - ```Homework/5_VaribleForce.ipynb```: Find the position ```x``` of a mass-spring-damper system with force ```f``` decomposed in ```step``` and ```ramp``` using laplace transform.
* Experimental
    * ```BR``` - ```Experimental/first_experiment/```: Using a hammer on a cantilever beam, we find vibrational parameters ```xi``` and ```wn``` from exponential decay response ```a``` mesured by an accelerometer. Uses the ```estimate-exponential-decay.ipynb``` theory.
    * ```BR``` - ```Experimental/second_experiment/```: Using a cantilever beam connected with a oscilating piston at its end, we find parameters ```m```, ```c``` and ```k``` from the timed graphs of ```f```(force) and ```a```(acceleration) with different frequencies. Uses the ```estimate-forced-harmonic.ipynb``` theory.
* ```EN``` - ```dynamic-vibration-absorber.ipynb```: Transform a 1 DOF system into a 2 DOF system to minimize the gain ```X1``` of a mass-spring-damper system (```m1```, ```c1```, ```k1``` fixed) by adding another mass-spring-damper system (```m2```, ```c2```, ```k2``` variable)
* ```EN``` - ```estimate-exponential-decay.ipynb```: Using a 'mesured' (artificial generated noisy data) timed exponential decay response ```a``` of 1 DOF mass-sprint-damper system, we find the best parameters ```xi``` and ```wn``` to fit the curve using non-linear least square method with newton's iteration. Made for the ```first_experiment```.
* ```EN``` - ```estimate-forced-harmonic.ipynb```: Using 'mesured' (artificial generated noisy data) force ```f``` and acceleration ```a``` with different frequencies ```w```  of a 1 DOF mass-spring-damper system, we find the best values for ```m```, ```c``` and ```k``` of this system using least square method to fit the curves. Made for the ```second_experiment```.
* ```BR``` - ```forcamento-harmonico.ipynb```: From a 1 DOF mass-spring-damper system with applied harmonic force ```f0*exp(i*w*t)```, we compute the analitical and numeric response from given parameters  ```m```, ```c``` and ```k``` and initial conditions ```x0``` and ```v0```.
* ```BR``` - ```sistema-massa-mola.ipynb```: Using a free (```f=0```) mass-spring-damper system with parameters ```m```, ```c``` and ```k``` and initial conditions ```x0``` and ```v0```, we compute the analitical and numeric response.
* ```EN``` - ```multi-dofs-system.ipynb```: Has the theory and the numerical implementation and modal decomposition for a ```N``` DOFs system.

-------------------------

## Resumo

This folder containts codes made for vibrations course at UnB. 
Briefly, the vibrations ODE is given by
Essa pasta contém códigos feitos para o curso de vibrações na UnB.
Em resumo, a EDO de dinâmica é dada por

<div style="text-align: center;"><img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{150}\bg{white}m&space;\ddot{x}&space;&plus;&space;c&space;\dot{x}&space;&plus;&space;k&space;x&space;=&space;f" title="https://latex.codecogs.com/png.image?\inline \dpi{150}\bg{white}m \ddot{x} + c \dot{x} + k x = f" /></div>

Então

* Vibração livre (```f = 0```):
    * Resolução analitica da equação
    * Implementação da solução numérica
* Vibração harmônica (```f = f0*exp(i*w*t)```):
    * Resolução analitica da equação
    * Implementação da solução numérica
    * Análise de frequência do sistema
* Vibração não oscilatória (```f``` transiente):
    * Resolução analítica usando transformada de Laplace
    * Implementação da solução numérica
* Sistema de vários graus de liberdade:
    * Decomposição modal
    * Absorvedor dinâmico de vibrações
* Experimental: Estimar parâmetros
    * Experimento de vibração livre
    * Experimento de vibração harmônica

Alguns documentos estão escritos em Ingles (indicados por ```EN```) e outros em Português (```BR```).

## Documentos

* Homework
    * ```BR``` - ```Homework/1_VigaMassaEquivalente.pdf```: Dada uma viga com resistência à flexão ```EI``` e densidade linear ```mu```, estimamos a massa ```m``` and mola ```k```  equivalente do primeiro modo de frequência.
    * ```BR``` - ```Homework/2_EquacoesGovernantes.pdf```: Modela dois objetos (encontrar a EDO) usando a equação diferencial da mecânica lagrangeana.
    * ```EN``` - ```Homework/3_DropMass.ipynb```: Modela a colisão de um objeto livre em outro conectado a uma mola e amortecedor.
    * ```BR``` - ```Homework/4_MassaDesbalanceada.pdf```: Modela um rotor desbalanceado que rotaciona com velocidade angular ```w```.
    * ```EN``` - ```Homework/5_VaribleForce.ipynb```: Encontra a posição ```x``` do sistema massa-mola-amortecedor com força ```f``` decomposta em ```degrau``` e ```rampa``` usando a transformada de laplace.
* Experimental
    * ```BR``` - ```Experimental/first_experiment/```: Usando um martelo em uma viga em balanço, encontramos os parâmetros vibracionais ```xi``` e ```wn``` do decaimento exponencial da resposta ```a``` medido por um acelerômetro. Utiliza a teoria do arquivo ```estimate-exponential-decay.ipynb```.
    * ```BR``` - ```Experimental/second_experiment/```: Usando uma viga em balanço conectada a um pistão oscilante na sua extremidade, encontramos os parâmetros ```m```, ```c``` e ```k``` dos gráficos temporais da força ```f``` e aceleração ```a``` com diferentes frequências. Usa a teoria do arquivo ```estimate-forced-harmonic.ipynb```.
* ```EN``` - ```dynamic-vibration-absorber.ipynb```: Tranforma um sistem massa-mola-amortecedor de 1 GL (grau de liberdade) em um sistema com dois GLs para minimizar o ganho ```X1``` do primeiro objeto (```m1```, ```c1```, ```k1``` fixos) adicionando um outro massa-mola-amortecedor (```m2```, ```c2```, ```k2``` variável)
* ```EN``` - ```estimate-exponential-decay.ipynb```: Usando gráficos temporais 'medidos' (dados ruidosos gerados artificialmente) da aceleração ```a``` de um sistema massa-mola-amortecedor, encontramos os melhores parâmetros ```xi``` e ```wn``` para ajudar a curva ```x``` usando o método dos mínimos quadráticos não linear com o método de newton. Feito para o primeiro experimento ```first_experiment```.
* ```EN``` - ```estimate-forced-harmonic.ipynb```: Usando gráficos temporais 'medidos' (dados ruidosos gerados artificialmente) da força ```f``` e aceleração ```a``` com diferentes frequências ```w``` de um sistema massa-mola-amortecedor, encontramos os melhores valores para ```m```, ```c``` e ```k``` do sistema utilizando o método dos mínimos quadráticos. Feito para o segundo experimento ```second_experiment```.
* ```BR``` - ```forcamento-harmonico.ipynb```: Usando um sistema massa-mola-amortecedor com força harmônica aplicada ```f0*exp(i*w*t)```, calculamos a resposta analítica e numérica dados parâmetros  ```m```, ```c``` e ```k``` com condições iniciais ```x0``` e ```v0```.
* ```BR``` - ```sistema-massa-mola.ipynb```: Usando um sistema massa-mola-amortecedor livre (```f=0```) com parâmetros ```m```, ```c``` e ```k``` com condições iniciais ```x0``` e ```v0```, nós calculamos a resposta analítica e numérica do sistema.
* ```EN``` - ```multi-dofs-system.ipynb```: Tem a teoria e uma implementação numérica da decomposição modal para um sistema de ```N``` graus de liberdade.