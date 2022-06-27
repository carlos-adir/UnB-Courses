# Vibrações

## Abstract

This folder containts codes made for vibrations. 
Briefly, the easier **DOE**(Differential Ordinary Equation) in vibrations are given by

<img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{150}\bg{white}m&space;\ddot{x}&space;&plus;&space;c&space;\dot{x}&space;&plus;&space;k&space;x&space;=&space;f" title="https://latex.codecogs.com/png.image?\inline \dpi{150}\bg{white}m \ddot{x} + c \dot{x} + k x = f" />

Then

* We solve it using numerical methods
    * Runge-Kutta
    * Generalized-Finite-Difference
* Applicable to ```n``` Degrees Of Freedom.
* We do the analysis of stability of a given system

Some documents are written in English (indicated with ```EN```) and others are in Portuguese (```BR```).

## Resumo

Essa pasta contém códigos feitos para vibrações.
Em resumo, a **EDO**(Equação Diferencial Ordinária) em vibrações é dada por

<img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{150}\bg{white}m&space;\ddot{x}&space;&plus;&space;c&space;\dot{x}&space;&plus;&space;k&space;x&space;=&space;f" title="https://latex.codecogs.com/png.image?\inline \dpi{150}\bg{white}m \ddot{x} + c \dot{x} + k x = f" />

Então

* Resolvemos a equação usando métodos numéricos
    * Runge-Kutta
    * Diferenças Finitas Generalizadas
* Aplicável a ```n``` Graus de Liberdade
* Fazemos a análise de estabilidade de um sistema dado 

Alguns documentos estão escritos em Ingles (indicados por ```EN```) e outros em Português (```BR```).

## Documentos

* ```PT``` ```sistema-massa-mola.ipynb```: Resolve a EDO com apenas 1 grau de liberdade analiticamente e numericamente usando Runge-Kutta e Diferenças Finitas Generalizadas.
* ```PT``` ```Vibracoes_Viga.pdf```: Um exemplo de calcular a massa equivalente de uma viga de Euler-Bernoulli
* ```PT``` ```Vibracoes_Exercicios2.pdf```: Calcula a equação governante de dois objetos usando a equação diferencial de Lagrange.