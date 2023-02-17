# Métodos Numéricos em Termofluidos

Fluidos frequentemente são simulados por computador.

O jeito que são simulados é resolvendo a EDP de Navier-Stokes.

Até chegarmos ao ponto de resolver a EDP, é necessário aprender sobre

* Trabalho 1 - Introdução à computação científica
    * Multiplicar matrizes
    * Encontrar raizes (Bisecção, secante, Newton, ...)
    * Aproximar Taylor
* Trabalho 2 - Resolver EDOs:
    * Método de Euler explicito
    * Método de Euler implicito
    * Método de Runge-Kutta
    * Sistemas lineares de EDOs
* Trabalho 3 - Resolver EDPs:
    * Usar métodos explicitos de Método das Diferenças Finitas (FDM)
    * Resolver sistemas linares 
    * Usar métodos implicitos de FDM

Como trabalho final, foi colocado resolver o problema da cavidade bidimensional quadrada

<img src="https://raw.githubusercontent.com/carlos-adir/UnB-Courses/main/mntf/img/liddrivencavity.png" width="30%">

Daí pra resolver o problema, tem dois arquivos:

* cavidade_fdm_explicit.py
* cavidade_fdm_explicit.c

Ambos não funcionam (ainda).

## FEM

Também foi feito usando o método dos elementos finitos, que estão dentro da pasta ```fem```.

