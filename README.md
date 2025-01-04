# Modelo de Hodgkin-Huxley com Bainha de Mielina

Este projeto implementa o modelo de Hodgkin-Huxley (HH) para um cabo (neurônio representado em 1D) com bainha de mielina. O trabalho foi desenvolvido como parte da disciplina **DCC103 - Neurociência Computacional** da **Universidade Federal de Juiz de Fora (UFJF)**.

## 📋 Objetivo

Simular a dinâmica de um neurônio utilizando o modelo HH em 1D, analisando a influência da bainha de mielina no comportamento do sinal elétrico.

## 🖥️ Execução do Código

O código pode ser executado via terminal utilizando o comando:

```bash
$ main.<<ext>> config.txt
```

- **`<<ext>>`**: Substitua pela extensão do código (e.g., `.py`, `.cpp`).
- **`config.txt`**: Arquivo de configuração contendo os parâmetros da simulação.

### Saídas Geradas

1. **Tabela**: Valores em cada instante da simulação.
2. **GIF**: Gráfico Voltagem x Posição ao longo do tempo.
3. Outros gráficos e tabelas opcionais, como:
   - Comparação entre regiões com e sem bainha de mielina.
   - Gráficos de abertura dos canais iônicos e suas relações.

## 📊 Estrutura do Projeto

- **Código-fonte**: Arquivos `.py` para implementação do modelo.
- **Apresentação**: Arquivo `.pdf` contendo os tópicos abordados.
- **Documentação**: Arquivo `.pdf` ou no formato web (`html/css/js`) explicando o código.
- **Configuração**: Arquivo `config.txt` com os parâmetros utilizados na simulação.

## 🚀 Funcionalidades Extras

- Geração de gráficos comparativos e análises detalhadas.
- Possibilidade de personalizar os parâmetros do modelo via função.
