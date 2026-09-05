# Dependências de Sistemas — Meridiano Logística

## Sistema Atlas

O Sistema Atlas processa os pedidos da região Norte.
O Sistema Atlas armazena dados pessoais de clientes.
O Sistema Atlas depende do Banco de Dados Órion para persistência.

## Banco de Dados Órion

A manutenção do Banco de Dados Órion ocorre no primeiro domingo de cada mês.
Durante a manutenção do Banco de Dados Órion o Sistema Atlas fica indisponível.
O Banco de Dados Órion replica para o Banco de Dados Vesta a cada seis horas.

## Sistema Helios

O Sistema Helios processa os pagamentos a fornecedores.
O Sistema Helios depende do Gateway Íris para autorizar transações.
Uma falha no Gateway Íris interrompe todos os pagamentos do Sistema Helios.
O Sistema Helios consulta o Sistema Atlas para validar cada pedido.
