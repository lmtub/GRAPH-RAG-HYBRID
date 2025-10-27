static void mcf_fec_enable_rx(mcf_fec_state *s)

{

    mcf_fec_bd bd;



    mcf_fec_read_bd(&bd, s->rx_descriptor);

    s->rx_enabled = ((bd.flags & FEC_BD_E) != 0);

    if (!s->rx_enabled)

        DPRINTF("RX buffer full\n");

}
