static bool e1000_has_rxbufs(E1000State *s, size_t total_size)

{

    int bufs;

    /* Fast-path short packets */

    if (total_size <= s->rxbuf_size) {

        return s->mac_reg[RDH] != s->mac_reg[RDT] || !s->check_rxov;

    }

    if (s->mac_reg[RDH] < s->mac_reg[RDT]) {

        bufs = s->mac_reg[RDT] - s->mac_reg[RDH];

    } else if (s->mac_reg[RDH] > s->mac_reg[RDT] || !s->check_rxov) {

        bufs = s->mac_reg[RDLEN] /  sizeof(struct e1000_rx_desc) +

            s->mac_reg[RDT] - s->mac_reg[RDH];

    } else {

        return false;

    }

    return total_size <= bufs * s->rxbuf_size;

}
