set_rdt(E1000State *s, int index, uint32_t val)

{

    s->check_rxov = 0;

    s->mac_reg[index] = val & 0xffff;

    if (e1000_has_rxbufs(s, 1)) {

        qemu_flush_queued_packets(&s->nic->nc);

    }

}
