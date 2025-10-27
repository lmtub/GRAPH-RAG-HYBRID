static void smc91c111_release_packet(smc91c111_state *s, int packet)

{

    s->allocated &= ~(1 << packet);

    if (s->tx_alloc == 0x80)

        smc91c111_tx_alloc(s);

    qemu_flush_queued_packets(qemu_get_queue(s->nic));

}
