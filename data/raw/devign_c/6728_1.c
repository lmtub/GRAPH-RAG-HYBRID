static void gem_transmit(CadenceGEMState *s)
{
    unsigned    desc[2];
    hwaddr packet_desc_addr;
    uint8_t     tx_packet[2048];
    uint8_t     *p;
    unsigned    total_bytes;
    /* Do nothing if transmit is not enabled. */
    if (!(s->regs[GEM_NWCTRL] & GEM_NWCTRL_TXENA)) {
        return;
    DB_PRINT("\n");
    /* The packet we will hand off to QEMU.
     * Packets scattered across multiple descriptors are gathered to this
     * one contiguous buffer first.
     */
    p = tx_packet;
    total_bytes = 0;
    /* read current descriptor */
    packet_desc_addr = s->tx_desc_addr;
    DB_PRINT("read descriptor 0x%" HWADDR_PRIx "\n", packet_desc_addr);
    cpu_physical_memory_read(packet_desc_addr,
                             (uint8_t *)desc, sizeof(desc));
    /* Handle all descriptors owned by hardware */
    while (tx_desc_get_used(desc) == 0) {
        /* Do nothing if transmit is not enabled. */
        if (!(s->regs[GEM_NWCTRL] & GEM_NWCTRL_TXENA)) {
            return;
        print_gem_tx_desc(desc);
        /* The real hardware would eat this (and possibly crash).
         * For QEMU let's lend a helping hand.
         */
        if ((tx_desc_get_buffer(desc) == 0) ||
            (tx_desc_get_length(desc) == 0)) {
            DB_PRINT("Invalid TX descriptor @ 0x%x\n",
                     (unsigned)packet_desc_addr);
        /* Gather this fragment of the packet from "dma memory" to our contig.
         * buffer.
         */
        cpu_physical_memory_read(tx_desc_get_buffer(desc), p,
                                 tx_desc_get_length(desc));
        p += tx_desc_get_length(desc);
        total_bytes += tx_desc_get_length(desc);
        /* Last descriptor for this packet; hand the whole thing off */
        if (tx_desc_get_last(desc)) {
            unsigned    desc_first[2];
            /* Modify the 1st descriptor of this packet to be owned by
             * the processor.
             */
            cpu_physical_memory_read(s->tx_desc_addr, (uint8_t *)desc_first,
                                     sizeof(desc_first));
            tx_desc_set_used(desc_first);
            cpu_physical_memory_write(s->tx_desc_addr, (uint8_t *)desc_first,
                                      sizeof(desc_first));
            /* Advance the hardware current descriptor past this packet */
            if (tx_desc_get_wrap(desc)) {
                s->tx_desc_addr = s->regs[GEM_TXQBASE];
            } else {
                s->tx_desc_addr = packet_desc_addr + 8;
            DB_PRINT("TX descriptor next: 0x%08x\n", s->tx_desc_addr);
            s->regs[GEM_TXSTATUS] |= GEM_TXSTATUS_TXCMPL;
            s->regs[GEM_ISR] |= GEM_INT_TXCMPL & ~(s->regs[GEM_IMR]);
            /* Handle interrupt consequences */
            gem_update_int_status(s);
            /* Is checksum offload enabled? */
            if (s->regs[GEM_DMACFG] & GEM_DMACFG_TXCSUM_OFFL) {
                net_checksum_calculate(tx_packet, total_bytes);
            /* Update MAC statistics */
            gem_transmit_updatestats(s, tx_packet, total_bytes);
            /* Send the packet somewhere */
            if (s->phy_loop || (s->regs[GEM_NWCTRL] & GEM_NWCTRL_LOCALLOOP)) {
                gem_receive(qemu_get_queue(s->nic), tx_packet, total_bytes);
            } else {
                qemu_send_packet(qemu_get_queue(s->nic), tx_packet,
                                 total_bytes);
            /* Prepare for next packet */
            p = tx_packet;
            total_bytes = 0;
        /* read next descriptor */
        if (tx_desc_get_wrap(desc)) {
            packet_desc_addr = s->regs[GEM_TXQBASE];
        } else {
            packet_desc_addr += 8;
        DB_PRINT("read descriptor 0x%" HWADDR_PRIx "\n", packet_desc_addr);
        cpu_physical_memory_read(packet_desc_addr,
                                 (uint8_t *)desc, sizeof(desc));
    if (tx_desc_get_used(desc)) {
        s->regs[GEM_TXSTATUS] |= GEM_TXSTATUS_USED;
        s->regs[GEM_ISR] |= GEM_INT_TXUSED & ~(s->regs[GEM_IMR]);
        gem_update_int_status(s);