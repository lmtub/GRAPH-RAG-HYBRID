static void kill_channel(DBDMA_channel *ch)

{

    DBDMA_DPRINTF("kill_channel\n");



    ch->regs[DBDMA_STATUS] |= cpu_to_be32(DEAD);

    ch->regs[DBDMA_STATUS] &= cpu_to_be32(~ACTIVE);



    qemu_irq_raise(ch->irq);

}
