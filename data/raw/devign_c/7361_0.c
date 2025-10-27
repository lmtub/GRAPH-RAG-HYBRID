static void store_word(DBDMA_channel *ch, int key, uint32_t addr,

                      uint16_t len)

{

    dbdma_cmd *current = &ch->current;

    uint32_t val;



    DBDMA_DPRINTF("store_word\n");



    /* only implements KEY_SYSTEM */



    if (key != KEY_SYSTEM) {

        printf("DBDMA: STORE_WORD, unimplemented key %x\n", key);

        kill_channel(ch);

        return;

    }



    val = current->cmd_dep;

    if (len == 2)

        val >>= 16;

    else if (len == 1)

        val >>= 24;



    cpu_physical_memory_write(addr, (uint8_t*)&val, len);



    if (conditional_wait(ch))

        goto wait;



    current->xfer_status = cpu_to_le16(be32_to_cpu(ch->regs[DBDMA_STATUS]));

    dbdma_cmdptr_save(ch);

    ch->regs[DBDMA_STATUS] &= cpu_to_be32(~FLUSH);



    conditional_interrupt(ch);

    next(ch);



wait:

    qemu_bh_schedule(dbdma_bh);

}
