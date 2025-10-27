static void start_output(DBDMA_channel *ch, int key, uint32_t addr,

                        uint16_t req_count, int is_last)

{

    DBDMA_DPRINTF("start_output\n");



    /* KEY_REGS, KEY_DEVICE and KEY_STREAM

     * are not implemented in the mac-io chip

     */



    DBDMA_DPRINTF("addr 0x%x key 0x%x\n", addr, key);

    if (!addr || key > KEY_STREAM3) {

        kill_channel(ch);

        return;

    }



    ch->io.addr = addr;

    ch->io.len = req_count;

    ch->io.is_last = is_last;

    ch->io.dma_end = dbdma_end;

    ch->io.is_dma_out = 1;

    ch->processing = 1;

    ch->rw(&ch->io);

}
