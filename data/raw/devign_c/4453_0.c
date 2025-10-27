static void dbdma_cmdptr_save(DBDMA_channel *ch)

{

    DBDMA_DPRINTF("dbdma_cmdptr_save 0x%08x\n",

                  be32_to_cpu(ch->regs[DBDMA_CMDPTR_LO]));

    DBDMA_DPRINTF("xfer_status 0x%08x res_count 0x%04x\n",

                  le16_to_cpu(ch->current.xfer_status),

                  le16_to_cpu(ch->current.res_count));

    cpu_physical_memory_write(be32_to_cpu(ch->regs[DBDMA_CMDPTR_LO]),

                              (uint8_t*)&ch->current, sizeof(dbdma_cmd));

}
