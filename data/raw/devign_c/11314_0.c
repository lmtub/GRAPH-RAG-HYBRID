static void branch(DBDMA_channel *ch)

{

    dbdma_cmd *current = &ch->current;



    ch->regs[DBDMA_CMDPTR_LO] = current->cmd_dep;

    ch->regs[DBDMA_STATUS] |= cpu_to_be32(BT);

    dbdma_cmdptr_load(ch);

}
