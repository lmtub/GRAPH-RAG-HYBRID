static void next(DBDMA_channel *ch)

{

    uint32_t cp;



    ch->regs[DBDMA_STATUS] &= cpu_to_be32(~BT);



    cp = be32_to_cpu(ch->regs[DBDMA_CMDPTR_LO]);

    ch->regs[DBDMA_CMDPTR_LO] = cpu_to_be32(cp + sizeof(dbdma_cmd));

    dbdma_cmdptr_load(ch);

}
