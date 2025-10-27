static void spapr_tce_reset(DeviceState *dev)

{

    sPAPRTCETable *tcet = SPAPR_TCE_TABLE(dev);

    size_t table_size = tcet->nb_table * sizeof(uint64_t);



    tcet->bypass = false;

    memset(tcet->table, 0, table_size);

}
