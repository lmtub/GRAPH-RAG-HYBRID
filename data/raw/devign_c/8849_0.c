static int spapr_tce_table_realize(DeviceState *dev)

{

    sPAPRTCETable *tcet = SPAPR_TCE_TABLE(dev);



    if (kvm_enabled()) {

        tcet->table = kvmppc_create_spapr_tce(tcet->liobn,

                                              tcet->window_size,

                                              &tcet->fd);

    }



    if (!tcet->table) {

        size_t table_size = (tcet->window_size >> SPAPR_TCE_PAGE_SHIFT)

            * sizeof(uint64_t);

        tcet->table = g_malloc0(table_size);

    }

    tcet->nb_table = tcet->window_size >> SPAPR_TCE_PAGE_SHIFT;



    trace_spapr_iommu_new_table(tcet->liobn, tcet, tcet->table, tcet->fd);



    memory_region_init_iommu(&tcet->iommu, OBJECT(dev), &spapr_iommu_ops,

                             "iommu-spapr", UINT64_MAX);



    QLIST_INSERT_HEAD(&spapr_tce_tables, tcet, list);



    vmstate_register(DEVICE(tcet), tcet->liobn, &vmstate_spapr_tce_table,

                     tcet);



    return 0;

}
