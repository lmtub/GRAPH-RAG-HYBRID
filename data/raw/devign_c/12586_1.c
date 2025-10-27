static int spapr_tce_table_realize(DeviceState *dev)

{

    sPAPRTCETable *tcet = SPAPR_TCE_TABLE(dev);



    if (kvm_enabled()) {

        tcet->table = kvmppc_create_spapr_tce(tcet->liobn,

                                              tcet->nb_table <<

                                              tcet->page_shift,

                                              &tcet->fd,

                                              tcet->vfio_accel);

    }



    if (!tcet->table) {

        size_t table_size = tcet->nb_table * sizeof(uint64_t);

        tcet->table = g_malloc0(table_size);

    }



    trace_spapr_iommu_new_table(tcet->liobn, tcet, tcet->table, tcet->fd);



    memory_region_init_iommu(&tcet->iommu, OBJECT(dev), &spapr_iommu_ops,

                             "iommu-spapr",

                             (uint64_t)tcet->nb_table << tcet->page_shift);



    QLIST_INSERT_HEAD(&spapr_tce_tables, tcet, list);



    vmstate_register(DEVICE(tcet), tcet->liobn, &vmstate_spapr_tce_table,

                     tcet);



    return 0;

}
