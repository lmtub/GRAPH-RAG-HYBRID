void spapr_iommu_init(void)

{

    QLIST_INIT(&spapr_tce_tables);



    /* hcall-tce */

    spapr_register_hypercall(H_PUT_TCE, h_put_tce);

}
