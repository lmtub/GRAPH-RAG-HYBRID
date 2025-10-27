static int gen_rp_interrupts_init(PCIDevice *d, Error **errp)

{

    int rc;



    rc = msix_init_exclusive_bar(d, GEN_PCIE_ROOT_PORT_MSIX_NR_VECTOR, 0);



    if (rc < 0) {

        assert(rc == -ENOTSUP);

        error_setg(errp, "Unable to init msix vectors");

    } else {

        msix_vector_use(d, 0);

    }



    return rc;

}
