int pci_add_capability(PCIDevice *pdev, uint8_t cap_id,

                       uint8_t offset, uint8_t size)

{

    int ret;

    Error *local_err = NULL;



    ret = pci_add_capability2(pdev, cap_id, offset, size, &local_err);

    if (local_err) {

        assert(ret < 0);

        error_report_err(local_err);

    } else {

        /* success implies a positive offset in config space */

        assert(ret > 0);

    }

    return ret;

}
