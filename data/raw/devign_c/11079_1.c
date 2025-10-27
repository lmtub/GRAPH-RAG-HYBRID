static int vfio_add_std_cap(VFIODevice *vdev, uint8_t pos)

{

    PCIDevice *pdev = &vdev->pdev;

    uint8_t cap_id, next, size;

    int ret;



    cap_id = pdev->config[pos];

    next = pdev->config[pos + 1];



    /*

     * If it becomes important to configure capabilities to their actual

     * size, use this as the default when it's something we don't recognize.

     * Since QEMU doesn't actually handle many of the config accesses,

     * exact size doesn't seem worthwhile.

     */

    size = vfio_std_cap_max_size(pdev, pos);



    /*

     * pci_add_capability always inserts the new capability at the head

     * of the chain.  Therefore to end up with a chain that matches the

     * physical device, we insert from the end by making this recursive.

     * This is also why we pre-caclulate size above as cached config space

     * will be changed as we unwind the stack.

     */

    if (next) {

        ret = vfio_add_std_cap(vdev, next);

        if (ret) {

            return ret;

        }

    } else {

        pdev->config[PCI_CAPABILITY_LIST] = 0; /* Begin the rebuild */

    }



    switch (cap_id) {

    case PCI_CAP_ID_MSI:

        ret = vfio_setup_msi(vdev, pos);

        break;

    case PCI_CAP_ID_MSIX:

        ret = vfio_setup_msix(vdev, pos);

        break;

    default:

        ret = pci_add_capability(pdev, cap_id, pos, size);

        break;

    }



    if (ret < 0) {

        error_report("vfio: %04x:%02x:%02x.%x Error adding PCI capability "

                     "0x%x[0x%x]@0x%x: %d", vdev->host.domain,

                     vdev->host.bus, vdev->host.slot, vdev->host.function,

                     cap_id, size, pos, ret);

        return ret;

    }



    return 0;

}
