static int vfio_setup_pcie_cap(VFIOPCIDevice *vdev, int pos, uint8_t size)

{

    uint16_t flags;

    uint8_t type;



    flags = pci_get_word(vdev->pdev.config + pos + PCI_CAP_FLAGS);

    type = (flags & PCI_EXP_FLAGS_TYPE) >> 4;



    if (type != PCI_EXP_TYPE_ENDPOINT &&

        type != PCI_EXP_TYPE_LEG_END &&

        type != PCI_EXP_TYPE_RC_END) {



        error_report("vfio: Assignment of PCIe type 0x%x "

                     "devices is not currently supported", type);

        return -EINVAL;

    }



    if (!pci_bus_is_express(vdev->pdev.bus)) {

        /*

         * Use express capability as-is on PCI bus.  It doesn't make much

         * sense to even expose, but some drivers (ex. tg3) depend on it

         * and guests don't seem to be particular about it.  We'll need

         * to revist this or force express devices to express buses if we

         * ever expose an IOMMU to the guest.

         */

    } else if (pci_bus_is_root(vdev->pdev.bus)) {

        /*

         * On a Root Complex bus Endpoints become Root Complex Integrated

         * Endpoints, which changes the type and clears the LNK & LNK2 fields.

         */

        if (type == PCI_EXP_TYPE_ENDPOINT) {

            vfio_add_emulated_word(vdev, pos + PCI_CAP_FLAGS,

                                   PCI_EXP_TYPE_RC_END << 4,

                                   PCI_EXP_FLAGS_TYPE);



            /* Link Capabilities, Status, and Control goes away */

            if (size > PCI_EXP_LNKCTL) {

                vfio_add_emulated_long(vdev, pos + PCI_EXP_LNKCAP, 0, ~0);

                vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKCTL, 0, ~0);

                vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKSTA, 0, ~0);



#ifndef PCI_EXP_LNKCAP2

#define PCI_EXP_LNKCAP2 44

#endif

#ifndef PCI_EXP_LNKSTA2

#define PCI_EXP_LNKSTA2 50

#endif

                /* Link 2 Capabilities, Status, and Control goes away */

                if (size > PCI_EXP_LNKCAP2) {

                    vfio_add_emulated_long(vdev, pos + PCI_EXP_LNKCAP2, 0, ~0);

                    vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKCTL2, 0, ~0);

                    vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKSTA2, 0, ~0);

                }

            }



        } else if (type == PCI_EXP_TYPE_LEG_END) {

            /*

             * Legacy endpoints don't belong on the root complex.  Windows

             * seems to be happier with devices if we skip the capability.

             */

            return 0;

        }



    } else {

        /*

         * Convert Root Complex Integrated Endpoints to regular endpoints.

         * These devices don't support LNK/LNK2 capabilities, so make them up.

         */

        if (type == PCI_EXP_TYPE_RC_END) {

            vfio_add_emulated_word(vdev, pos + PCI_CAP_FLAGS,

                                   PCI_EXP_TYPE_ENDPOINT << 4,

                                   PCI_EXP_FLAGS_TYPE);

            vfio_add_emulated_long(vdev, pos + PCI_EXP_LNKCAP,

                                   PCI_EXP_LNK_MLW_1 | PCI_EXP_LNK_LS_25, ~0);

            vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKCTL, 0, ~0);

        }



        /* Mark the Link Status bits as emulated to allow virtual negotiation */

        vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKSTA,

                               pci_get_word(vdev->pdev.config + pos +

                                            PCI_EXP_LNKSTA),

                               PCI_EXP_LNKCAP_MLW | PCI_EXP_LNKCAP_SLS);

    }



    pos = pci_add_capability(&vdev->pdev, PCI_CAP_ID_EXP, pos, size);

    if (pos >= 0) {

        vdev->pdev.exp.exp_cap = pos;

    }



    return pos;

}
