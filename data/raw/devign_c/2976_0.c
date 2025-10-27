static PCIReqIDCache pci_req_id_cache_get(PCIDevice *dev)

{

    PCIDevice *parent;

    PCIReqIDCache cache = {

        .dev = dev,

        .type = PCI_REQ_ID_BDF,

    };



    while (!pci_bus_is_root(dev->bus)) {

        /* We are under PCI/PCIe bridges */

        parent = dev->bus->parent_dev;

        if (pci_is_express(parent)) {

            if (pcie_cap_get_type(parent) == PCI_EXP_TYPE_PCI_BRIDGE) {

                /* When we pass through PCIe-to-PCI/PCIX bridges, we

                 * override the requester ID using secondary bus

                 * number of parent bridge with zeroed devfn

                 * (pcie-to-pci bridge spec chap 2.3). */

                cache.type = PCI_REQ_ID_SECONDARY_BUS;

                cache.dev = dev;

            }

        } else {

            /* Legacy PCI, override requester ID with the bridge's

             * BDF upstream.  When the root complex connects to

             * legacy PCI devices (including buses), it can only

             * obtain requester ID info from directly attached

             * devices.  If devices are attached under bridges, only

             * the requester ID of the bridge that is directly

             * attached to the root complex can be recognized. */

            cache.type = PCI_REQ_ID_BDF;

            cache.dev = parent;

        }

        dev = parent;

    }



    return cache;

}
