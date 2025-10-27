void pci_bridge_write_config(PCIDevice *d,

                             uint32_t address, uint32_t val, int len)

{

    PCIBridge *s = PCI_BRIDGE(d);

    uint16_t oldctl = pci_get_word(d->config + PCI_BRIDGE_CONTROL);

    uint16_t newctl;



    pci_default_write_config(d, address, val, len);



    if (ranges_overlap(address, len, PCI_COMMAND, 2) ||



        /* io base/limit */

        ranges_overlap(address, len, PCI_IO_BASE, 2) ||



        /* memory base/limit, prefetchable base/limit and

           io base/limit upper 16 */

        ranges_overlap(address, len, PCI_MEMORY_BASE, 20) ||



        /* vga enable */

        ranges_overlap(address, len, PCI_BRIDGE_CONTROL, 2)) {

        pci_bridge_update_mappings(s);

    }



    newctl = pci_get_word(d->config + PCI_BRIDGE_CONTROL);

    if (~oldctl & newctl & PCI_BRIDGE_CTL_BUS_RESET) {

        /* Trigger hot reset on 0->1 transition. */

        pci_bus_reset(&s->sec_bus);

    }

}
