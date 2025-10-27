static pcibus_t pci_bar_address(PCIDevice *d,

				int reg, uint8_t type, pcibus_t size)

{

    pcibus_t new_addr, last_addr;

    int bar = pci_bar(d, reg);

    uint16_t cmd = pci_get_word(d->config + PCI_COMMAND);



    if (type & PCI_BASE_ADDRESS_SPACE_IO) {

        if (!(cmd & PCI_COMMAND_IO)) {

            return PCI_BAR_UNMAPPED;

        }

        new_addr = pci_get_long(d->config + bar) & ~(size - 1);

        last_addr = new_addr + size - 1;

        /* Check if 32 bit BAR wraps around explicitly.

         * TODO: make priorities correct and remove this work around.

         */

        if (last_addr <= new_addr || new_addr == 0 || last_addr >= UINT32_MAX) {

            return PCI_BAR_UNMAPPED;

        }

        return new_addr;

    }



    if (!(cmd & PCI_COMMAND_MEMORY)) {

        return PCI_BAR_UNMAPPED;

    }

    if (type & PCI_BASE_ADDRESS_MEM_TYPE_64) {

        new_addr = pci_get_quad(d->config + bar);

    } else {

        new_addr = pci_get_long(d->config + bar);

    }

    /* the ROM slot has a specific enable bit */

    if (reg == PCI_ROM_SLOT && !(new_addr & PCI_ROM_ADDRESS_ENABLE)) {

        return PCI_BAR_UNMAPPED;

    }

    new_addr &= ~(size - 1);

    last_addr = new_addr + size - 1;

    /* NOTE: we do not support wrapping */

    /* XXX: as we cannot support really dynamic

       mappings, we handle specific values as invalid

       mappings. */

    if (last_addr <= new_addr || new_addr == 0 ||

        last_addr == PCI_BAR_UNMAPPED) {

        return PCI_BAR_UNMAPPED;

    }



    /* Now pcibus_t is 64bit.

     * Check if 32 bit BAR wraps around explicitly.

     * Without this, PC ide doesn't work well.

     * TODO: remove this work around.

     */

    if  (!(type & PCI_BASE_ADDRESS_MEM_TYPE_64) && last_addr >= UINT32_MAX) {

        return PCI_BAR_UNMAPPED;

    }



    /*

     * OS is allowed to set BAR beyond its addressable

     * bits. For example, 32 bit OS can set 64bit bar

     * to >4G. Check it. TODO: we might need to support

     * it in the future for e.g. PAE.

     */

    if (last_addr >= HWADDR_MAX) {

        return PCI_BAR_UNMAPPED;

    }



    return new_addr;

}
