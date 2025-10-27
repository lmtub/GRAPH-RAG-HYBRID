static int pci_vmsvga_initfn(PCIDevice *dev)

{

    struct pci_vmsvga_state_s *s =

        DO_UPCAST(struct pci_vmsvga_state_s, card, dev);



    pci_config_set_vendor_id(s->card.config, PCI_VENDOR_ID_VMWARE);

    pci_config_set_device_id(s->card.config, SVGA_PCI_DEVICE_ID);

    pci_config_set_class(s->card.config, PCI_CLASS_DISPLAY_VGA);

    s->card.config[PCI_CACHE_LINE_SIZE]	= 0x08;		/* Cache line size */

    s->card.config[PCI_LATENCY_TIMER] = 0x40;		/* Latency timer */

    s->card.config[PCI_SUBSYSTEM_VENDOR_ID] = PCI_VENDOR_ID_VMWARE & 0xff;

    s->card.config[PCI_SUBSYSTEM_VENDOR_ID + 1]	= PCI_VENDOR_ID_VMWARE >> 8;

    s->card.config[PCI_SUBSYSTEM_ID] = SVGA_PCI_DEVICE_ID & 0xff;

    s->card.config[PCI_SUBSYSTEM_ID + 1] = SVGA_PCI_DEVICE_ID >> 8;

    s->card.config[PCI_INTERRUPT_LINE] = 0xff;		/* End */



    pci_register_bar(&s->card, 0, 0x10,

                    PCI_BASE_ADDRESS_SPACE_IO, pci_vmsvga_map_ioport);

    pci_register_bar(&s->card, 1, VGA_RAM_SIZE,

                    PCI_BASE_ADDRESS_MEM_PREFETCH, pci_vmsvga_map_mem);



    pci_register_bar(&s->card, 2, SVGA_FIFO_SIZE,

                    PCI_BASE_ADDRESS_MEM_PREFETCH, pci_vmsvga_map_fifo);



    vmsvga_init(&s->chip, VGA_RAM_SIZE);



    if (!dev->rom_bar) {

        /* compatibility with pc-0.13 and older */

        vga_init_vbe(&s->chip.vga);

    }



    return 0;

}
