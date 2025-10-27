void pci_cirrus_vga_init(PCIBus *bus, DisplayState *ds, uint8_t *vga_ram_base,

                         unsigned long vga_ram_offset, int vga_ram_size)

{

    PCICirrusVGAState *d;

    uint8_t *pci_conf;

    CirrusVGAState *s;

    int device_id;



    device_id = CIRRUS_ID_CLGD5446;



    /* setup PCI configuration registers */

    d = (PCICirrusVGAState *)pci_register_device(bus, "Cirrus VGA",

                                                 sizeof(PCICirrusVGAState),

                                                 -1, NULL, NULL);

    pci_conf = d->dev.config;

    pci_conf[0x00] = (uint8_t) (PCI_VENDOR_CIRRUS & 0xff);

    pci_conf[0x01] = (uint8_t) (PCI_VENDOR_CIRRUS >> 8);

    pci_conf[0x02] = (uint8_t) (device_id & 0xff);

    pci_conf[0x03] = (uint8_t) (device_id >> 8);

    pci_conf[0x04] = PCI_COMMAND_IOACCESS | PCI_COMMAND_MEMACCESS;

    pci_conf[0x0a] = PCI_CLASS_SUB_VGA;

    pci_conf[0x0b] = PCI_CLASS_BASE_DISPLAY;

    pci_conf[0x0e] = PCI_CLASS_HEADERTYPE_00h;



    /* setup VGA */

    s = &d->cirrus_vga;

    vga_common_init((VGAState *)s,

                    ds, vga_ram_base, vga_ram_offset, vga_ram_size);

    cirrus_init_common(s, device_id, 1);



    s->console = graphic_console_init(s->ds, s->update, s->invalidate,

                                      s->screen_dump, s->text_update, s);



    s->pci_dev = (PCIDevice *)d;



    /* setup memory space */

    /* memory #0 LFB */

    /* memory #1 memory-mapped I/O */

    /* XXX: s->vram_size must be a power of two */

    pci_register_io_region((PCIDevice *)d, 0, 0x2000000,

			   PCI_ADDRESS_SPACE_MEM_PREFETCH, cirrus_pci_lfb_map);

    if (device_id == CIRRUS_ID_CLGD5446) {

        pci_register_io_region((PCIDevice *)d, 1, CIRRUS_PNPMMIO_SIZE,

                               PCI_ADDRESS_SPACE_MEM, cirrus_pci_mmio_map);

    }

    /* XXX: ROM BIOS */

}
