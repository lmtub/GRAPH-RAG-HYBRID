static int qxl_init_primary(PCIDevice *dev)

{

    PCIQXLDevice *qxl = DO_UPCAST(PCIQXLDevice, pci, dev);

    VGACommonState *vga = &qxl->vga;

    PortioList *qxl_vga_port_list = g_new(PortioList, 1);

    int rc;



    qxl->id = 0;

    qxl_init_ramsize(qxl);

    vga->vram_size_mb = qxl->vga.vram_size >> 20;

    vga_common_init(vga, OBJECT(dev), true);

    vga_init(vga, OBJECT(dev),

             pci_address_space(dev), pci_address_space_io(dev), false);

    portio_list_init(qxl_vga_port_list, OBJECT(dev), qxl_vga_portio_list,

                     vga, "vga");

    portio_list_set_flush_coalesced(qxl_vga_port_list);

    portio_list_add(qxl_vga_port_list, pci_address_space_io(dev), 0x3b0);



    vga->con = graphic_console_init(DEVICE(dev), 0, &qxl_ops, qxl);

    qemu_spice_display_init_common(&qxl->ssd);



    rc = qxl_init_common(qxl);

    if (rc != 0) {

        return rc;

    }



    qxl->ssd.dcl.ops = &display_listener_ops;

    qxl->ssd.dcl.con = vga->con;

    register_displaychangelistener(&qxl->ssd.dcl);

    return rc;

}
