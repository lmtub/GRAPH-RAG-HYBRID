static int qxl_init_primary(PCIDevice *dev)

{

    PCIQXLDevice *qxl = DO_UPCAST(PCIQXLDevice, pci, dev);

    VGACommonState *vga = &qxl->vga;

    PortioList *qxl_vga_port_list = g_new(PortioList, 1);

    DisplayState *ds;

    int rc;



    qxl->id = 0;

    qxl_init_ramsize(qxl);

    vga->vram_size_mb = qxl->vga.vram_size >> 20;

    vga_common_init(vga);

    vga_init(vga, pci_address_space(dev), pci_address_space_io(dev), false);

    portio_list_init(qxl_vga_port_list, qxl_vga_portio_list, vga, "vga");

    portio_list_add(qxl_vga_port_list, pci_address_space_io(dev), 0x3b0);



    vga->con = graphic_console_init(qxl_hw_update, qxl_hw_invalidate,

                                    qxl_hw_screen_dump, qxl_hw_text_update,

                                    qxl);

    qxl->ssd.con = vga->con,

    qemu_spice_display_init_common(&qxl->ssd);



    rc = qxl_init_common(qxl);

    if (rc != 0) {

        return rc;

    }



    qxl->ssd.dcl.ops = &display_listener_ops;

    ds = qemu_console_displaystate(vga->con);

    register_displaychangelistener(ds, &qxl->ssd.dcl);

    return rc;

}
