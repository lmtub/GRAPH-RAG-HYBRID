static void qxl_realize_secondary(PCIDevice *dev, Error **errp)

{

    static int device_id = 1;

    PCIQXLDevice *qxl = PCI_QXL(dev);



    qxl->id = device_id++;

    qxl_init_ramsize(qxl);

    memory_region_init_ram(&qxl->vga.vram, OBJECT(dev), "qxl.vgavram",

                           qxl->vga.vram_size, &error_abort);

    vmstate_register_ram(&qxl->vga.vram, &qxl->pci.qdev);

    qxl->vga.vram_ptr = memory_region_get_ram_ptr(&qxl->vga.vram);

    qxl->vga.con = graphic_console_init(DEVICE(dev), 0, &qxl_ops, qxl);



    qxl_realize_common(qxl, errp);

}
