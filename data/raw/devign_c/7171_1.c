static int qxl_init_secondary(PCIDevice *dev)

{

    static int device_id = 1;

    PCIQXLDevice *qxl = DO_UPCAST(PCIQXLDevice, pci, dev);



    qxl->id = device_id++;

    qxl_init_ramsize(qxl, 16);

    memory_region_init_ram(&qxl->vga.vram, "qxl.vgavram", qxl->vga.vram_size);

    vmstate_register_ram(&qxl->vga.vram, &qxl->pci.qdev);

    qxl->vga.vram_ptr = memory_region_get_ram_ptr(&qxl->vga.vram);



    return qxl_init_common(qxl);

}
