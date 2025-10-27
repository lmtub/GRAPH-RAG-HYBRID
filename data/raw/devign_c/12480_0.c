static void create_pcie(const VirtBoardInfo *vbi, qemu_irq *pic,

                        bool use_highmem)

{

    hwaddr base_mmio = vbi->memmap[VIRT_PCIE_MMIO].base;

    hwaddr size_mmio = vbi->memmap[VIRT_PCIE_MMIO].size;

    hwaddr base_mmio_high = vbi->memmap[VIRT_PCIE_MMIO_HIGH].base;

    hwaddr size_mmio_high = vbi->memmap[VIRT_PCIE_MMIO_HIGH].size;

    hwaddr base_pio = vbi->memmap[VIRT_PCIE_PIO].base;

    hwaddr size_pio = vbi->memmap[VIRT_PCIE_PIO].size;

    hwaddr base_ecam = vbi->memmap[VIRT_PCIE_ECAM].base;

    hwaddr size_ecam = vbi->memmap[VIRT_PCIE_ECAM].size;

    hwaddr base = base_mmio;

    int nr_pcie_buses = size_ecam / PCIE_MMCFG_SIZE_MIN;

    int irq = vbi->irqmap[VIRT_PCIE];

    MemoryRegion *mmio_alias;

    MemoryRegion *mmio_reg;

    MemoryRegion *ecam_alias;

    MemoryRegion *ecam_reg;

    DeviceState *dev;

    char *nodename;

    int i;



    dev = qdev_create(NULL, TYPE_GPEX_HOST);

    qdev_init_nofail(dev);



    /* Map only the first size_ecam bytes of ECAM space */

    ecam_alias = g_new0(MemoryRegion, 1);

    ecam_reg = sysbus_mmio_get_region(SYS_BUS_DEVICE(dev), 0);

    memory_region_init_alias(ecam_alias, OBJECT(dev), "pcie-ecam",

                             ecam_reg, 0, size_ecam);

    memory_region_add_subregion(get_system_memory(), base_ecam, ecam_alias);



    /* Map the MMIO window into system address space so as to expose

     * the section of PCI MMIO space which starts at the same base address

     * (ie 1:1 mapping for that part of PCI MMIO space visible through

     * the window).

     */

    mmio_alias = g_new0(MemoryRegion, 1);

    mmio_reg = sysbus_mmio_get_region(SYS_BUS_DEVICE(dev), 1);

    memory_region_init_alias(mmio_alias, OBJECT(dev), "pcie-mmio",

                             mmio_reg, base_mmio, size_mmio);

    memory_region_add_subregion(get_system_memory(), base_mmio, mmio_alias);



    if (use_highmem) {

        /* Map high MMIO space */

        MemoryRegion *high_mmio_alias = g_new0(MemoryRegion, 1);



        memory_region_init_alias(high_mmio_alias, OBJECT(dev), "pcie-mmio-high",

                                 mmio_reg, base_mmio_high, size_mmio_high);

        memory_region_add_subregion(get_system_memory(), base_mmio_high,

                                    high_mmio_alias);

    }



    /* Map IO port space */

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 2, base_pio);



    for (i = 0; i < GPEX_NUM_IRQS; i++) {

        sysbus_connect_irq(SYS_BUS_DEVICE(dev), i, pic[irq + i]);

    }



    nodename = g_strdup_printf("/pcie@%" PRIx64, base);

    qemu_fdt_add_subnode(vbi->fdt, nodename);

    qemu_fdt_setprop_string(vbi->fdt, nodename,

                            "compatible", "pci-host-ecam-generic");

    qemu_fdt_setprop_string(vbi->fdt, nodename, "device_type", "pci");

    qemu_fdt_setprop_cell(vbi->fdt, nodename, "#address-cells", 3);

    qemu_fdt_setprop_cell(vbi->fdt, nodename, "#size-cells", 2);

    qemu_fdt_setprop_cells(vbi->fdt, nodename, "bus-range", 0,

                           nr_pcie_buses - 1);



    qemu_fdt_setprop_cells(vbi->fdt, nodename, "msi-parent", vbi->v2m_phandle);



    qemu_fdt_setprop_sized_cells(vbi->fdt, nodename, "reg",

                                 2, base_ecam, 2, size_ecam);



    if (use_highmem) {

        qemu_fdt_setprop_sized_cells(vbi->fdt, nodename, "ranges",

                                     1, FDT_PCI_RANGE_IOPORT, 2, 0,

                                     2, base_pio, 2, size_pio,

                                     1, FDT_PCI_RANGE_MMIO, 2, base_mmio,

                                     2, base_mmio, 2, size_mmio,

                                     1, FDT_PCI_RANGE_MMIO_64BIT,

                                     2, base_mmio_high,

                                     2, base_mmio_high, 2, size_mmio_high);

    } else {

        qemu_fdt_setprop_sized_cells(vbi->fdt, nodename, "ranges",

                                     1, FDT_PCI_RANGE_IOPORT, 2, 0,

                                     2, base_pio, 2, size_pio,

                                     1, FDT_PCI_RANGE_MMIO, 2, base_mmio,

                                     2, base_mmio, 2, size_mmio);

    }



    qemu_fdt_setprop_cell(vbi->fdt, nodename, "#interrupt-cells", 1);

    create_pcie_irq_map(vbi, vbi->gic_phandle, irq, nodename);



    g_free(nodename);

}
