TC6393xbState *tc6393xb_init(MemoryRegion *sysmem, uint32_t base, qemu_irq irq)

{

    TC6393xbState *s;

    DriveInfo *nand;

    static const MemoryRegionOps tc6393xb_ops = {

        .read = tc6393xb_readb,

        .write = tc6393xb_writeb,

        .endianness = DEVICE_NATIVE_ENDIAN,

        .impl = {

            .min_access_size = 1,

            .max_access_size = 1,

        },

    };



    s = (TC6393xbState *) g_malloc0(sizeof(TC6393xbState));

    s->irq = irq;

    s->gpio_in = qemu_allocate_irqs(tc6393xb_gpio_set, s, TC6393XB_GPIOS);



    s->l3v = *qemu_allocate_irqs(tc6393xb_l3v, s, 1);

    s->blanked = 1;



    s->sub_irqs = qemu_allocate_irqs(tc6393xb_sub_irq, s, TC6393XB_NR_IRQS);



    nand = drive_get(IF_MTD, 0, 0);

    s->flash = nand_init(nand ? blk_bs(blk_by_legacy_dinfo(nand)) : NULL,

                         NAND_MFR_TOSHIBA, 0x76);



    memory_region_init_io(&s->iomem, NULL, &tc6393xb_ops, s, "tc6393xb", 0x10000);

    memory_region_add_subregion(sysmem, base, &s->iomem);



    memory_region_init_ram(&s->vram, NULL, "tc6393xb.vram", 0x100000,

                           &error_abort);

    vmstate_register_ram_global(&s->vram);

    s->vram_ptr = memory_region_get_ram_ptr(&s->vram);

    memory_region_add_subregion(sysmem, base + 0x100000, &s->vram);

    s->scr_width = 480;

    s->scr_height = 640;

    s->con = graphic_console_init(NULL, 0, &tc6393xb_gfx_ops, s);



    return s;

}
