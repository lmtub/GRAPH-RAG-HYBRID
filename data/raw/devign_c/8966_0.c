static int onenand_initfn(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    OneNANDState *s = ONE_NAND(dev);

    uint32_t size = 1 << (24 + ((s->id.dev >> 4) & 7));

    void *ram;



    s->base = (hwaddr)-1;

    s->rdy = NULL;

    s->blocks = size >> BLOCK_SHIFT;

    s->secs = size >> 9;

    s->blockwp = g_malloc(s->blocks);

    s->density_mask = (s->id.dev & 0x08)

        ? (1 << (6 + ((s->id.dev >> 4) & 7))) : 0;

    memory_region_init_io(&s->iomem, OBJECT(s), &onenand_ops, s, "onenand",

                          0x10000 << s->shift);

    if (!s->bdrv) {

        s->image = memset(g_malloc(size + (size >> 5)),

                          0xff, size + (size >> 5));

    } else {

        if (bdrv_is_read_only(s->bdrv)) {

            error_report("Can't use a read-only drive");

            return -1;

        }

        s->bdrv_cur = s->bdrv;

    }

    s->otp = memset(g_malloc((64 + 2) << PAGE_SHIFT),

                    0xff, (64 + 2) << PAGE_SHIFT);

    memory_region_init_ram(&s->ram, OBJECT(s), "onenand.ram",

                           0xc000 << s->shift, &error_abort);

    vmstate_register_ram_global(&s->ram);

    ram = memory_region_get_ram_ptr(&s->ram);

    s->boot[0] = ram + (0x0000 << s->shift);

    s->boot[1] = ram + (0x8000 << s->shift);

    s->data[0][0] = ram + ((0x0200 + (0 << (PAGE_SHIFT - 1))) << s->shift);

    s->data[0][1] = ram + ((0x8010 + (0 << (PAGE_SHIFT - 6))) << s->shift);

    s->data[1][0] = ram + ((0x0200 + (1 << (PAGE_SHIFT - 1))) << s->shift);

    s->data[1][1] = ram + ((0x8010 + (1 << (PAGE_SHIFT - 6))) << s->shift);

    onenand_mem_setup(s);

    sysbus_init_irq(sbd, &s->intr);

    sysbus_init_mmio(sbd, &s->container);

    vmstate_register(dev,

                     ((s->shift & 0x7f) << 24)

                     | ((s->id.man & 0xff) << 16)

                     | ((s->id.dev & 0xff) << 8)

                     | (s->id.ver & 0xff),

                     &vmstate_onenand, s);

    return 0;

}
