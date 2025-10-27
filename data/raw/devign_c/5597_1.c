static void tcx_realizefn(DeviceState *dev, Error **errp)

{

    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);

    TCXState *s = TCX(dev);

    ram_addr_t vram_offset = 0;

    int size, ret;

    uint8_t *vram_base;

    char *fcode_filename;



    memory_region_init_ram(&s->vram_mem, OBJECT(s), "tcx.vram",

                           s->vram_size * (1 + 4 + 4), &error_abort);

    vmstate_register_ram_global(&s->vram_mem);

    memory_region_set_log(&s->vram_mem, true, DIRTY_MEMORY_VGA);

    vram_base = memory_region_get_ram_ptr(&s->vram_mem);



    /* 10/ROM : FCode ROM */

    vmstate_register_ram_global(&s->rom);

    fcode_filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, TCX_ROM_FILE);

    if (fcode_filename) {

        ret = load_image_targphys(fcode_filename, s->prom_addr,

                                  FCODE_MAX_ROM_SIZE);

        g_free(fcode_filename);

        if (ret < 0 || ret > FCODE_MAX_ROM_SIZE) {

            error_report("tcx: could not load prom '%s'", TCX_ROM_FILE);

        }

    }



    /* 0/DFB8 : 8-bit plane */

    s->vram = vram_base;

    size = s->vram_size;

    memory_region_init_alias(&s->vram_8bit, OBJECT(s), "tcx.vram.8bit",

                             &s->vram_mem, vram_offset, size);

    sysbus_init_mmio(sbd, &s->vram_8bit);

    vram_offset += size;

    vram_base += size;



    /* 1/DFB24 : 24bit plane */

    size = s->vram_size * 4;

    s->vram24 = (uint32_t *)vram_base;

    s->vram24_offset = vram_offset;

    memory_region_init_alias(&s->vram_24bit, OBJECT(s), "tcx.vram.24bit",

                             &s->vram_mem, vram_offset, size);

    sysbus_init_mmio(sbd, &s->vram_24bit);

    vram_offset += size;

    vram_base += size;



    /* 4/RDFB32 : Raw Framebuffer */

    size = s->vram_size * 4;

    s->cplane = (uint32_t *)vram_base;

    s->cplane_offset = vram_offset;

    memory_region_init_alias(&s->vram_cplane, OBJECT(s), "tcx.vram.cplane",

                             &s->vram_mem, vram_offset, size);

    sysbus_init_mmio(sbd, &s->vram_cplane);



    /* 9/THC24bits : NetBSD writes here even with 8-bit display: dummy */

    if (s->depth == 8) {

        memory_region_init_io(&s->thc24, OBJECT(s), &tcx_dummy_ops, s,

                              "tcx.thc24", TCX_THC_NREGS);

        sysbus_init_mmio(sbd, &s->thc24);

    }



    sysbus_init_irq(sbd, &s->irq);



    if (s->depth == 8) {

        s->con = graphic_console_init(DEVICE(dev), 0, &tcx_ops, s);

    } else {

        s->con = graphic_console_init(DEVICE(dev), 0, &tcx24_ops, s);

    }

    s->thcmisc = 0;



    qemu_console_resize(s->con, s->width, s->height);

}
