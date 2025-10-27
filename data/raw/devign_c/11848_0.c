static void tcx_init(hwaddr addr, qemu_irq irq, int vram_size, int width,

                     int height, int depth)

{

    DeviceState *dev;

    SysBusDevice *s;



    dev = qdev_create(NULL, "SUNW,tcx");

    qdev_prop_set_uint32(dev, "vram_size", vram_size);

    qdev_prop_set_uint16(dev, "width", width);

    qdev_prop_set_uint16(dev, "height", height);

    qdev_prop_set_uint16(dev, "depth", depth);

    qdev_prop_set_uint64(dev, "prom_addr", addr);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);



    /* 10/ROM : FCode ROM */

    sysbus_mmio_map(s, 0, addr);

    /* 2/STIP : Stipple */

    sysbus_mmio_map(s, 1, addr + 0x04000000ULL);

    /* 3/BLIT : Blitter */

    sysbus_mmio_map(s, 2, addr + 0x06000000ULL);

    /* 5/RSTIP : Raw Stipple */

    sysbus_mmio_map(s, 3, addr + 0x0c000000ULL);

    /* 6/RBLIT : Raw Blitter */

    sysbus_mmio_map(s, 4, addr + 0x0e000000ULL);

    /* 7/TEC : Transform Engine */

    sysbus_mmio_map(s, 5, addr + 0x00700000ULL);

    /* 8/CMAP  : DAC */

    sysbus_mmio_map(s, 6, addr + 0x00200000ULL);

    /* 9/THC : */

    if (depth == 8) {

        sysbus_mmio_map(s, 7, addr + 0x00300000ULL);

    } else {

        sysbus_mmio_map(s, 7, addr + 0x00301000ULL);

    }

    /* 11/DHC : */

    sysbus_mmio_map(s, 8, addr + 0x00240000ULL);

    /* 12/ALT : */

    sysbus_mmio_map(s, 9, addr + 0x00280000ULL);

    /* 0/DFB8 : 8-bit plane */

    sysbus_mmio_map(s, 10, addr + 0x00800000ULL);

    /* 1/DFB24 : 24bit plane */

    sysbus_mmio_map(s, 11, addr + 0x02000000ULL);

    /* 4/RDFB32: Raw framebuffer. Control plane */

    sysbus_mmio_map(s, 12, addr + 0x0a000000ULL);

    /* 9/THC24bits : NetBSD writes here even with 8-bit display: dummy */

    if (depth == 8) {

        sysbus_mmio_map(s, 13, addr + 0x00301000ULL);

    }



    sysbus_connect_irq(s, 0, irq);

}
