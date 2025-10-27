static void platform_fixed_ioport_init(PCIXenPlatformState* s)

{

    register_ioport_write(XEN_PLATFORM_IOPORT, 16, 4, platform_fixed_ioport_writel, s);

    register_ioport_write(XEN_PLATFORM_IOPORT, 16, 2, platform_fixed_ioport_writew, s);

    register_ioport_write(XEN_PLATFORM_IOPORT, 16, 1, platform_fixed_ioport_writeb, s);

    register_ioport_read(XEN_PLATFORM_IOPORT, 16, 2, platform_fixed_ioport_readw, s);

    register_ioport_read(XEN_PLATFORM_IOPORT, 16, 1, platform_fixed_ioport_readb, s);

}
