void sm501_init(MemoryRegion *address_space_mem, uint32_t base,

                uint32_t local_mem_bytes, qemu_irq irq, CharDriverState *chr)

{

    SM501State * s;

    DeviceState *dev;

    MemoryRegion *sm501_system_config = g_new(MemoryRegion, 1);

    MemoryRegion *sm501_disp_ctrl = g_new(MemoryRegion, 1);

    MemoryRegion *sm501_2d_engine = g_new(MemoryRegion, 1);



    /* allocate management data region */

    s = (SM501State *)g_malloc0(sizeof(SM501State));

    s->base = base;

    s->local_mem_size_index

	= get_local_mem_size_index(local_mem_bytes);

    SM501_DPRINTF("local mem size=%x. index=%d\n", get_local_mem_size(s),

		  s->local_mem_size_index);

    s->system_control = 0x00100000;

    s->misc_control = 0x00001000; /* assumes SH, active=low */

    s->dc_panel_control = 0x00010000;

    s->dc_crt_control = 0x00010000;



    /* allocate local memory */

    memory_region_init_ram(&s->local_mem_region, "sm501.local",

                           local_mem_bytes);

    vmstate_register_ram_global(&s->local_mem_region);

    s->local_mem = memory_region_get_ram_ptr(&s->local_mem_region);

    memory_region_add_subregion(address_space_mem, base, &s->local_mem_region);



    /* map mmio */

    memory_region_init_io(sm501_system_config, &sm501_system_config_ops, s,

                          "sm501-system-config", 0x6c);

    memory_region_add_subregion(address_space_mem, base + MMIO_BASE_OFFSET,

                                sm501_system_config);

    memory_region_init_io(sm501_disp_ctrl, &sm501_disp_ctrl_ops, s,

                          "sm501-disp-ctrl", 0x1000);

    memory_region_add_subregion(address_space_mem,

                                base + MMIO_BASE_OFFSET + SM501_DC,

                                sm501_disp_ctrl);

    memory_region_init_io(sm501_2d_engine, &sm501_2d_engine_ops, s,

                          "sm501-2d-engine", 0x54);

    memory_region_add_subregion(address_space_mem,

                                base + MMIO_BASE_OFFSET + SM501_2D_ENGINE,

                                sm501_2d_engine);



    /* bridge to usb host emulation module */

    dev = qdev_create(NULL, "sysbus-ohci");

    qdev_prop_set_uint32(dev, "num-ports", 2);

    qdev_prop_set_taddr(dev, "dma-offset", base);

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0,

                    base + MMIO_BASE_OFFSET + SM501_USB_HOST);

    sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, irq);



    /* bridge to serial emulation module */

    if (chr) {

        serial_mm_init(address_space_mem,

                       base + MMIO_BASE_OFFSET + SM501_UART0, 2,

                       NULL, /* TODO : chain irq to IRL */

                       115200, chr, DEVICE_NATIVE_ENDIAN);

    }



    /* create qemu graphic console */

    s->con = graphic_console_init(sm501_update_display, NULL,

                                  NULL, NULL, s);

}
