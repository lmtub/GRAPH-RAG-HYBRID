static int milkymist_minimac2_init(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    MilkymistMinimac2State *s = MILKYMIST_MINIMAC2(dev);

    size_t buffers_size = TARGET_PAGE_ALIGN(3 * MINIMAC2_BUFFER_SIZE);



    sysbus_init_irq(sbd, &s->rx_irq);

    sysbus_init_irq(sbd, &s->tx_irq);



    memory_region_init_io(&s->regs_region, OBJECT(dev), &minimac2_ops, s,

                          "milkymist-minimac2", R_MAX * 4);

    sysbus_init_mmio(sbd, &s->regs_region);



    /* register buffers memory */

    memory_region_init_ram(&s->buffers, OBJECT(dev), "milkymist-minimac2.buffers",

                           buffers_size, &error_abort);

    vmstate_register_ram_global(&s->buffers);

    s->rx0_buf = memory_region_get_ram_ptr(&s->buffers);

    s->rx1_buf = s->rx0_buf + MINIMAC2_BUFFER_SIZE;

    s->tx_buf = s->rx1_buf + MINIMAC2_BUFFER_SIZE;



    sysbus_init_mmio(sbd, &s->buffers);



    qemu_macaddr_default_if_unset(&s->conf.macaddr);

    s->nic = qemu_new_nic(&net_milkymist_minimac2_info, &s->conf,

                          object_get_typename(OBJECT(dev)), dev->id, s);

    qemu_format_nic_info_str(qemu_get_queue(s->nic), s->conf.macaddr.a);



    return 0;

}
