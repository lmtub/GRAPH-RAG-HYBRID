void dp83932_init(NICInfo *nd, target_phys_addr_t base, int it_shift,

                  MemoryRegion *address_space,

                  qemu_irq irq, void* mem_opaque,

                  void (*memory_rw)(void *opaque, target_phys_addr_t addr, uint8_t *buf, int len, int is_write))

{

    dp8393xState *s;



    qemu_check_nic_model(nd, "dp83932");



    s = g_malloc0(sizeof(dp8393xState));



    s->address_space = address_space;

    s->mem_opaque = mem_opaque;

    s->memory_rw = memory_rw;

    s->it_shift = it_shift;

    s->irq = irq;

    s->watchdog = qemu_new_timer_ns(vm_clock, dp8393x_watchdog, s);

    s->regs[SONIC_SR] = 0x0004; /* only revision recognized by Linux */



    s->conf.macaddr = nd->macaddr;

    s->conf.peer = nd->netdev;



    s->nic = qemu_new_nic(&net_dp83932_info, &s->conf, nd->model, nd->name, s);



    qemu_format_nic_info_str(&s->nic->nc, s->conf.macaddr.a);

    qemu_register_reset(nic_reset, s);

    nic_reset(s);



    memory_region_init_io(&s->mmio, &dp8393x_ops, s,

                          "dp8393x", 0x40 << it_shift);

    memory_region_add_subregion(address_space, base, &s->mmio);

}
