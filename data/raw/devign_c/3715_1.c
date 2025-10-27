void dp83932_init(NICInfo *nd, target_phys_addr_t base, int it_shift,

                  qemu_irq irq, void* mem_opaque,

                  void (*memory_rw)(void *opaque, target_phys_addr_t addr, uint8_t *buf, int len, int is_write))

{

    dp8393xState *s;

    int io;



    qemu_check_nic_model(nd, "dp83932");



    s = qemu_mallocz(sizeof(dp8393xState));



    s->mem_opaque = mem_opaque;

    s->memory_rw = memory_rw;

    s->it_shift = it_shift;

    s->irq = irq;

    s->watchdog = qemu_new_timer(vm_clock, dp8393x_watchdog, s);

    s->regs[SONIC_SR] = 0x0004; /* only revision recognized by Linux */



    s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 nic_receive, nic_can_receive, s);



    qemu_format_nic_info_str(s->vc, nd->macaddr);

    qemu_register_reset(nic_reset, s);

    nic_reset(s);



    io = cpu_register_io_memory(0, dp8393x_read, dp8393x_write, s);

    cpu_register_physical_memory(base, 0x40 << it_shift, io);

}
