void mipsnet_init (int base, qemu_irq irq, NICInfo *nd)

{

    MIPSnetState *s;



    qemu_check_nic_model(nd, "mipsnet");



    s = qemu_mallocz(sizeof(MIPSnetState));



    register_ioport_write(base, 36, 1, mipsnet_ioport_write, s);

    register_ioport_read(base, 36, 1, mipsnet_ioport_read, s);

    register_ioport_write(base, 36, 2, mipsnet_ioport_write, s);

    register_ioport_read(base, 36, 2, mipsnet_ioport_read, s);

    register_ioport_write(base, 36, 4, mipsnet_ioport_write, s);

    register_ioport_read(base, 36, 4, mipsnet_ioport_read, s);



    s->io_base = base;

    s->irq = irq;

    if (nd && nd->vlan) {

        s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                     mipsnet_can_receive, mipsnet_receive, NULL,

                                     mipsnet_cleanup, s);

    } else {

        s->vc = NULL;

    }



    qemu_format_nic_info_str(s->vc, nd->macaddr);



    mipsnet_reset(s);

    register_savevm("mipsnet", 0, 0, mipsnet_save, mipsnet_load, s);

}
