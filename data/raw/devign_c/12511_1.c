void smc91c111_init(NICInfo *nd, uint32_t base, qemu_irq irq)

{

    smc91c111_state *s;

    int iomemtype;



    qemu_check_nic_model(nd, "smc91c111");



    s = (smc91c111_state *)qemu_mallocz(sizeof(smc91c111_state));

    iomemtype = cpu_register_io_memory(0, smc91c111_readfn,

                                       smc91c111_writefn, s);

    cpu_register_physical_memory(base, 16, iomemtype);

    s->irq = irq;

    memcpy(s->macaddr, nd->macaddr, 6);



    smc91c111_reset(s);



    s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 smc91c111_receive, smc91c111_can_receive, s);

    qemu_format_nic_info_str(s->vc, s->macaddr);

    /* ??? Save/restore.  */

}
