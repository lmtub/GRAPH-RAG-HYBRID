void mcf_fec_init(NICInfo *nd, target_phys_addr_t base, qemu_irq *irq)

{

    mcf_fec_state *s;



    qemu_check_nic_model(nd, "mcf_fec");



    s = (mcf_fec_state *)qemu_mallocz(sizeof(mcf_fec_state));

    s->irq = irq;

    s->mmio_index = cpu_register_io_memory(mcf_fec_readfn,

                                           mcf_fec_writefn, s);

    cpu_register_physical_memory(base, 0x400, s->mmio_index);



    s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 mcf_fec_can_receive, mcf_fec_receive, NULL,

                                 mcf_fec_cleanup, s);

    memcpy(s->macaddr, nd->macaddr, 6);

    qemu_format_nic_info_str(s->vc, s->macaddr);

}
