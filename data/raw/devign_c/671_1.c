static void mv88w8618_eth_init(NICInfo *nd, uint32_t base, qemu_irq irq)

{

    mv88w8618_eth_state *s;

    int iomemtype;



    qemu_check_nic_model(nd, "mv88w8618");



    s = qemu_mallocz(sizeof(mv88w8618_eth_state));

    s->irq = irq;

    s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 eth_receive, eth_can_receive, s);

    iomemtype = cpu_register_io_memory(0, mv88w8618_eth_readfn,

                                       mv88w8618_eth_writefn, s);

    cpu_register_physical_memory(base, MP_ETH_SIZE, iomemtype);

}
