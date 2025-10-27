void mcf_fec_init(MemoryRegion *sysmem, NICInfo *nd,

                  target_phys_addr_t base, qemu_irq *irq)

{

    mcf_fec_state *s;



    qemu_check_nic_model(nd, "mcf_fec");



    s = (mcf_fec_state *)g_malloc0(sizeof(mcf_fec_state));

    s->sysmem = sysmem;

    s->irq = irq;



    memory_region_init_io(&s->iomem, &mcf_fec_ops, s, "fec", 0x400);

    memory_region_add_subregion(sysmem, base, &s->iomem);



    s->conf.macaddr = nd->macaddr;

    s->conf.peer = nd->netdev;



    s->nic = qemu_new_nic(&net_mcf_fec_info, &s->conf, nd->model, nd->name, s);



    qemu_format_nic_info_str(&s->nic->nc, s->conf.macaddr.a);

}
