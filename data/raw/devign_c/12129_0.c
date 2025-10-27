qemu_irq *mcf_intc_init(MemoryRegion *sysmem,

                        target_phys_addr_t base,

                        CPUM68KState *env)

{

    mcf_intc_state *s;



    s = g_malloc0(sizeof(mcf_intc_state));

    s->env = env;

    mcf_intc_reset(s);



    memory_region_init_io(&s->iomem, &mcf_intc_ops, s, "mcf", 0x100);

    memory_region_add_subregion(sysmem, base, &s->iomem);



    return qemu_allocate_irqs(mcf_intc_set_irq, s, 64);

}
