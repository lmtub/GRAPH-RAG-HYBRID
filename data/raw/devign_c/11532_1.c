qemu_irq *arm_gic_init(uint32_t base, qemu_irq parent_irq)

{

    gic_state *s;

    qemu_irq *qi;

    int iomemtype;



    s = (gic_state *)qemu_mallocz(sizeof(gic_state));

    if (!s)

        return NULL;

    qi = qemu_allocate_irqs(gic_set_irq, s, GIC_NIRQ);

    s->parent_irq = parent_irq;

    if (base != 0xffffffff) {

        iomemtype = cpu_register_io_memory(0, gic_cpu_readfn,

                                           gic_cpu_writefn, s);

        cpu_register_physical_memory(base, 0x00000fff, iomemtype);

        iomemtype = cpu_register_io_memory(0, gic_dist_readfn,

                                           gic_dist_writefn, s);

        cpu_register_physical_memory(base + 0x1000, 0x00000fff, iomemtype);

        s->base = base;

    } else {

        s->base = 0;

    }

    gic_reset(s);

    return qi;

}
