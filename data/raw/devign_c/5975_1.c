static int armv7m_nvic_init(SysBusDevice *dev)

{

    nvic_state *s = NVIC(dev);

    NVICClass *nc = NVIC_GET_CLASS(s);



    /* The NVIC always has only one CPU */

    s->gic.num_cpu = 1;

    /* Tell the common code we're an NVIC */

    s->gic.revision = 0xffffffff;

    s->num_irq = s->gic.num_irq;

    nc->parent_init(dev);

    gic_init_irqs_and_distributor(&s->gic, s->num_irq);

    /* The NVIC and system controller register area looks like this:

     *  0..0xff : system control registers, including systick

     *  0x100..0xcff : GIC-like registers

     *  0xd00..0xfff : system control registers

     * We use overlaying to put the GIC like registers

     * over the top of the system control register region.

     */

    memory_region_init(&s->container, "nvic", 0x1000);

    /* The system register region goes at the bottom of the priority

     * stack as it covers the whole page.

     */

    memory_region_init_io(&s->sysregmem, &nvic_sysreg_ops, s,

                          "nvic_sysregs", 0x1000);

    memory_region_add_subregion(&s->container, 0, &s->sysregmem);

    /* Alias the GIC region so we can get only the section of it

     * we need, and layer it on top of the system register region.

     */

    memory_region_init_alias(&s->gic_iomem_alias, "nvic-gic", &s->gic.iomem,

                             0x100, 0xc00);

    memory_region_add_subregion_overlap(&s->container, 0x100,

                                        &s->gic_iomem_alias, 1);

    /* Map the whole thing into system memory at the location required

     * by the v7M architecture.

     */

    memory_region_add_subregion(get_system_memory(), 0xe000e000, &s->container);

    s->systick.timer = qemu_new_timer_ns(vm_clock, systick_timer_tick, s);

    return 0;

}
