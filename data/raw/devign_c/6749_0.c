static void fdt_add_timer_nodes(const VirtBoardInfo *vbi)

{

    /* Note that on A15 h/w these interrupts are level-triggered,

     * but for the GIC implementation provided by both QEMU and KVM

     * they are edge-triggered.

     */

    uint32_t irqflags = GIC_FDT_IRQ_FLAGS_EDGE_LO_HI;



    irqflags = deposit32(irqflags, GIC_FDT_IRQ_PPI_CPU_START,

                         GIC_FDT_IRQ_PPI_CPU_WIDTH, (1 << vbi->smp_cpus) - 1);



    qemu_fdt_add_subnode(vbi->fdt, "/timer");

    qemu_fdt_setprop_string(vbi->fdt, "/timer",

                                "compatible", "arm,armv7-timer");

    qemu_fdt_setprop_cells(vbi->fdt, "/timer", "interrupts",

                               GIC_FDT_IRQ_TYPE_PPI, 13, irqflags,

                               GIC_FDT_IRQ_TYPE_PPI, 14, irqflags,

                               GIC_FDT_IRQ_TYPE_PPI, 11, irqflags,

                               GIC_FDT_IRQ_TYPE_PPI, 10, irqflags);

}
