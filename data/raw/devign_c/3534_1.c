void icp_pit_init(uint32_t base, qemu_irq *pic, int irq)

{

    int iomemtype;

    icp_pit_state *s;



    s = (icp_pit_state *)qemu_mallocz(sizeof(icp_pit_state));

    s->base = base;

    /* Timer 0 runs at the system clock speed (40MHz).  */

    s->timer[0] = arm_timer_init(40000000, pic[irq]);

    /* The other two timers run at 1MHz.  */

    s->timer[1] = arm_timer_init(1000000, pic[irq + 1]);

    s->timer[2] = arm_timer_init(1000000, pic[irq + 2]);



    iomemtype = cpu_register_io_memory(0, icp_pit_readfn,

                                       icp_pit_writefn, s);

    cpu_register_physical_memory(base, 0x00000fff, iomemtype);

    /* ??? Save/restore.  */

}
