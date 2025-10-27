void pl181_init(uint32_t base, BlockDriverState *bd,

                qemu_irq irq0, qemu_irq irq1)

{

    int iomemtype;

    pl181_state *s;



    s = (pl181_state *)qemu_mallocz(sizeof(pl181_state));

    iomemtype = cpu_register_io_memory(0, pl181_readfn,

                                       pl181_writefn, s);

    cpu_register_physical_memory(base, 0x00000fff, iomemtype);

    s->base = base;

    s->card = sd_init(bd);

    s->irq[0] = irq0;

    s->irq[1] = irq1;

    qemu_register_reset(pl181_reset, s);

    pl181_reset(s);

    /* ??? Save/restore.  */

}
