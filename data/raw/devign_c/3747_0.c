static void pxa2xx_pic_save(QEMUFile *f, void *opaque)

{

    PXA2xxPICState *s = (PXA2xxPICState *) opaque;

    int i;



    for (i = 0; i < 2; i ++)

        qemu_put_be32s(f, &s->int_enabled[i]);

    for (i = 0; i < 2; i ++)

        qemu_put_be32s(f, &s->int_pending[i]);

    for (i = 0; i < 2; i ++)

        qemu_put_be32s(f, &s->is_fiq[i]);

    qemu_put_be32s(f, &s->int_idle);

    for (i = 0; i < PXA2XX_PIC_SRCS; i ++)

        qemu_put_be32s(f, &s->priority[i]);

}
