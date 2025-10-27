static void pl061_save(QEMUFile *f, void *opaque)

{

    pl061_state *s = (pl061_state *)opaque;



    qemu_put_be32(f, s->locked);

    qemu_put_be32(f, s->data);

    qemu_put_be32(f, s->old_data);

    qemu_put_be32(f, s->dir);

    qemu_put_be32(f, s->isense);

    qemu_put_be32(f, s->ibe);

    qemu_put_be32(f, s->iev);

    qemu_put_be32(f, s->im);

    qemu_put_be32(f, s->istate);

    qemu_put_be32(f, s->afsel);

    qemu_put_be32(f, s->dr2r);

    qemu_put_be32(f, s->dr4r);

    qemu_put_be32(f, s->dr8r);

    qemu_put_be32(f, s->odr);

    qemu_put_be32(f, s->pur);

    qemu_put_be32(f, s->pdr);

    qemu_put_be32(f, s->slr);

    qemu_put_be32(f, s->den);

    qemu_put_be32(f, s->cr);

    qemu_put_be32(f, s->float_high);

}
