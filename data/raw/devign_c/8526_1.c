static void stellaris_enet_save(QEMUFile *f, void *opaque)

{

    stellaris_enet_state *s = (stellaris_enet_state *)opaque;

    int i;



    qemu_put_be32(f, s->ris);

    qemu_put_be32(f, s->im);

    qemu_put_be32(f, s->rctl);

    qemu_put_be32(f, s->tctl);

    qemu_put_be32(f, s->thr);

    qemu_put_be32(f, s->mctl);

    qemu_put_be32(f, s->mdv);

    qemu_put_be32(f, s->mtxd);

    qemu_put_be32(f, s->mrxd);

    qemu_put_be32(f, s->np);

    qemu_put_be32(f, s->tx_fifo_len);

    qemu_put_buffer(f, s->tx_fifo, sizeof(s->tx_fifo));

    for (i = 0; i < 31; i++) {

        qemu_put_be32(f, s->rx[i].len);

        qemu_put_buffer(f, s->rx[i].data, sizeof(s->rx[i].data));



    }

    qemu_put_be32(f, s->next_packet);

    qemu_put_be32(f, s->rx_fifo_offset);

}
