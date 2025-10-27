static void pmac_ide_atapi_transfer_cb(void *opaque, int ret)

{

    DBDMA_io *io = opaque;

    MACIOIDEState *m = io->opaque;

    IDEState *s = idebus_active_if(&m->bus);

    int unaligned;



    if (ret < 0) {

        m->aiocb = NULL;

        qemu_sglist_destroy(&s->sg);

        ide_atapi_io_error(s, ret);

        io->remainder_len = 0;

        goto done;

    }



    if (!m->dma_active) {

        MACIO_DPRINTF("waiting for data (%#x - %#x - %x)\n",

                      s->nsector, io->len, s->status);

        /* data not ready yet, wait for the channel to get restarted */

        io->processing = false;

        return;

    }



    MACIO_DPRINTF("io_buffer_size = %#x\n", s->io_buffer_size);



    if (s->io_buffer_size > 0) {

        m->aiocb = NULL;

        qemu_sglist_destroy(&s->sg);



        s->packet_transfer_size -= s->io_buffer_size;



        s->io_buffer_index += s->io_buffer_size;

        s->lba += s->io_buffer_index >> 11;

        s->io_buffer_index &= 0x7ff;

    }



    s->io_buffer_size = MIN(io->len, s->packet_transfer_size);



    MACIO_DPRINTF("remainder: %d io->len: %d size: %d\n", io->remainder_len,

                  io->len, s->packet_transfer_size);

    if (io->remainder_len && io->len) {

        /* guest wants the rest of its previous transfer */

        int remainder_len = MIN(io->remainder_len, io->len);



        MACIO_DPRINTF("copying remainder %d bytes\n", remainder_len);



        cpu_physical_memory_write(io->addr, io->remainder + 0x200 -

                                  remainder_len, remainder_len);



        io->addr += remainder_len;

        io->len -= remainder_len;

        s->io_buffer_size = remainder_len;

        io->remainder_len -= remainder_len;

        /* treat remainder as individual transfer, start again */

        qemu_sglist_init(&s->sg, DEVICE(m), io->len / MACIO_PAGE_SIZE + 1,

                         &address_space_memory);

        pmac_ide_atapi_transfer_cb(opaque, 0);

        return;

    }



    if (!s->packet_transfer_size) {

        MACIO_DPRINTF("end of transfer\n");

        ide_atapi_cmd_ok(s);

        m->dma_active = false;

    }



    if (io->len == 0) {

        MACIO_DPRINTF("end of DMA\n");

        goto done;

    }



    /* launch next transfer */



    /* handle unaligned accesses first, get them over with and only do the

       remaining bulk transfer using our async DMA helpers */

    unaligned = io->len & 0x1ff;

    if (unaligned) {

        int sector_num = (s->lba << 2) + (s->io_buffer_index >> 9);

        int nsector = io->len >> 9;



        MACIO_DPRINTF("precopying unaligned %d bytes to %#" HWADDR_PRIx "\n",

                      unaligned, io->addr + io->len - unaligned);



        bdrv_read(s->bs, sector_num + nsector, io->remainder, 1);

        cpu_physical_memory_write(io->addr + io->len - unaligned,

                                  io->remainder, unaligned);



        io->len -= unaligned;

    }



    MACIO_DPRINTF("io->len = %#x\n", io->len);



    qemu_sglist_init(&s->sg, DEVICE(m), io->len / MACIO_PAGE_SIZE + 1,

                     &address_space_memory);

    qemu_sglist_add(&s->sg, io->addr, io->len);

    io->addr += s->io_buffer_size;

    io->remainder_len = MIN(s->packet_transfer_size - s->io_buffer_size,

                            (0x200 - unaligned) & 0x1ff);

    MACIO_DPRINTF("set remainder to: %d\n", io->remainder_len);



    /* We would read no data from the block layer, thus not get a callback.

       Just fake completion manually. */

    if (!io->len) {

        pmac_ide_atapi_transfer_cb(opaque, 0);

        return;

    }



    io->len = 0;



    MACIO_DPRINTF("sector_num=%d size=%d, cmd_cmd=%d\n",

                  (s->lba << 2) + (s->io_buffer_index >> 9),

                  s->packet_transfer_size, s->dma_cmd);



    m->aiocb = dma_bdrv_read(s->bs, &s->sg,

                             (int64_t)(s->lba << 2) + (s->io_buffer_index >> 9),

                             pmac_ide_atapi_transfer_cb, io);

    return;



done:

    MACIO_DPRINTF("done DMA\n");

    block_acct_done(bdrv_get_stats(s->bs), &s->acct);

    io->dma_end(opaque);

}
