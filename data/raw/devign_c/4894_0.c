static void pmac_ide_transfer_cb(void *opaque, int ret)

{

    DBDMA_io *io = opaque;

    MACIOIDEState *m = io->opaque;

    IDEState *s = idebus_active_if(&m->bus);

    int n = 0;

    int64_t sector_num;

    int unaligned;



    if (ret < 0) {

        MACIO_DPRINTF("DMA error\n");

        m->aiocb = NULL;

        qemu_sglist_destroy(&s->sg);

        ide_dma_error(s);

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



    sector_num = ide_get_sector(s);

    MACIO_DPRINTF("io_buffer_size = %#x\n", s->io_buffer_size);

    if (s->io_buffer_size > 0) {

        m->aiocb = NULL;

        qemu_sglist_destroy(&s->sg);

        n = (s->io_buffer_size + 0x1ff) >> 9;

        sector_num += n;

        ide_set_sector(s, sector_num);

        s->nsector -= n;

    }



    MACIO_DPRINTF("remainder: %d io->len: %d nsector: %d "

                  "sector_num: %" PRId64 "\n",

                  io->remainder_len, io->len, s->nsector, sector_num);

    if (io->remainder_len && io->len) {

        /* guest wants the rest of its previous transfer */

        int remainder_len = MIN(io->remainder_len, io->len);

        uint8_t *p = &io->remainder[0x200 - remainder_len];



        MACIO_DPRINTF("copying remainder %d bytes at %#" HWADDR_PRIx "\n",

                      remainder_len, io->addr);



        switch (s->dma_cmd) {

        case IDE_DMA_READ:

            cpu_physical_memory_write(io->addr, p, remainder_len);

            break;

        case IDE_DMA_WRITE:

            cpu_physical_memory_read(io->addr, p, remainder_len);

            bdrv_write(s->bs, sector_num - 1, io->remainder, 1);

            break;

        case IDE_DMA_TRIM:

            break;

        }

        io->addr += remainder_len;

        io->len -= remainder_len;

        io->remainder_len -= remainder_len;

    }



    if (s->nsector == 0 && !io->remainder_len) {

        MACIO_DPRINTF("end of transfer\n");

        s->status = READY_STAT | SEEK_STAT;

        ide_set_irq(s->bus);

        m->dma_active = false;

    }



    if (io->len == 0) {

        MACIO_DPRINTF("end of DMA\n");

        goto done;

    }



    /* launch next transfer */



    s->io_buffer_index = 0;

    s->io_buffer_size = MIN(io->len, s->nsector * 512);



    /* handle unaligned accesses first, get them over with and only do the

       remaining bulk transfer using our async DMA helpers */

    unaligned = io->len & 0x1ff;

    if (unaligned) {

        int nsector = io->len >> 9;



        MACIO_DPRINTF("precopying unaligned %d bytes to %#" HWADDR_PRIx "\n",

                      unaligned, io->addr + io->len - unaligned);



        switch (s->dma_cmd) {

        case IDE_DMA_READ:

            bdrv_read(s->bs, sector_num + nsector, io->remainder, 1);

            cpu_physical_memory_write(io->addr + io->len - unaligned,

                                      io->remainder, unaligned);

            break;

        case IDE_DMA_WRITE:

            /* cache the contents in our io struct */

            cpu_physical_memory_read(io->addr + io->len - unaligned,

                                     io->remainder, unaligned);

            break;

        case IDE_DMA_TRIM:

            break;

        }



        io->len -= unaligned;

    }



    MACIO_DPRINTF("io->len = %#x\n", io->len);



    qemu_sglist_init(&s->sg, DEVICE(m), io->len / MACIO_PAGE_SIZE + 1,

                     &address_space_memory);

    qemu_sglist_add(&s->sg, io->addr, io->len);

    io->addr += io->len + unaligned;

    io->remainder_len = (0x200 - unaligned) & 0x1ff;

    MACIO_DPRINTF("set remainder to: %d\n", io->remainder_len);



    /* We would read no data from the block layer, thus not get a callback.

       Just fake completion manually. */

    if (!io->len) {

        pmac_ide_transfer_cb(opaque, 0);

        return;

    }



    io->len = 0;



    MACIO_DPRINTF("sector_num=%" PRId64 " n=%d, nsector=%d, cmd_cmd=%d\n",

                  sector_num, n, s->nsector, s->dma_cmd);



    switch (s->dma_cmd) {

    case IDE_DMA_READ:

        m->aiocb = dma_bdrv_read(s->bs, &s->sg, sector_num,

                                 pmac_ide_transfer_cb, io);

        break;

    case IDE_DMA_WRITE:

        m->aiocb = dma_bdrv_write(s->bs, &s->sg, sector_num,

                                  pmac_ide_transfer_cb, io);

        break;

    case IDE_DMA_TRIM:

        m->aiocb = dma_bdrv_io(s->bs, &s->sg, sector_num,

                               ide_issue_trim, pmac_ide_transfer_cb, io,

                               DMA_DIRECTION_TO_DEVICE);

        break;

    }

    return;



done:

    if (s->dma_cmd == IDE_DMA_READ || s->dma_cmd == IDE_DMA_WRITE) {

        bdrv_acct_done(s->bs, &s->acct);

    }

    io->dma_end(io);

}
