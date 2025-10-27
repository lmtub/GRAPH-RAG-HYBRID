static void pmac_dma_read(BlockBackend *blk,

                          int64_t offset, unsigned int bytes,

                          void (*cb)(void *opaque, int ret), void *opaque)

{

    DBDMA_io *io = opaque;

    MACIOIDEState *m = io->opaque;

    IDEState *s = idebus_active_if(&m->bus);

    dma_addr_t dma_addr, dma_len;

    void *mem;

    int64_t sector_num;

    int nsector;

    uint64_t align = BDRV_SECTOR_SIZE;

    size_t head_bytes, tail_bytes;



    qemu_iovec_destroy(&io->iov);

    qemu_iovec_init(&io->iov, io->len / MACIO_PAGE_SIZE + 1);



    sector_num = (offset >> 9);

    nsector = (io->len >> 9);



    MACIO_DPRINTF("--- DMA read transfer (0x%" HWADDR_PRIx ",0x%x): "

                  "sector_num: %" PRId64 ", nsector: %d\n", io->addr, io->len,

                  sector_num, nsector);



    dma_addr = io->addr;

    dma_len = io->len;

    mem = dma_memory_map(&address_space_memory, dma_addr, &dma_len,

                         DMA_DIRECTION_FROM_DEVICE);



    if (offset & (align - 1)) {

        head_bytes = offset & (align - 1);



        MACIO_DPRINTF("--- DMA unaligned head: sector %" PRId64 ", "

                      "discarding %zu bytes\n", sector_num, head_bytes);



        qemu_iovec_add(&io->iov, &io->remainder, head_bytes);



        bytes += offset & (align - 1);

        offset = offset & ~(align - 1);

    }



    qemu_iovec_add(&io->iov, mem, io->len);



    if ((offset + bytes) & (align - 1)) {

        tail_bytes = (offset + bytes) & (align - 1);



        MACIO_DPRINTF("--- DMA unaligned tail: sector %" PRId64 ", "

                      "discarding bytes %zu\n", sector_num, tail_bytes);



        qemu_iovec_add(&io->iov, &io->remainder, align - tail_bytes);

        bytes = ROUND_UP(bytes, align);

    }



    s->io_buffer_size -= io->len;

    s->io_buffer_index += io->len;



    io->len = 0;



    MACIO_DPRINTF("--- Block read transfer - sector_num: %" PRIx64 "  "

                  "nsector: %x\n", (offset >> 9), (bytes >> 9));



    m->aiocb = blk_aio_readv(blk, (offset >> 9), &io->iov, (bytes >> 9),

                             cb, io);

}
