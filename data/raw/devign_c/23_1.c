static void dma_blk_cb(void *opaque, int ret)

{

    DMAAIOCB *dbs = (DMAAIOCB *)opaque;

    dma_addr_t cur_addr, cur_len;

    void *mem;



    trace_dma_blk_cb(dbs, ret);



    dbs->acb = NULL;

    dbs->sector_num += dbs->iov.size / 512;



    if (dbs->sg_cur_index == dbs->sg->nsg || ret < 0) {

        dma_complete(dbs, ret);

        return;

    }

    dma_blk_unmap(dbs);



    while (dbs->sg_cur_index < dbs->sg->nsg) {

        cur_addr = dbs->sg->sg[dbs->sg_cur_index].base + dbs->sg_cur_byte;

        cur_len = dbs->sg->sg[dbs->sg_cur_index].len - dbs->sg_cur_byte;

        mem = dma_memory_map(dbs->sg->as, cur_addr, &cur_len, dbs->dir);

        if (!mem)

            break;

        qemu_iovec_add(&dbs->iov, mem, cur_len);

        dbs->sg_cur_byte += cur_len;

        if (dbs->sg_cur_byte == dbs->sg->sg[dbs->sg_cur_index].len) {

            dbs->sg_cur_byte = 0;

            ++dbs->sg_cur_index;

        }

    }



    if (dbs->iov.size == 0) {

        trace_dma_map_wait(dbs);

        cpu_register_map_client(dbs, continue_after_map_failure);

        return;

    }



    if (dbs->iov.size & ~BDRV_SECTOR_MASK) {

        qemu_iovec_discard_back(&dbs->iov, dbs->iov.size & ~BDRV_SECTOR_MASK);

    }



    dbs->acb = dbs->io_func(dbs->blk, dbs->sector_num, &dbs->iov,

                            dbs->iov.size / 512, dma_blk_cb, dbs);

    assert(dbs->acb);

}
