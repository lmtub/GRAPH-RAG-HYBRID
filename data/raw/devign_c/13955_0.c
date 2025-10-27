static int dma_buf_rw(BMDMAState *bm, int is_write)

{

    IDEState *s = bmdma_active_if(bm);

    struct {

        uint32_t addr;

        uint32_t size;

    } prd;

    int l, len;



    for(;;) {

        l = s->io_buffer_size - s->io_buffer_index;

        if (l <= 0)

            break;

        if (bm->cur_prd_len == 0) {

            /* end of table (with a fail safe of one page) */

            if (bm->cur_prd_last ||

                (bm->cur_addr - bm->addr) >= 4096)

                return 0;

            cpu_physical_memory_read(bm->cur_addr, (uint8_t *)&prd, 8);

            bm->cur_addr += 8;

            prd.addr = le32_to_cpu(prd.addr);

            prd.size = le32_to_cpu(prd.size);

            len = prd.size & 0xfffe;

            if (len == 0)

                len = 0x10000;

            bm->cur_prd_len = len;

            bm->cur_prd_addr = prd.addr;

            bm->cur_prd_last = (prd.size & 0x80000000);

        }

        if (l > bm->cur_prd_len)

            l = bm->cur_prd_len;

        if (l > 0) {

            if (is_write) {

                cpu_physical_memory_write(bm->cur_prd_addr,

                                          s->io_buffer + s->io_buffer_index, l);

            } else {

                cpu_physical_memory_read(bm->cur_prd_addr,

                                          s->io_buffer + s->io_buffer_index, l);

            }

            bm->cur_prd_addr += l;

            bm->cur_prd_len -= l;

            s->io_buffer_index += l;

        }

    }

    return 1;

}
