static int ide_write_dma_cb(IDEState *s, 

                            target_phys_addr_t phys_addr, 

                            int transfer_size1)

{

    int len, transfer_size, n;

    int64_t sector_num;



    transfer_size = transfer_size1;

    for(;;) {

        len = s->io_buffer_size - s->io_buffer_index;

        if (len == 0) {

            n = s->io_buffer_size >> 9;

            sector_num = ide_get_sector(s);

            bdrv_write(s->bs, sector_num, s->io_buffer, 

                       s->io_buffer_size >> 9);

            sector_num += n;

            ide_set_sector(s, sector_num);

            s->nsector -= n;

            n = s->nsector;

            if (n == 0) {

                /* end of transfer */

                s->status = READY_STAT | SEEK_STAT;

                ide_set_irq(s);

                return 0;

            }

            if (n > MAX_MULT_SECTORS)

                n = MAX_MULT_SECTORS;

            s->io_buffer_index = 0;

            s->io_buffer_size = n * 512;

            len = s->io_buffer_size;

        }

        if (transfer_size <= 0)

            break;

        if (len > transfer_size)

            len = transfer_size;

        cpu_physical_memory_read(phys_addr, 

                                 s->io_buffer + s->io_buffer_index, len);

        s->io_buffer_index += len;

        transfer_size -= len;

        phys_addr += len;

    }

    return transfer_size1 - transfer_size;

}
