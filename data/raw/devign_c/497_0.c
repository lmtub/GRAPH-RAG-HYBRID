void ide_sector_read(IDEState *s)

{

    int64_t sector_num;

    int ret, n;



    s->status = READY_STAT | SEEK_STAT;

    s->error = 0; /* not needed by IDE spec, but needed by Windows */

    sector_num = ide_get_sector(s);

    n = s->nsector;

    if (n == 0) {

        /* no more sector to read from disk */

        ide_transfer_stop(s);

    } else {

#if defined(DEBUG_IDE)

        printf("read sector=%" PRId64 "\n", sector_num);

#endif

        if (n > s->req_nb_sectors)

            n = s->req_nb_sectors;



        bdrv_acct_start(s->bs, &s->acct, n * BDRV_SECTOR_SIZE, BDRV_ACCT_READ);

        ret = bdrv_read(s->bs, sector_num, s->io_buffer, n);

        bdrv_acct_done(s->bs, &s->acct);

        if (ret != 0) {

            if (ide_handle_rw_error(s, -ret,

                BM_STATUS_PIO_RETRY | BM_STATUS_RETRY_READ))

            {

                return;

            }

        }

        ide_transfer_start(s, s->io_buffer, 512 * n, ide_sector_read);

        ide_set_irq(s->bus);

        ide_set_sector(s, sector_num + n);

        s->nsector -= n;

    }

}
