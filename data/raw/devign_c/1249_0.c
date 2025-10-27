static void ide_flush_cb(void *opaque, int ret)

{

    IDEState *s = opaque;



    s->pio_aiocb = NULL;



    if (ret == -ECANCELED) {

        return;

    }

    if (ret < 0) {

        /* XXX: What sector number to set here? */

        if (ide_handle_rw_error(s, -ret, IDE_RETRY_FLUSH)) {

            return;

        }

    }



    if (s->bs) {

        block_acct_done(bdrv_get_stats(s->bs), &s->acct);

    }

    s->status = READY_STAT | SEEK_STAT;

    ide_cmd_done(s);

    ide_set_irq(s->bus);

}
