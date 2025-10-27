static void ncq_cb(void *opaque, int ret)

{

    NCQTransferState *ncq_tfs = (NCQTransferState *)opaque;

    IDEState *ide_state = &ncq_tfs->drive->port.ifs[0];



    if (ret == -ECANCELED) {

        return;

    }

    /* Clear bit for this tag in SActive */

    ncq_tfs->drive->port_regs.scr_act &= ~(1 << ncq_tfs->tag);



    if (ret < 0) {

        /* error */

        ide_state->error = ABRT_ERR;

        ide_state->status = READY_STAT | ERR_STAT;

        ncq_tfs->drive->port_regs.scr_err |= (1 << ncq_tfs->tag);

    } else {

        ide_state->status = READY_STAT | SEEK_STAT;

    }



    ahci_write_fis_sdb(ncq_tfs->drive->hba, ncq_tfs->drive->port_no,

                       (1 << ncq_tfs->tag));



    DPRINTF(ncq_tfs->drive->port_no, "NCQ transfer tag %d finished\n",

            ncq_tfs->tag);



    block_acct_done(bdrv_get_stats(ncq_tfs->drive->port.ifs[0].bs),

                    &ncq_tfs->acct);

    qemu_sglist_destroy(&ncq_tfs->sglist);

    ncq_tfs->used = 0;

}
