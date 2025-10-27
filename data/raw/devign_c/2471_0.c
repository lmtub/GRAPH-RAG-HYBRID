static void scsi_read_data(SCSIDevice *d, uint32_t tag)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, d);

    SCSIDiskReq *r;



    r = scsi_find_request(s, tag);

    if (!r) {

        BADF("Bad read tag 0x%x\n", tag);

        /* ??? This is the wrong error.  */

        scsi_command_complete(r, CHECK_CONDITION, HARDWARE_ERROR);

        return;

    }



    /* No data transfer may already be in progress */

    assert(r->req.aiocb == NULL);



    scsi_read_request(r);

}
