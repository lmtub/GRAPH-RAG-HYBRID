static void virtio_set_status(struct subchannel_id schid,

                              unsigned long dev_addr)

{

    unsigned char status = dev_addr;

    if (run_ccw(schid, CCW_CMD_WRITE_STATUS, &status, sizeof(status))) {

        virtio_panic("Could not write status to host!\n");

    }

}
