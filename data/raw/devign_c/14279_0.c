static void virtio_set_status(struct subchannel_id schid,

                              unsigned long dev_addr)

{

    unsigned char status = dev_addr;

    run_ccw(schid, CCW_CMD_WRITE_STATUS, &status, sizeof(status));

}
