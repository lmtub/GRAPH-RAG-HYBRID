static int vhost_user_read(struct vhost_dev *dev, VhostUserMsg *msg)

{

    CharDriverState *chr = dev->opaque;

    uint8_t *p = (uint8_t *) msg;

    int r, size = VHOST_USER_HDR_SIZE;



    r = qemu_chr_fe_read_all(chr, p, size);

    if (r != size) {

        error_report("Failed to read msg header. Read %d instead of %d.", r,

                size);

        goto fail;

    }



    /* validate received flags */

    if (msg->flags != (VHOST_USER_REPLY_MASK | VHOST_USER_VERSION)) {

        error_report("Failed to read msg header."

                " Flags 0x%x instead of 0x%x.", msg->flags,

                VHOST_USER_REPLY_MASK | VHOST_USER_VERSION);

        goto fail;

    }



    /* validate message size is sane */

    if (msg->size > VHOST_USER_PAYLOAD_SIZE) {

        error_report("Failed to read msg header."

                " Size %d exceeds the maximum %zu.", msg->size,

                VHOST_USER_PAYLOAD_SIZE);

        goto fail;

    }



    if (msg->size) {

        p += VHOST_USER_HDR_SIZE;

        size = msg->size;

        r = qemu_chr_fe_read_all(chr, p, size);

        if (r != size) {

            error_report("Failed to read msg payload."

                         " Read %d instead of %d.", r, msg->size);

            goto fail;

        }

    }



    return 0;



fail:

    return -1;

}
