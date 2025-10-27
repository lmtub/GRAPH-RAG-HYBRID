static int nbd_parse_offset_hole_payload(NBDStructuredReplyChunk *chunk,

                                         uint8_t *payload, uint64_t orig_offset,

                                         QEMUIOVector *qiov, Error **errp)

{

    uint64_t offset;

    uint32_t hole_size;



    if (chunk->length != sizeof(offset) + sizeof(hole_size)) {

        error_setg(errp, "Protocol error: invalid payload for "

                         "NBD_REPLY_TYPE_OFFSET_HOLE");

        return -EINVAL;

    }



    offset = payload_advance64(&payload);

    hole_size = payload_advance32(&payload);



    if (offset < orig_offset || hole_size > qiov->size ||

        offset > orig_offset + qiov->size - hole_size) {

        error_setg(errp, "Protocol error: server sent chunk exceeding requested"

                         " region");

        return -EINVAL;

    }



    qemu_iovec_memset(qiov, offset - orig_offset, 0, hole_size);



    return 0;

}
