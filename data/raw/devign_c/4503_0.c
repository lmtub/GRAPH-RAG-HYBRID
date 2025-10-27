static ssize_t test_block_write_func(QCryptoBlock *block,

                                     size_t offset,

                                     const uint8_t *buf,

                                     size_t buflen,

                                     Error **errp,

                                     void *opaque)

{

    Buffer *header = opaque;



    g_assert_cmpint(buflen + offset, <=, header->capacity);



    memcpy(header->buffer + offset, buf, buflen);

    header->offset = offset + buflen;



    return buflen;

}
