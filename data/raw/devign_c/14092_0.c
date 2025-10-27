static ssize_t test_block_read_func(QCryptoBlock *block,

                                    void *opaque,

                                    size_t offset,

                                    uint8_t *buf,

                                    size_t buflen,

                                    Error **errp)

{

    Buffer *header = opaque;



    g_assert_cmpint(offset + buflen, <=, header->capacity);



    memcpy(buf, header->buffer + offset, buflen);



    return buflen;

}
