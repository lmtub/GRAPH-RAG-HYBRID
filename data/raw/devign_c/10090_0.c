static ssize_t test_block_init_func(QCryptoBlock *block,

                                    size_t headerlen,

                                    Error **errp,

                                    void *opaque)

{

    Buffer *header = opaque;



    g_assert_cmpint(header->capacity, ==, 0);



    buffer_reserve(header, headerlen);



    return headerlen;

}
