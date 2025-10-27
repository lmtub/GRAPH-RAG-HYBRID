static ssize_t test_block_init_func(QCryptoBlock *block,

                                    void *opaque,

                                    size_t headerlen,

                                    Error **errp)

{

    Buffer *header = opaque;



    g_assert_cmpint(header->capacity, ==, 0);



    buffer_reserve(header, headerlen);



    return headerlen;

}
