static void put_ebml_uint(ByteIOContext *pb, unsigned int elementid, uint64_t val)

{

    int i, bytes = 1;

    while (val >> bytes*8 && bytes < 8) bytes++;



    put_ebml_id(pb, elementid);

    put_ebml_num(pb, bytes, 0);

    for (i = bytes - 1; i >= 0; i--)

        put_byte(pb, val >> i*8);

}
