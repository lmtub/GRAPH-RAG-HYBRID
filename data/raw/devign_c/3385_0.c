static unsigned tget_long(GetByteContext *gb, int le)

{

    unsigned v = le ? bytestream2_get_le32u(gb) : bytestream2_get_be32u(gb);

    return v;

}
