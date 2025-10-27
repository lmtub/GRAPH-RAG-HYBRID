static int bytes_left(ByteIOContext *bc)

{

    return bc->buf_end - bc->buf_ptr;

}
