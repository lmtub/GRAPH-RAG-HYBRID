static int ipvideo_decode_block_opcode_0xD(IpvideoContext *s, AVFrame *frame)
{
    int y;
    unsigned char P[2];
    /* 4-color block encoding: each 4x4 block is a different color */
    for (y = 0; y < 8; y++) {
        if (!(y & 3)) {
            P[0] = bytestream2_get_byte(&s->stream_ptr);
            P[1] = bytestream2_get_byte(&s->stream_ptr);
        memset(s->pixel_ptr,     P[0], 4);
        memset(s->pixel_ptr + 4, P[1], 4);
        s->pixel_ptr += s->stride;
    /* report success */
    return 0;