static int ipvideo_decode_block_opcode_0xA(IpvideoContext *s, AVFrame *frame)
{
    int x, y;
    unsigned char P[8];
    int flags = 0;
    bytestream2_get_buffer(&s->stream_ptr, P, 4);
    /* 4-color encoding for each 4x4 quadrant, or 4-color encoding on
     * either top and bottom or left and right halves */
    if (P[0] <= P[1]) {
        /* 4-color encoding for each quadrant; need 32 bytes */
        for (y = 0; y < 16; y++) {
            // new values for each 4x4 block
            if (!(y & 3)) {
                if (y) bytestream2_get_buffer(&s->stream_ptr, P, 4);
                flags = bytestream2_get_le32(&s->stream_ptr);
            for (x = 0; x < 4; x++, flags >>= 2)
                *s->pixel_ptr++ = P[flags & 0x03];
            s->pixel_ptr += s->stride - 4;
            // switch to right half
            if (y == 7) s->pixel_ptr -= 8 * s->stride - 4;
    } else {
        // vertical split?
        int vert;
        uint64_t flags = bytestream2_get_le64(&s->stream_ptr);
        bytestream2_get_buffer(&s->stream_ptr, P + 4, 4);
        vert = P[4] <= P[5];
        /* 4-color encoding for either left and right or top and bottom
         * halves */
        for (y = 0; y < 16; y++) {
            for (x = 0; x < 4; x++, flags >>= 2)
                *s->pixel_ptr++ = P[flags & 0x03];
            if (vert) {
                s->pixel_ptr += s->stride - 4;
                // switch to right half
                if (y == 7) s->pixel_ptr -= 8 * s->stride - 4;
            } else if (y & 1) s->pixel_ptr += s->line_inc;
            // load values for second half
            if (y == 7) {
                memcpy(P, P + 4, 4);
                flags = bytestream2_get_le64(&s->stream_ptr);
    /* report success */
    return 0;