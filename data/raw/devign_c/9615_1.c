static int ipvideo_decode_block_opcode_0x8(IpvideoContext *s, AVFrame *frame)
{
    int x, y;
    unsigned char P[4];
    unsigned int flags = 0;
    /* 2-color encoding for each 4x4 quadrant, or 2-color encoding on
     * either top and bottom or left and right halves */
    P[0] = bytestream2_get_byte(&s->stream_ptr);
    P[1] = bytestream2_get_byte(&s->stream_ptr);
    if (P[0] <= P[1]) {
        for (y = 0; y < 16; y++) {
            // new values for each 4x4 block
            if (!(y & 3)) {
                if (y) {
                    P[0]  = bytestream2_get_byte(&s->stream_ptr);
                    P[1]  = bytestream2_get_byte(&s->stream_ptr);
                flags = bytestream2_get_le16(&s->stream_ptr);
            for (x = 0; x < 4; x++, flags >>= 1)
                *s->pixel_ptr++ = P[flags & 1];
            s->pixel_ptr += s->stride - 4;
            // switch to right half
            if (y == 7) s->pixel_ptr -= 8 * s->stride - 4;
    } else {
        flags = bytestream2_get_le32(&s->stream_ptr);
        P[2] = bytestream2_get_byte(&s->stream_ptr);
        P[3] = bytestream2_get_byte(&s->stream_ptr);
        if (P[2] <= P[3]) {
            /* vertical split; left & right halves are 2-color encoded */
            for (y = 0; y < 16; y++) {
                for (x = 0; x < 4; x++, flags >>= 1)
                    *s->pixel_ptr++ = P[flags & 1];
                s->pixel_ptr += s->stride - 4;
                // switch to right half
                if (y == 7) {
                    s->pixel_ptr -= 8 * s->stride - 4;
                    P[0]  = P[2];
                    P[1]  = P[3];
                    flags = bytestream2_get_le32(&s->stream_ptr);
        } else {
            /* horizontal split; top & bottom halves are 2-color encoded */
            for (y = 0; y < 8; y++) {
                if (y == 4) {
                    P[0]  = P[2];
                    P[1]  = P[3];
                    flags = bytestream2_get_le32(&s->stream_ptr);
                for (x = 0; x < 8; x++, flags >>= 1)
                    *s->pixel_ptr++ = P[flags & 1];
                s->pixel_ptr += s->line_inc;
    /* report success */
    return 0;