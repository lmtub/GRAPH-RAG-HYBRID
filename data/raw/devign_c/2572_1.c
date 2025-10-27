static void mm_decode_inter(MmContext * s, int half_horiz, int half_vert, const uint8_t *buf, int buf_size)
{
    const int data_ptr = 2 + AV_RL16(&buf[0]);
    int d, r, y;
    d = data_ptr; r = 2; y = 0;
    while(r < data_ptr) {
        int i, j;
        int length = buf[r] & 0x7f;
        int x = buf[r+1] + ((buf[r] & 0x80) << 1);
        r += 2;
        if (length==0) {
            y += x;
            continue;
        }
        for(i=0; i<length; i++) {
            for(j=0; j<8; j++) {
                int replace = (buf[r+i] >> (7-j)) & 1;
                if (replace) {
                    int color = buf[d];
                    s->frame.data[0][y*s->frame.linesize[0] + x] = color;
                    if (half_horiz)
                        s->frame.data[0][y*s->frame.linesize[0] + x + 1] = color;
                    if (half_vert) {
                        s->frame.data[0][(y+1)*s->frame.linesize[0] + x] = color;
                        if (half_horiz)
                            s->frame.data[0][(y+1)*s->frame.linesize[0] + x + 1] = color;
                    }
                    d++;
                }
                x += 1 + half_horiz;
            }
        }
        r += length;
        y += 1 + half_vert;
    }
}