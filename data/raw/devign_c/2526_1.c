static void conv411(uint8_t *dst, int dst_wrap, 
                    uint8_t *src, int src_wrap,
                    int width, int height)
{
    int w, c;
    uint8_t *s1, *s2, *d;
    for(;height > 0; height--) {
        s1 = src;
        s2 = src + src_wrap;
        d = dst;
        for(w = width;w > 0; w--) {
            c = (s1[0] + s2[0]) >> 1;
            d[0] = c;
            d[1] = c;
            s1++;
            s2++;
            d += 2;
        }
        src += src_wrap * 2;
        dst += dst_wrap;
    }
}