static int dca_convert_bitstream(uint8_t * src, int src_size, uint8_t * dst,
                          int max_size)
{
    uint32_t mrk;
    int i, tmp;
    uint16_t *ssrc = (uint16_t *) src, *sdst = (uint16_t *) dst;
    PutBitContext pb;
    mrk = AV_RB32(src);
    switch (mrk) {
    case DCA_MARKER_RAW_BE:
        memcpy(dst, src, FFMIN(src_size, max_size));
        return FFMIN(src_size, max_size);
    case DCA_MARKER_RAW_LE:
        for (i = 0; i < (FFMIN(src_size, max_size) + 1) >> 1; i++)
            *sdst++ = bswap_16(*ssrc++);
        return FFMIN(src_size, max_size);
    case DCA_MARKER_14B_BE:
    case DCA_MARKER_14B_LE:
        init_put_bits(&pb, dst, max_size);
        for (i = 0; i < (src_size + 1) >> 1; i++, src += 2) {
            tmp = ((mrk == DCA_MARKER_14B_BE) ? AV_RB16(src) : AV_RL16(src)) & 0x3FFF;
            put_bits(&pb, 14, tmp);
        }
        flush_put_bits(&pb);
        return (put_bits_count(&pb) + 7) >> 3;
    default:
    }
}