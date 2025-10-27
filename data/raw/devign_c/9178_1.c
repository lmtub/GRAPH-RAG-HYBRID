int avpriv_dca_convert_bitstream(const uint8_t *src, int src_size, uint8_t *dst,

                             int max_size)

{

    uint32_t mrk;

    int i, tmp;

    const uint16_t *ssrc = (const uint16_t *) src;

    uint16_t *sdst = (uint16_t *) dst;

    PutBitContext pb;



    if ((unsigned) src_size > (unsigned) max_size)

        src_size = max_size;



    mrk = AV_RB32(src);

    switch (mrk) {

    case DCA_SYNCWORD_CORE_BE:

        memcpy(dst, src, src_size);

        return src_size;

    case DCA_SYNCWORD_CORE_LE:

        for (i = 0; i < (src_size + 1) >> 1; i++)

            *sdst++ = av_bswap16(*ssrc++);

        return src_size;

    case DCA_SYNCWORD_CORE_14B_BE:

    case DCA_SYNCWORD_CORE_14B_LE:

        init_put_bits(&pb, dst, max_size);

        for (i = 0; i < (src_size + 1) >> 1; i++, src += 2) {

            tmp = ((mrk == DCA_SYNCWORD_CORE_14B_BE) ? AV_RB16(src) : AV_RL16(src)) & 0x3FFF;

            put_bits(&pb, 14, tmp);

        }

        flush_put_bits(&pb);

        return (put_bits_count(&pb) + 7) >> 3;

    default:

        return AVERROR_INVALIDDATA;

    }

}
