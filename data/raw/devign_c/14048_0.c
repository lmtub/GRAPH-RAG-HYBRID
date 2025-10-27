static int convert_bitstream(const uint8_t *src, int src_size, uint8_t *dst, int max_size)

{

    switch (AV_RB32(src)) {

    case DCA_SYNCWORD_CORE_BE:

    case DCA_SYNCWORD_SUBSTREAM:

        memcpy(dst, src, src_size);

        return src_size;

    case DCA_SYNCWORD_CORE_LE:

    case DCA_SYNCWORD_CORE_14B_BE:

    case DCA_SYNCWORD_CORE_14B_LE:

        return avpriv_dca_convert_bitstream(src, src_size, dst, max_size);

    default:

        return AVERROR_INVALIDDATA;

    }

}
