int avpriv_dca_parse_core_frame_header(DCACoreFrameHeader *h, const uint8_t *buf, int size)

{

    GetBitContext gb;



    if (init_get_bits8(&gb, buf, size) < 0)

        return DCA_PARSE_ERROR_INVALIDDATA;



    return ff_dca_parse_core_frame_header(h, &gb);

}
