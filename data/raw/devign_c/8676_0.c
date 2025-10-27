static int ac3_parse_sync_info(AC3DecodeContext *ctx)

{

    ac3_sync_info *sync_info = &ctx->sync_info;

    GetBitContext *gb = &ctx->gb;



    sync_info->sync_word = get_bits(gb, 16);

    sync_info->crc1 = get_bits(gb, 16);

    sync_info->fscod = get_bits(gb, 2);

    if (sync_info->fscod == 0x03)

        return -1;

    sync_info->frmsizecod = get_bits(gb, 6);

    if (sync_info->frmsizecod >= 0x38)

        return -1;

    sync_info->sampling_rate = ac3_freqs[sync_info->fscod];

    sync_info->bit_rate = ac3_bitratetab[sync_info->frmsizecod >> 1];



    return 0;

}
