static int aac_sync(uint64_t state, AACAC3ParseContext *hdr_info,

        int *need_next_header, int *new_frame_start)

{

    GetBitContext bits;

    int size, rdb, ch, sr;

    union {

        uint64_t u64;

        uint8_t  u8[8];

    } tmp;



    tmp.u64 = be2me_64(state);

    init_get_bits(&bits, tmp.u8+8-AAC_HEADER_SIZE, AAC_HEADER_SIZE * 8);



    if(get_bits(&bits, 12) != 0xfff)

        return 0;



    skip_bits1(&bits);          /* id */

    skip_bits(&bits, 2);        /* layer */

    skip_bits1(&bits);          /* protection_absent */

    skip_bits(&bits, 2);        /* profile_objecttype */

    sr = get_bits(&bits, 4);    /* sample_frequency_index */

    if(!ff_mpeg4audio_sample_rates[sr])

        return 0;

    skip_bits1(&bits);          /* private_bit */

    ch = get_bits(&bits, 3);    /* channel_configuration */

    if(!ff_mpeg4audio_channels[ch])

        return 0;

    skip_bits1(&bits);          /* original/copy */

    skip_bits1(&bits);          /* home */



    /* adts_variable_header */

    skip_bits1(&bits);          /* copyright_identification_bit */

    skip_bits1(&bits);          /* copyright_identification_start */

    size = get_bits(&bits, 13); /* aac_frame_length */

    if(size < AAC_HEADER_SIZE)

        return 0;



    skip_bits(&bits, 11);       /* adts_buffer_fullness */

    rdb = get_bits(&bits, 2);   /* number_of_raw_data_blocks_in_frame */



    hdr_info->channels = ff_mpeg4audio_channels[ch];

    hdr_info->sample_rate = ff_mpeg4audio_sample_rates[sr];

    hdr_info->samples = (rdb + 1) * 1024;

    hdr_info->bit_rate = size * 8 * hdr_info->sample_rate / hdr_info->samples;



    *need_next_header = 0;

    *new_frame_start  = 1;

    return size;

}
