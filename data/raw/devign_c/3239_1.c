static void decode_channel_map(uint8_t layout_map[][3],

                               enum ChannelPosition type,

                               GetBitContext *gb, int n)

{

    while (n--) {

        enum RawDataBlockType syn_ele;

        switch (type) {

        case AAC_CHANNEL_FRONT:

        case AAC_CHANNEL_BACK:

        case AAC_CHANNEL_SIDE:

            syn_ele = get_bits1(gb);

            break;

        case AAC_CHANNEL_CC:

            skip_bits1(gb);

            syn_ele = TYPE_CCE;

            break;

        case AAC_CHANNEL_LFE:

            syn_ele = TYPE_LFE;

            break;



        }

        layout_map[0][0] = syn_ele;

        layout_map[0][1] = get_bits(gb, 4);

        layout_map[0][2] = type;

        layout_map++;

    }

}