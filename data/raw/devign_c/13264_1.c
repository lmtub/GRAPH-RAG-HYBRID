int ff_mov_read_chan(AVFormatContext *s, AVStream *st, int64_t size)

{

    AVIOContext *pb = s->pb;

    uint32_t layout_tag, bitmap, num_descr, label_mask;

    int i;



    if (size < 12)

        return AVERROR_INVALIDDATA;



    layout_tag = avio_rb32(pb);

    bitmap     = avio_rb32(pb);

    num_descr  = avio_rb32(pb);



    av_dlog(s, "chan: layout=%u bitmap=%u num_descr=%u\n",

            layout_tag, bitmap, num_descr);



    if (size < 12ULL + num_descr * 20ULL)

        return 0;



    label_mask = 0;

    for (i = 0; i < num_descr; i++) {

        uint32_t label;

        label     = avio_rb32(pb);          // mChannelLabel

        avio_rb32(pb);                      // mChannelFlags

        avio_rl32(pb);                      // mCoordinates[0]

        avio_rl32(pb);                      // mCoordinates[1]

        avio_rl32(pb);                      // mCoordinates[2]

        if (layout_tag == 0) {

            uint32_t mask_incr = mov_get_channel_label(label);

            if (mask_incr == 0) {

                label_mask = 0;

                break;

            }

            label_mask |= mask_incr;

        }

    }

    if (layout_tag == 0)

            st->codec->channel_layout = label_mask;

    else

        st->codec->channel_layout = ff_mov_get_channel_layout(layout_tag, bitmap);



    return 0;

}
