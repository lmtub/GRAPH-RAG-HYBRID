int pcm_read_seek(AVFormatContext *s,

                  int stream_index, int64_t timestamp, int flags)

{

    AVStream *st;

    int block_align, byte_rate;

    int64_t pos;



    st = s->streams[0];



    block_align = st->codec->block_align ? st->codec->block_align :

        (av_get_bits_per_sample(st->codec->codec_id) * st->codec->channels) >> 3;

    byte_rate = st->codec->bit_rate ? st->codec->bit_rate >> 3 :

        block_align * st->codec->sample_rate;



    if (block_align <= 0 || byte_rate <= 0)

        return -1;



    /* compute the position by aligning it to block_align */

    pos = av_rescale_rnd(timestamp * byte_rate,

                         st->time_base.num,

                         st->time_base.den * (int64_t)block_align,

                         (flags & AVSEEK_FLAG_BACKWARD) ? AV_ROUND_DOWN : AV_ROUND_UP);

    pos *= block_align;



    /* recompute exact position */

    st->cur_dts = av_rescale(pos, st->time_base.den, byte_rate * (int64_t)st->time_base.num);

    url_fseek(s->pb, pos + s->data_offset, SEEK_SET);

    return 0;

}
