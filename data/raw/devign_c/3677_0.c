static int copy_stream_props(AVStream *st, AVStream *source_st)

{

    int ret;



    if (st->codecpar->codec_id || !source_st->codecpar->codec_id) {

        if (st->codecpar->extradata_size < source_st->codecpar->extradata_size) {

            if (st->codecpar->extradata) {

                av_freep(&st->codecpar->extradata);

                st->codecpar->extradata_size = 0;

            }

            ret = ff_alloc_extradata(st->codecpar,

                                     source_st->codecpar->extradata_size);

            if (ret < 0)

                return ret;

        }

        memcpy(st->codecpar->extradata, source_st->codecpar->extradata,

               source_st->codecpar->extradata_size);

        return 0;

    }

    if ((ret = avcodec_parameters_copy(st->codecpar, source_st->codecpar)) < 0)

        return ret;

    st->r_frame_rate        = source_st->r_frame_rate;

    st->avg_frame_rate      = source_st->avg_frame_rate;

    st->time_base           = source_st->time_base;

    st->sample_aspect_ratio = source_st->sample_aspect_ratio;



    av_dict_copy(&st->metadata, source_st->metadata, 0);

    return 0;

}
