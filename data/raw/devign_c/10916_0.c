static int ffm_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    FFMContext *ffm = s->priv_data;

    AVStream *st;

    ByteIOContext *pb = s->pb;

    AVCodecContext *codec;

    int i, nb_streams;

    uint32_t tag;



    /* header */

    tag = get_le32(pb);

    if (tag != MKTAG('F', 'F', 'M', '1'))

        goto fail;

    ffm->packet_size = get_be32(pb);

    if (ffm->packet_size != FFM_PACKET_SIZE)

        goto fail;

    ffm->write_index = get_be64(pb);

    /* get also filesize */

    if (!url_is_streamed(pb)) {

        ffm->file_size = url_fsize(pb);

        if (ffm->write_index)

            adjust_write_index(s);

    } else {

        ffm->file_size = (UINT64_C(1) << 63) - 1;

    }



    nb_streams = get_be32(pb);

    get_be32(pb); /* total bitrate */

    /* read each stream */

    for(i=0;i<nb_streams;i++) {

        char rc_eq_buf[128];



        st = av_new_stream(s, 0);

        if (!st)

            goto fail;



        av_set_pts_info(st, 64, 1, 1000000);



        codec = st->codec;

        /* generic info */

        codec->codec_id = get_be32(pb);

        codec->codec_type = get_byte(pb); /* codec_type */

        codec->bit_rate = get_be32(pb);

        st->quality = get_be32(pb);

        codec->flags = get_be32(pb);

        codec->flags2 = get_be32(pb);

        codec->debug = get_be32(pb);

        /* specific info */

        switch(codec->codec_type) {

        case CODEC_TYPE_VIDEO:

            codec->time_base.num = get_be32(pb);

            codec->time_base.den = get_be32(pb);

            codec->width = get_be16(pb);

            codec->height = get_be16(pb);

            codec->gop_size = get_be16(pb);

            codec->pix_fmt = get_be32(pb);

            codec->qmin = get_byte(pb);

            codec->qmax = get_byte(pb);

            codec->max_qdiff = get_byte(pb);

            codec->qcompress = get_be16(pb) / 10000.0;

            codec->qblur = get_be16(pb) / 10000.0;

            codec->bit_rate_tolerance = get_be32(pb);

            codec->rc_eq = av_strdup(get_strz(pb, rc_eq_buf, sizeof(rc_eq_buf)));

            codec->rc_max_rate = get_be32(pb);

            codec->rc_min_rate = get_be32(pb);

            codec->rc_buffer_size = get_be32(pb);

            codec->i_quant_factor = av_int2dbl(get_be64(pb));

            codec->b_quant_factor = av_int2dbl(get_be64(pb));

            codec->i_quant_offset = av_int2dbl(get_be64(pb));

            codec->b_quant_offset = av_int2dbl(get_be64(pb));

            codec->dct_algo = get_be32(pb);

            codec->strict_std_compliance = get_be32(pb);

            codec->max_b_frames = get_be32(pb);

            codec->luma_elim_threshold = get_be32(pb);

            codec->chroma_elim_threshold = get_be32(pb);

            codec->mpeg_quant = get_be32(pb);

            codec->intra_dc_precision = get_be32(pb);

            codec->me_method = get_be32(pb);

            codec->mb_decision = get_be32(pb);

            codec->nsse_weight = get_be32(pb);

            codec->frame_skip_cmp = get_be32(pb);

            codec->rc_buffer_aggressivity = av_int2dbl(get_be64(pb));

            codec->codec_tag = get_be32(pb);

            codec->thread_count = get_byte(pb);

            codec->coder_type = get_be32(pb);

            codec->me_cmp = get_be32(pb);

            codec->partitions = get_be32(pb);

            codec->me_subpel_quality = get_be32(pb);

            codec->me_range = get_be32(pb);

            codec->keyint_min = get_be32(pb);

            codec->scenechange_threshold = get_be32(pb);

            codec->b_frame_strategy = get_be32(pb);

            codec->qcompress = av_int2dbl(get_be64(pb));

            codec->qblur = av_int2dbl(get_be64(pb));

            codec->max_qdiff = get_be32(pb);

            codec->refs = get_be32(pb);

            codec->directpred = get_be32(pb);

            break;

        case CODEC_TYPE_AUDIO:

            codec->sample_rate = get_be32(pb);

            codec->channels = get_le16(pb);

            codec->frame_size = get_le16(pb);

            codec->sample_fmt = get_le16(pb);

            break;

        default:

            goto fail;

        }

        if (codec->flags & CODEC_FLAG_GLOBAL_HEADER) {

            codec->extradata_size = get_be32(pb);

            codec->extradata = av_malloc(codec->extradata_size);

            if (!codec->extradata)

                return AVERROR(ENOMEM);

            get_buffer(pb, codec->extradata, codec->extradata_size);

        }

    }



    /* get until end of block reached */

    while ((url_ftell(pb) % ffm->packet_size) != 0)

        get_byte(pb);



    /* init packet demux */

    ffm->packet_ptr = ffm->packet;

    ffm->packet_end = ffm->packet;

    ffm->frame_offset = 0;

    ffm->dts = 0;

    ffm->read_state = READ_HEADER;

    ffm->first_packet = 1;

    return 0;

 fail:

    for(i=0;i<s->nb_streams;i++) {

        st = s->streams[i];

        if (st) {

            av_free(st);

        }

    }

    return -1;

}
