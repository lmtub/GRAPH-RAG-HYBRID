static int ffm_write_header(AVFormatContext *s)

{

    FFMContext *ffm = s->priv_data;

    AVStream *st;

    ByteIOContext *pb = s->pb;

    AVCodecContext *codec;

    int bit_rate, i;



    ffm->packet_size = FFM_PACKET_SIZE;



    /* header */

    put_le32(pb, MKTAG('F', 'F', 'M', '1'));

    put_be32(pb, ffm->packet_size);

    /* XXX: store write position in other file ? */

    put_be64(pb, ffm->packet_size); /* current write position */



    put_be32(pb, s->nb_streams);

    bit_rate = 0;

    for(i=0;i<s->nb_streams;i++) {

        st = s->streams[i];

        bit_rate += st->codec->bit_rate;

    }

    put_be32(pb, bit_rate);



    /* list of streams */

    for(i=0;i<s->nb_streams;i++) {

        st = s->streams[i];

        av_set_pts_info(st, 64, 1, 1000000);



        codec = st->codec;

        /* generic info */

        put_be32(pb, codec->codec_id);

        put_byte(pb, codec->codec_type);

        put_be32(pb, codec->bit_rate);

        put_be32(pb, st->quality);

        put_be32(pb, codec->flags);

        put_be32(pb, codec->flags2);

        put_be32(pb, codec->debug);

        /* specific info */

        switch(codec->codec_type) {

        case CODEC_TYPE_VIDEO:

            put_be32(pb, codec->time_base.num);

            put_be32(pb, codec->time_base.den);

            put_be16(pb, codec->width);

            put_be16(pb, codec->height);

            put_be16(pb, codec->gop_size);

            put_be32(pb, codec->pix_fmt);

            put_byte(pb, codec->qmin);

            put_byte(pb, codec->qmax);

            put_byte(pb, codec->max_qdiff);

            put_be16(pb, (int) (codec->qcompress * 10000.0));

            put_be16(pb, (int) (codec->qblur * 10000.0));

            put_be32(pb, codec->bit_rate_tolerance);

            put_strz(pb, codec->rc_eq);

            put_be32(pb, codec->rc_max_rate);

            put_be32(pb, codec->rc_min_rate);

            put_be32(pb, codec->rc_buffer_size);

            put_be64(pb, av_dbl2int(codec->i_quant_factor));

            put_be64(pb, av_dbl2int(codec->b_quant_factor));

            put_be64(pb, av_dbl2int(codec->i_quant_offset));

            put_be64(pb, av_dbl2int(codec->b_quant_offset));

            put_be32(pb, codec->dct_algo);

            put_be32(pb, codec->strict_std_compliance);

            put_be32(pb, codec->max_b_frames);

            put_be32(pb, codec->luma_elim_threshold);

            put_be32(pb, codec->chroma_elim_threshold);

            put_be32(pb, codec->mpeg_quant);

            put_be32(pb, codec->intra_dc_precision);

            put_be32(pb, codec->me_method);

            put_be32(pb, codec->mb_decision);

            put_be32(pb, codec->nsse_weight);

            put_be32(pb, codec->frame_skip_cmp);

            put_be64(pb, av_dbl2int(codec->rc_buffer_aggressivity));

            put_be32(pb, codec->codec_tag);

            put_byte(pb, codec->thread_count);

            break;

        case CODEC_TYPE_AUDIO:

            put_be32(pb, codec->sample_rate);

            put_le16(pb, codec->channels);

            put_le16(pb, codec->frame_size);

            break;

        default:

            return -1;

        }

        if (codec->flags & CODEC_FLAG_GLOBAL_HEADER) {

            put_be32(pb, codec->extradata_size);

            put_buffer(pb, codec->extradata, codec->extradata_size);

        }

    }



    /* flush until end of block reached */

    while ((url_ftell(pb) % ffm->packet_size) != 0)

        put_byte(pb, 0);



    put_flush_packet(pb);



    /* init packet mux */

    ffm->packet_ptr = ffm->packet;

    ffm->packet_end = ffm->packet + ffm->packet_size - FFM_HEADER_SIZE;

    assert(ffm->packet_end >= ffm->packet);

    ffm->frame_offset = 0;

    ffm->dts = 0;

    ffm->first_packet = 1;



    return 0;

}
