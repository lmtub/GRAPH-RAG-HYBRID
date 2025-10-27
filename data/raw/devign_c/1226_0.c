static int mxf_write_header(AVFormatContext *s)

{

    MXFContext *mxf = s->priv_data;

    int i, ret;

    uint8_t present[FF_ARRAY_ELEMS(mxf_essence_container_uls)] = {0};

    const MXFSamplesPerFrame *spf = NULL;

    AVDictionaryEntry *t;

    int64_t timestamp = 0;



    if (!s->nb_streams)

        return -1;



    if (s->oformat == &ff_mxf_opatom_muxer && s->nb_streams !=1){

        av_log(s, AV_LOG_ERROR, "there must be exactly one stream for mxf opatom\n");

        return -1;

    }



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        MXFStreamContext *sc = av_mallocz(sizeof(*sc));

        if (!sc)

            return AVERROR(ENOMEM);

        st->priv_data = sc;



        if (((i == 0) ^ (st->codec->codec_type == AVMEDIA_TYPE_VIDEO)) && s->oformat != &ff_mxf_opatom_muxer) {

            av_log(s, AV_LOG_ERROR, "there must be exactly one video stream and it must be the first one\n");

            return -1;

        }



        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            const AVPixFmtDescriptor *pix_desc = av_pix_fmt_desc_get(st->codec->pix_fmt);

            // TODO: should be avg_frame_rate

            AVRational rate, tbc = st->time_base;

            // Default component depth to 8

            sc->component_depth = 8;

            sc->h_chroma_sub_sample = 2;

            sc->color_siting = 0xFF;



            if (pix_desc) {

                sc->component_depth     = pix_desc->comp[0].depth_minus1 + 1;

                sc->h_chroma_sub_sample = 1 << pix_desc->log2_chroma_w;

            }

            switch (ff_choose_chroma_location(s, st)) {

            case AVCHROMA_LOC_TOPLEFT: sc->color_siting = 0; break;

            case AVCHROMA_LOC_LEFT:    sc->color_siting = 6; break;

            case AVCHROMA_LOC_TOP:     sc->color_siting = 1; break;

            case AVCHROMA_LOC_CENTER:  sc->color_siting = 3; break;

            }



            mxf->timecode_base = (tbc.den + tbc.num/2) / tbc.num;

            spf = ff_mxf_get_samples_per_frame(s, tbc);

            if (!spf) {

                av_log(s, AV_LOG_ERROR, "Unsupported video frame rate %d/%d\n",

                       tbc.den, tbc.num);

                return AVERROR(EINVAL);

            }

            mxf->time_base = spf->time_base;

            rate = av_inv_q(mxf->time_base);

            avpriv_set_pts_info(st, 64, mxf->time_base.num, mxf->time_base.den);

            if((ret = mxf_init_timecode(s, st, rate)) < 0)

                return ret;



            sc->video_bit_rate = st->codec->bit_rate ? st->codec->bit_rate : st->codec->rc_max_rate;

            if (s->oformat == &ff_mxf_d10_muxer) {

                if (sc->video_bit_rate == 50000000) {

                    if (mxf->time_base.den == 25) sc->index = 3;

                    else                          sc->index = 5;

                } else if (sc->video_bit_rate == 40000000) {

                    if (mxf->time_base.den == 25) sc->index = 7;

                    else                          sc->index = 9;

                } else if (sc->video_bit_rate == 30000000) {

                    if (mxf->time_base.den == 25) sc->index = 11;

                    else                          sc->index = 13;

                } else {

                    av_log(s, AV_LOG_ERROR, "error MXF D-10 only support 30/40/50 mbit/s\n");

                    return -1;

                }



                mxf->edit_unit_byte_count = KAG_SIZE; // system element

                mxf->edit_unit_byte_count += 16 + 4 + (uint64_t)sc->video_bit_rate *

                    mxf->time_base.num / (8*mxf->time_base.den);

                mxf->edit_unit_byte_count += klv_fill_size(mxf->edit_unit_byte_count);

                mxf->edit_unit_byte_count += 16 + 4 + 4 + spf->samples_per_frame[0]*8*4;

                mxf->edit_unit_byte_count += klv_fill_size(mxf->edit_unit_byte_count);



                sc->signal_standard = 1;

            }

        } else if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            if (st->codec->sample_rate != 48000) {

                av_log(s, AV_LOG_ERROR, "only 48khz is implemented\n");

                return -1;

            }

            avpriv_set_pts_info(st, 64, 1, st->codec->sample_rate);

            if (s->oformat == &ff_mxf_d10_muxer) {

                if (st->index != 1) {

                    av_log(s, AV_LOG_ERROR, "MXF D-10 only support one audio track\n");

                    return -1;

                }

                if (st->codec->codec_id != AV_CODEC_ID_PCM_S16LE &&

                    st->codec->codec_id != AV_CODEC_ID_PCM_S24LE) {

                    av_log(s, AV_LOG_ERROR, "MXF D-10 only support 16 or 24 bits le audio\n");

                }

                sc->index = ((MXFStreamContext*)s->streams[0]->priv_data)->index + 1;

            } else if (s->oformat == &ff_mxf_opatom_muxer) {

                AVRational tbc = av_inv_q(mxf->audio_edit_rate);



                if (st->codec->codec_id != AV_CODEC_ID_PCM_S16LE &&

                    st->codec->codec_id != AV_CODEC_ID_PCM_S24LE) {

                    av_log(s, AV_LOG_ERROR, "Only pcm_s16le and pcm_s24le audio codecs are implemented\n");

                    return AVERROR_PATCHWELCOME;

                }

                if (st->codec->channels != 1) {

                    av_log(s, AV_LOG_ERROR, "MXF OPAtom only supports single channel audio\n");

                    return AVERROR(EINVAL);

                }



                spf = ff_mxf_get_samples_per_frame(s, tbc);

                if (!spf){

                    av_log(s, AV_LOG_ERROR, "Unsupported timecode frame rate %d/%d\n", tbc.den, tbc.num);

                    return AVERROR(EINVAL);

                }



                mxf->time_base = st->time_base;

                if((ret = mxf_init_timecode(s, st, av_inv_q(spf->time_base))) < 0)

                    return ret;



                mxf->timecode_base = (tbc.den + tbc.num/2) / tbc.num;

                mxf->edit_unit_byte_count = (av_get_bits_per_sample(st->codec->codec_id) * st->codec->channels) >> 3;

                sc->index = 2;

            } else {

                mxf->slice_count = 1;

            }

        }



        if (!sc->index) {

            sc->index = mxf_get_essence_container_ul_index(st->codec->codec_id);

            if (sc->index == -1) {

                av_log(s, AV_LOG_ERROR, "track %d: could not find essence container ul, "

                       "codec not currently supported in container\n", i);

                return -1;

            }

        }



        sc->codec_ul = &mxf_essence_container_uls[sc->index].codec_ul;



        memcpy(sc->track_essence_element_key, mxf_essence_container_uls[sc->index].element_ul, 15);

        sc->track_essence_element_key[15] = present[sc->index];

        PRINT_KEY(s, "track essence element key", sc->track_essence_element_key);



        if (!present[sc->index])

            mxf->essence_container_count++;

        present[sc->index]++;

    }



    if (s->oformat == &ff_mxf_d10_muxer || s->oformat == &ff_mxf_opatom_muxer) {

        mxf->essence_container_count = 1;

    }



    if (!(s->flags & AVFMT_FLAG_BITEXACT))

        mxf_gen_umid(s);



    for (i = 0; i < s->nb_streams; i++) {

        MXFStreamContext *sc = s->streams[i]->priv_data;

        // update element count

        sc->track_essence_element_key[13] = present[sc->index];

        if (!memcmp(sc->track_essence_element_key, mxf_essence_container_uls[15].element_ul, 13)) // DV

            sc->order = (0x15 << 24) | AV_RB32(sc->track_essence_element_key+13);

        else

            sc->order = AV_RB32(sc->track_essence_element_key+12);

    }



    if (t = av_dict_get(s->metadata, "creation_time", NULL, 0))

        timestamp = ff_iso8601_to_unix_time(t->value);

    if (timestamp)

        mxf->timestamp = mxf_parse_timestamp(timestamp);

    mxf->duration = -1;



    mxf->timecode_track = av_mallocz(sizeof(*mxf->timecode_track));

    if (!mxf->timecode_track)

        return AVERROR(ENOMEM);

    mxf->timecode_track->priv_data = av_mallocz(sizeof(MXFStreamContext));

    if (!mxf->timecode_track->priv_data)

        return AVERROR(ENOMEM);

    mxf->timecode_track->index = -1;



    if (!spf)

        spf = ff_mxf_get_samples_per_frame(s, (AVRational){ 1, 25 });



    if (ff_audio_interleave_init(s, spf->samples_per_frame, mxf->time_base) < 0)

        return -1;



    return 0;

}
