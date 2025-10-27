static int ogg_write_header(AVFormatContext *s)

{

    OGGContext *ogg = s->priv_data;

    OGGStreamContext *oggstream = NULL;

    int i, j;



    if (ogg->pref_size)

        av_log(s, AV_LOG_WARNING, "The pagesize option is deprecated\n");



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        unsigned serial_num = i;



        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            if (st->codec->codec_id == AV_CODEC_ID_OPUS)

                /* Opus requires a fixed 48kHz clock */

                avpriv_set_pts_info(st, 64, 1, 48000);

            else

                avpriv_set_pts_info(st, 64, 1, st->codec->sample_rate);

        } else if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

            avpriv_set_pts_info(st, 64, st->codec->time_base.num, st->codec->time_base.den);

        if (st->codec->codec_id != AV_CODEC_ID_VORBIS &&

            st->codec->codec_id != AV_CODEC_ID_THEORA &&

            st->codec->codec_id != AV_CODEC_ID_SPEEX  &&

            st->codec->codec_id != AV_CODEC_ID_FLAC   &&

            st->codec->codec_id != AV_CODEC_ID_OPUS) {

            av_log(s, AV_LOG_ERROR, "Unsupported codec id in stream %d\n", i);

            return -1;

        }



        if (!st->codec->extradata || !st->codec->extradata_size) {

            av_log(s, AV_LOG_ERROR, "No extradata present\n");

            return -1;

        }

        oggstream = av_mallocz(sizeof(*oggstream));

        if (!oggstream)

            return AVERROR(ENOMEM);



        oggstream->page.stream_index = i;



        if (!(s->flags & AVFMT_FLAG_BITEXACT))

            do {

                serial_num = av_get_random_seed();

                for (j = 0; j < i; j++) {

                    OGGStreamContext *sc = s->streams[j]->priv_data;

                    if (serial_num == sc->serial_num)

                        break;

                }

            } while (j < i);

        oggstream->serial_num = serial_num;



        av_dict_copy(&st->metadata, s->metadata, AV_DICT_DONT_OVERWRITE);



        st->priv_data = oggstream;

        if (st->codec->codec_id == AV_CODEC_ID_FLAC) {

            int err = ogg_build_flac_headers(st->codec, oggstream,

                                             s->flags & AVFMT_FLAG_BITEXACT,

                                             &st->metadata);

            if (err) {

                av_log(s, AV_LOG_ERROR, "Error writing FLAC headers\n");

                av_freep(&st->priv_data);

                return err;

            }

        } else if (st->codec->codec_id == AV_CODEC_ID_SPEEX) {

            int err = ogg_build_speex_headers(st->codec, oggstream,

                                              s->flags & AVFMT_FLAG_BITEXACT,

                                              &st->metadata);

            if (err) {

                av_log(s, AV_LOG_ERROR, "Error writing Speex headers\n");

                av_freep(&st->priv_data);

                return err;

            }

        } else if (st->codec->codec_id == AV_CODEC_ID_OPUS) {

            int err = ogg_build_opus_headers(st->codec, oggstream,

                                             s->flags & AVFMT_FLAG_BITEXACT,

                                             &st->metadata);

            if (err) {

                av_log(s, AV_LOG_ERROR, "Error writing Opus headers\n");

                av_freep(&st->priv_data);

                return err;

            }

        } else {

            uint8_t *p;

            const char *cstr = st->codec->codec_id == AV_CODEC_ID_VORBIS ? "vorbis" : "theora";

            int header_type = st->codec->codec_id == AV_CODEC_ID_VORBIS ? 3 : 0x81;

            int framing_bit = st->codec->codec_id == AV_CODEC_ID_VORBIS ? 1 : 0;



            if (avpriv_split_xiph_headers(st->codec->extradata, st->codec->extradata_size,

                                      st->codec->codec_id == AV_CODEC_ID_VORBIS ? 30 : 42,

                                      oggstream->header, oggstream->header_len) < 0) {

                av_log(s, AV_LOG_ERROR, "Extradata corrupted\n");

                av_freep(&st->priv_data);

                return -1;

            }



            p = ogg_write_vorbiscomment(7, s->flags & AVFMT_FLAG_BITEXACT,

                                        &oggstream->header_len[1], &st->metadata,

                                        framing_bit);

            oggstream->header[1] = p;

            if (!p)

                return AVERROR(ENOMEM);



            bytestream_put_byte(&p, header_type);

            bytestream_put_buffer(&p, cstr, 6);



            if (st->codec->codec_id == AV_CODEC_ID_THEORA) {

                /** KFGSHIFT is the width of the less significant section of the granule position

                    The less significant section is the frame count since the last keyframe */

                oggstream->kfgshift = ((oggstream->header[0][40]&3)<<3)|(oggstream->header[0][41]>>5);

                oggstream->vrev = oggstream->header[0][9];

                av_log(s, AV_LOG_DEBUG, "theora kfgshift %d, vrev %d\n",

                       oggstream->kfgshift, oggstream->vrev);

            }

        }

    }



    for (j = 0; j < s->nb_streams; j++) {

        OGGStreamContext *oggstream = s->streams[j]->priv_data;

        ogg_buffer_data(s, s->streams[j], oggstream->header[0],

                        oggstream->header_len[0], 0, 1);

        oggstream->page.flags |= 2; // bos

        ogg_buffer_page(s, oggstream);

    }

    for (j = 0; j < s->nb_streams; j++) {

        AVStream *st = s->streams[j];

        OGGStreamContext *oggstream = st->priv_data;

        for (i = 1; i < 3; i++) {

            if (oggstream->header_len[i])

                ogg_buffer_data(s, st, oggstream->header[i],

                                oggstream->header_len[i], 0, 1);

        }

        ogg_buffer_page(s, oggstream);

    }



    oggstream->page.start_granule = AV_NOPTS_VALUE;



    ogg_write_pages(s, 1);



    return 0;

}
