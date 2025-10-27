int ff_mov_read_stsd_entries(MOVContext *c, AVIOContext *pb, int entries)

{

    AVStream *st;

    MOVStreamContext *sc;

    int j, pseudo_stream_id;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];

    sc = st->priv_data;



    for (pseudo_stream_id=0; pseudo_stream_id<entries; pseudo_stream_id++) {

        //Parsing Sample description table

        enum CodecID id;

        int dref_id = 1;

        MOVAtom a = { AV_RL32("stsd") };

        int64_t start_pos = avio_tell(pb);

        int size = avio_rb32(pb); /* size */

        uint32_t format = avio_rl32(pb); /* data format */



        if (size >= 16) {

            avio_rb32(pb); /* reserved */

            avio_rb16(pb); /* reserved */

            dref_id = avio_rb16(pb);




        }



        if (st->codec->codec_tag &&

            st->codec->codec_tag != format &&

            (c->fc->video_codec_id ? ff_codec_get_id(codec_movvideo_tags, format) != c->fc->video_codec_id

                                   : st->codec->codec_tag != MKTAG('j','p','e','g'))

           ){

            /* Multiple fourcc, we skip JPEG. This is not correct, we should

             * export it as a separate AVStream but this needs a few changes

             * in the MOV demuxer, patch welcome. */

        multiple_stsd:

            av_log(c->fc, AV_LOG_WARNING, "multiple fourcc not supported\n");

            avio_skip(pb, size - (avio_tell(pb) - start_pos));

            continue;

        }

        /* we cannot demux concatenated h264 streams because of different extradata */

        if (st->codec->codec_tag && st->codec->codec_tag == AV_RL32("avc1"))

            goto multiple_stsd;

        sc->pseudo_stream_id = st->codec->codec_tag ? -1 : pseudo_stream_id;

        sc->dref_id= dref_id;



        st->codec->codec_tag = format;

        id = ff_codec_get_id(codec_movaudio_tags, format);

        if (id<=0 && ((format&0xFFFF) == 'm'+('s'<<8) || (format&0xFFFF) == 'T'+('S'<<8)))

            id = ff_codec_get_id(ff_codec_wav_tags, av_bswap32(format)&0xFFFF);



        if (st->codec->codec_type != AVMEDIA_TYPE_VIDEO && id > 0) {

            st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        } else if (st->codec->codec_type != AVMEDIA_TYPE_AUDIO && /* do not overwrite codec type */

                   format && format != MKTAG('m','p','4','s')) { /* skip old asf mpeg4 tag */

            id = ff_codec_get_id(codec_movvideo_tags, format);

            if (id <= 0)

                id = ff_codec_get_id(ff_codec_bmp_tags, format);

            if (id > 0)

                st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

            else if (st->codec->codec_type == AVMEDIA_TYPE_DATA){

                id = ff_codec_get_id(ff_codec_movsubtitle_tags, format);

                if (id > 0)

                    st->codec->codec_type = AVMEDIA_TYPE_SUBTITLE;

            }

        }



        av_dlog(c->fc, "size=%d 4CC= %c%c%c%c codec_type=%d\n", size,

                (format >> 0) & 0xff, (format >> 8) & 0xff, (format >> 16) & 0xff,

                (format >> 24) & 0xff, st->codec->codec_type);



        if (st->codec->codec_type==AVMEDIA_TYPE_VIDEO) {

            unsigned int color_depth, len;

            int color_greyscale;



            st->codec->codec_id = id;

            avio_rb16(pb); /* version */

            avio_rb16(pb); /* revision level */

            avio_rb32(pb); /* vendor */

            avio_rb32(pb); /* temporal quality */

            avio_rb32(pb); /* spatial quality */



            st->codec->width = avio_rb16(pb); /* width */

            st->codec->height = avio_rb16(pb); /* height */



            avio_rb32(pb); /* horiz resolution */

            avio_rb32(pb); /* vert resolution */

            avio_rb32(pb); /* data size, always 0 */

            avio_rb16(pb); /* frames per samples */



            len = avio_r8(pb); /* codec name, pascal string */

            if (len > 31)

                len = 31;

            mov_read_mac_string(c, pb, len, st->codec->codec_name, 32);

            if (len < 31)

                avio_skip(pb, 31 - len);

            /* codec_tag YV12 triggers an UV swap in rawdec.c */

            if (!memcmp(st->codec->codec_name, "Planar Y'CbCr 8-bit 4:2:0", 25))

                st->codec->codec_tag=MKTAG('I', '4', '2', '0');



            st->codec->bits_per_coded_sample = avio_rb16(pb); /* depth */

            st->codec->color_table_id = avio_rb16(pb); /* colortable id */

            av_dlog(c->fc, "depth %d, ctab id %d\n",

                   st->codec->bits_per_coded_sample, st->codec->color_table_id);

            /* figure out the palette situation */

            color_depth = st->codec->bits_per_coded_sample & 0x1F;

            color_greyscale = st->codec->bits_per_coded_sample & 0x20;



            /* if the depth is 2, 4, or 8 bpp, file is palettized */

            if ((color_depth == 2) || (color_depth == 4) ||

                (color_depth == 8)) {

                /* for palette traversal */

                unsigned int color_start, color_count, color_end;

                unsigned char r, g, b;



                if (color_greyscale) {

                    int color_index, color_dec;

                    /* compute the greyscale palette */

                    st->codec->bits_per_coded_sample = color_depth;

                    color_count = 1 << color_depth;

                    color_index = 255;

                    color_dec = 256 / (color_count - 1);

                    for (j = 0; j < color_count; j++) {

                        if (id == CODEC_ID_CINEPAK){

                            r = g = b = color_count - 1 - color_index;

                        }else

                        r = g = b = color_index;

                        sc->palette[j] =

                            (r << 16) | (g << 8) | (b);

                        color_index -= color_dec;

                        if (color_index < 0)

                            color_index = 0;

                    }

                } else if (st->codec->color_table_id) {

                    const uint8_t *color_table;

                    /* if flag bit 3 is set, use the default palette */

                    color_count = 1 << color_depth;

                    if (color_depth == 2)

                        color_table = ff_qt_default_palette_4;

                    else if (color_depth == 4)

                        color_table = ff_qt_default_palette_16;

                    else

                        color_table = ff_qt_default_palette_256;



                    for (j = 0; j < color_count; j++) {

                        r = color_table[j * 3 + 0];

                        g = color_table[j * 3 + 1];

                        b = color_table[j * 3 + 2];

                        sc->palette[j] =

                            (r << 16) | (g << 8) | (b);

                    }

                } else {

                    /* load the palette from the file */

                    color_start = avio_rb32(pb);

                    color_count = avio_rb16(pb);

                    color_end = avio_rb16(pb);

                    if ((color_start <= 255) &&

                        (color_end <= 255)) {

                        for (j = color_start; j <= color_end; j++) {

                            /* each R, G, or B component is 16 bits;

                             * only use the top 8 bits; skip alpha bytes

                             * up front */

                            avio_r8(pb);

                            avio_r8(pb);

                            r = avio_r8(pb);

                            avio_r8(pb);

                            g = avio_r8(pb);

                            avio_r8(pb);

                            b = avio_r8(pb);

                            avio_r8(pb);

                            sc->palette[j] =

                                (r << 16) | (g << 8) | (b);

                        }

                    }

                }

                sc->has_palette = 1;

            }

        } else if (st->codec->codec_type==AVMEDIA_TYPE_AUDIO) {

            int bits_per_sample, flags;

            uint16_t version = avio_rb16(pb);



            st->codec->codec_id = id;

            avio_rb16(pb); /* revision level */

            avio_rb32(pb); /* vendor */



            st->codec->channels = avio_rb16(pb);             /* channel count */

            av_dlog(c->fc, "audio channels %d\n", st->codec->channels);

            st->codec->bits_per_coded_sample = avio_rb16(pb);      /* sample size */



            sc->audio_cid = avio_rb16(pb);

            avio_rb16(pb); /* packet size = 0 */



            st->codec->sample_rate = ((avio_rb32(pb) >> 16));



            //Read QT version 1 fields. In version 0 these do not exist.

            av_dlog(c->fc, "version =%d, isom =%d\n",version,c->isom);

            if (!c->isom) {

                if (version==1) {

                    sc->samples_per_frame = avio_rb32(pb);

                    avio_rb32(pb); /* bytes per packet */

                    sc->bytes_per_frame = avio_rb32(pb);

                    avio_rb32(pb); /* bytes per sample */

                } else if (version==2) {

                    avio_rb32(pb); /* sizeof struct only */

                    st->codec->sample_rate = av_int2double(avio_rb64(pb)); /* float 64 */

                    st->codec->channels = avio_rb32(pb);

                    avio_rb32(pb); /* always 0x7F000000 */

                    st->codec->bits_per_coded_sample = avio_rb32(pb); /* bits per channel if sound is uncompressed */

                    flags = avio_rb32(pb); /* lpcm format specific flag */

                    sc->bytes_per_frame = avio_rb32(pb); /* bytes per audio packet if constant */

                    sc->samples_per_frame = avio_rb32(pb); /* lpcm frames per audio packet if constant */

                    if (format == MKTAG('l','p','c','m'))

                        st->codec->codec_id = ff_mov_get_lpcm_codec_id(st->codec->bits_per_coded_sample, flags);

                }

            }



            switch (st->codec->codec_id) {

            case CODEC_ID_PCM_S8:

            case CODEC_ID_PCM_U8:

                if (st->codec->bits_per_coded_sample == 16)

                    st->codec->codec_id = CODEC_ID_PCM_S16BE;

                break;

            case CODEC_ID_PCM_S16LE:

            case CODEC_ID_PCM_S16BE:

                if (st->codec->bits_per_coded_sample == 8)

                    st->codec->codec_id = CODEC_ID_PCM_S8;

                else if (st->codec->bits_per_coded_sample == 24)

                    st->codec->codec_id =

                        st->codec->codec_id == CODEC_ID_PCM_S16BE ?

                        CODEC_ID_PCM_S24BE : CODEC_ID_PCM_S24LE;

                break;

            /* set values for old format before stsd version 1 appeared */

            case CODEC_ID_MACE3:

                sc->samples_per_frame = 6;

                sc->bytes_per_frame = 2*st->codec->channels;

                break;

            case CODEC_ID_MACE6:

                sc->samples_per_frame = 6;

                sc->bytes_per_frame = 1*st->codec->channels;

                break;

            case CODEC_ID_ADPCM_IMA_QT:

                sc->samples_per_frame = 64;

                sc->bytes_per_frame = 34*st->codec->channels;

                break;

            case CODEC_ID_GSM:

                sc->samples_per_frame = 160;

                sc->bytes_per_frame = 33;

                break;

            default:

                break;

            }



            bits_per_sample = av_get_bits_per_sample(st->codec->codec_id);

            if (bits_per_sample) {

                st->codec->bits_per_coded_sample = bits_per_sample;

                sc->sample_size = (bits_per_sample >> 3) * st->codec->channels;

            }

        } else if (st->codec->codec_type==AVMEDIA_TYPE_SUBTITLE){

            // ttxt stsd contains display flags, justification, background

            // color, fonts, and default styles, so fake an atom to read it

            MOVAtom fake_atom = { .size = size - (avio_tell(pb) - start_pos) };

            if (format != AV_RL32("mp4s")) // mp4s contains a regular esds atom

                mov_read_glbl(c, pb, fake_atom);

            st->codec->codec_id= id;

            st->codec->width = sc->width;

            st->codec->height = sc->height;

        } else {

            if (st->codec->codec_tag == MKTAG('t','m','c','d')) {

                int val;

                avio_rb32(pb);       /* reserved */

                val = avio_rb32(pb); /* flags */

                if (val & 1)

                    st->codec->flags2 |= CODEC_FLAG2_DROP_FRAME_TIMECODE;

                avio_rb32(pb);

                avio_rb32(pb);

                st->codec->time_base.den = avio_r8(pb);

                st->codec->time_base.num = 1;

            }

            /* other codec type, just skip (rtp, mp4s, ...) */

            avio_skip(pb, size - (avio_tell(pb) - start_pos));

        }

        /* this will read extra atoms at the end (wave, alac, damr, avcC, SMI ...) */

        a.size = size - (avio_tell(pb) - start_pos);

        if (a.size > 8) {

            if (mov_read_default(c, pb, a) < 0)


        } else if (a.size > 0)

            avio_skip(pb, a.size);

    }



    if (st->codec->codec_type==AVMEDIA_TYPE_AUDIO && st->codec->sample_rate==0 && sc->time_scale>1)

        st->codec->sample_rate= sc->time_scale;



    /* special codec parameters handling */

    switch (st->codec->codec_id) {

#if CONFIG_DV_DEMUXER

    case CODEC_ID_DVAUDIO:

        c->dv_fctx = avformat_alloc_context();

        c->dv_demux = avpriv_dv_init_demux(c->dv_fctx);

        if (!c->dv_demux) {

            av_log(c->fc, AV_LOG_ERROR, "dv demux context init error\n");


        }

        sc->dv_audio_container = 1;

        st->codec->codec_id = CODEC_ID_PCM_S16LE;

        break;

#endif

    /* no ifdef since parameters are always those */

    case CODEC_ID_QCELP:

        // force sample rate for qcelp when not stored in mov

        if (st->codec->codec_tag != MKTAG('Q','c','l','p'))

            st->codec->sample_rate = 8000;

        st->codec->frame_size= 160;

        st->codec->channels= 1; /* really needed */

        break;

    case CODEC_ID_AMR_NB:

        st->codec->channels= 1; /* really needed */

        /* force sample rate for amr, stsd in 3gp does not store sample rate */

        st->codec->sample_rate = 8000;

        /* force frame_size, too, samples_per_frame isn't always set properly */

        st->codec->frame_size  = 160;

        break;

    case CODEC_ID_AMR_WB:

        st->codec->channels    = 1;

        st->codec->sample_rate = 16000;

        st->codec->frame_size  = 320;

        break;

    case CODEC_ID_MP2:

    case CODEC_ID_MP3:

        st->codec->codec_type = AVMEDIA_TYPE_AUDIO; /* force type after stsd for m1a hdlr */

        st->need_parsing = AVSTREAM_PARSE_FULL;

        break;

    case CODEC_ID_GSM:

    case CODEC_ID_ADPCM_MS:

    case CODEC_ID_ADPCM_IMA_WAV:

        st->codec->frame_size = sc->samples_per_frame;

        st->codec->block_align = sc->bytes_per_frame;

        break;

    case CODEC_ID_ALAC:

        if (st->codec->extradata_size == 36) {

            st->codec->frame_size = AV_RB32(st->codec->extradata+12);

            st->codec->channels   = AV_RB8 (st->codec->extradata+21);

            st->codec->sample_rate = AV_RB32(st->codec->extradata+32);

        }

        break;

    case CODEC_ID_AC3:

        st->need_parsing = AVSTREAM_PARSE_FULL;

        break;

    case CODEC_ID_MPEG1VIDEO:

        st->need_parsing = AVSTREAM_PARSE_FULL;

        break;

    default:

        break;

    }



    return 0;

}