static int avi_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    AVIContext *avi = s->priv_data;

    ByteIOContext *pb = &s->pb;

    uint32_t tag, tag1, handler;

    int codec_type, stream_index, frame_period, bit_rate;

    unsigned int size, nb_frames;

    int i, n;

    AVStream *st;

    AVIStream *ast = NULL;

    int xan_video = 0;  /* hack to support Xan A/V */

    char str_track[4];



    avi->stream_index= -1;



    if (get_riff(avi, pb) < 0)

        return -1;



    /* first list tag */

    stream_index = -1;

    codec_type = -1;

    frame_period = 0;

    for(;;) {

        if (url_feof(pb))

            goto fail;

        tag = get_le32(pb);

        size = get_le32(pb);

#ifdef DEBUG

        print_tag("tag", tag, size);

#endif



        switch(tag) {

        case MKTAG('L', 'I', 'S', 'T'):

            /* ignored, except when start of video packets */

            tag1 = get_le32(pb);

#ifdef DEBUG

            print_tag("list", tag1, 0);

#endif

            if (tag1 == MKTAG('m', 'o', 'v', 'i')) {

                avi->movi_list = url_ftell(pb) - 4;

                if(size) avi->movi_end = avi->movi_list + size + (size & 1);

                else     avi->movi_end = url_fsize(pb);

#ifdef DEBUG

                printf("movi end=%"PRIx64"\n", avi->movi_end);

#endif

                goto end_of_header;

            }

            break;

        case MKTAG('d', 'm', 'l', 'h'):

            avi->is_odml = 1;

            url_fskip(pb, size + (size & 1));

            break;

        case MKTAG('a', 'v', 'i', 'h'):

            /* avi header */

            /* using frame_period is bad idea */

            frame_period = get_le32(pb);

            bit_rate = get_le32(pb) * 8;

            get_le32(pb);

            avi->non_interleaved |= get_le32(pb) & AVIF_MUSTUSEINDEX;



            url_fskip(pb, 2 * 4);

            n = get_le32(pb);

            for(i=0;i<n;i++) {

                AVIStream *ast;

                st = av_new_stream(s, i);

                if (!st)

                    goto fail;



                ast = av_mallocz(sizeof(AVIStream));

                if (!ast)

                    goto fail;

                st->priv_data = ast;

            }

            url_fskip(pb, size - 7 * 4);

            break;

        case MKTAG('s', 't', 'r', 'h'):

            /* stream header */

            stream_index++;

            tag1 = get_le32(pb);

            handler = get_le32(pb); /* codec tag */

#ifdef DEBUG

            print_tag("strh", tag1, -1);

#endif

            if(tag1 == MKTAG('i', 'a', 'v', 's') || tag1 == MKTAG('i', 'v', 'a', 's')){

                /*

                 * After some consideration -- I don't think we

                 * have to support anything but DV in a type1 AVIs.

                 */

                if (s->nb_streams != 1)

                    goto fail;



                if (handler != MKTAG('d', 'v', 's', 'd') &&

                    handler != MKTAG('d', 'v', 'h', 'd') &&

                    handler != MKTAG('d', 'v', 's', 'l'))

                   goto fail;



                ast = s->streams[0]->priv_data;

                av_freep(&s->streams[0]->codec->extradata);

                av_freep(&s->streams[0]);

                s->nb_streams = 0;

                if (ENABLE_DV_DEMUXER) {

                    avi->dv_demux = dv_init_demux(s);

                    if (!avi->dv_demux)

                        goto fail;

                }

                s->streams[0]->priv_data = ast;

                url_fskip(pb, 3 * 4);

                ast->scale = get_le32(pb);

                ast->rate = get_le32(pb);

                stream_index = s->nb_streams - 1;

                url_fskip(pb, size - 7*4);

                break;

            }



            if (stream_index >= s->nb_streams) {

                url_fskip(pb, size - 8);

                /* ignore padding stream */

                if (tag1 == MKTAG('p', 'a', 'd', 's'))

                    stream_index--;

                break;

            }

            st = s->streams[stream_index];

            ast = st->priv_data;

            st->codec->stream_codec_tag= handler;



            get_le32(pb); /* flags */

            get_le16(pb); /* priority */

            get_le16(pb); /* language */

            get_le32(pb); /* initial frame */

            ast->scale = get_le32(pb);

            ast->rate = get_le32(pb);

            if(ast->scale && ast->rate){

            }else if(frame_period){

                ast->rate = 1000000;

                ast->scale = frame_period;

            }else{

                ast->rate = 25;

                ast->scale = 1;

            }

            av_set_pts_info(st, 64, ast->scale, ast->rate);



            ast->cum_len=get_le32(pb); /* start */

            nb_frames = get_le32(pb);



            st->start_time = 0;

            st->duration = nb_frames;

            get_le32(pb); /* buffer size */

            get_le32(pb); /* quality */

            ast->sample_size = get_le32(pb); /* sample ssize */

            ast->cum_len *= FFMAX(1, ast->sample_size);

//            av_log(NULL, AV_LOG_DEBUG, "%d %d %d %d\n", ast->rate, ast->scale, ast->start, ast->sample_size);



            switch(tag1) {

            case MKTAG('v', 'i', 'd', 's'):

                codec_type = CODEC_TYPE_VIDEO;



                ast->sample_size = 0;

                break;

            case MKTAG('a', 'u', 'd', 's'):

                codec_type = CODEC_TYPE_AUDIO;

                break;

            case MKTAG('t', 'x', 't', 's'):

                //FIXME

                codec_type = CODEC_TYPE_DATA; //CODEC_TYPE_SUB ?  FIXME

                break;

            case MKTAG('p', 'a', 'd', 's'):

                codec_type = CODEC_TYPE_UNKNOWN;

                stream_index--;

                break;

            default:

                av_log(s, AV_LOG_ERROR, "unknown stream type %X\n", tag1);

                goto fail;

            }

            ast->frame_offset= ast->cum_len;

            url_fskip(pb, size - 12 * 4);

            break;

        case MKTAG('s', 't', 'r', 'f'):

            /* stream header */

            if (stream_index >= s->nb_streams || avi->dv_demux) {

                url_fskip(pb, size);

            } else {

                st = s->streams[stream_index];

                switch(codec_type) {

                case CODEC_TYPE_VIDEO:

                    get_le32(pb); /* size */

                    st->codec->width = get_le32(pb);

                    st->codec->height = get_le32(pb);

                    get_le16(pb); /* panes */

                    st->codec->bits_per_sample= get_le16(pb); /* depth */

                    tag1 = get_le32(pb);

                    get_le32(pb); /* ImageSize */

                    get_le32(pb); /* XPelsPerMeter */

                    get_le32(pb); /* YPelsPerMeter */

                    get_le32(pb); /* ClrUsed */

                    get_le32(pb); /* ClrImportant */



                    if(size > 10*4 && size<(1<<30)){

                        st->codec->extradata_size= size - 10*4;

                        st->codec->extradata= av_malloc(st->codec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

                        get_buffer(pb, st->codec->extradata, st->codec->extradata_size);

                    }



                    if(st->codec->extradata_size & 1) //FIXME check if the encoder really did this correctly

                        get_byte(pb);



                    /* Extract palette from extradata if bpp <= 8 */

                    /* This code assumes that extradata contains only palette */

                    /* This is true for all paletted codecs implemented in ffmpeg */

                    if (st->codec->extradata_size && (st->codec->bits_per_sample <= 8)) {

                        st->codec->palctrl = av_mallocz(sizeof(AVPaletteControl));

#ifdef WORDS_BIGENDIAN

                        for (i = 0; i < FFMIN(st->codec->extradata_size, AVPALETTE_SIZE)/4; i++)

                            st->codec->palctrl->palette[i] = bswap_32(((uint32_t*)st->codec->extradata)[i]);

#else

                        memcpy(st->codec->palctrl->palette, st->codec->extradata,

                               FFMIN(st->codec->extradata_size, AVPALETTE_SIZE));

#endif

                        st->codec->palctrl->palette_changed = 1;

                    }



#ifdef DEBUG

                    print_tag("video", tag1, 0);

#endif

                    st->codec->codec_type = CODEC_TYPE_VIDEO;

                    st->codec->codec_tag = tag1;

                    st->codec->codec_id = codec_get_id(codec_bmp_tags, tag1);

                    if (st->codec->codec_id == CODEC_ID_XAN_WC4)

                        xan_video = 1;

                    st->need_parsing = 2; //only parse headers dont do slower repacketization, this is needed to get the pict type which is needed for generating correct pts

//                    url_fskip(pb, size - 5 * 4);

                    break;

                case CODEC_TYPE_AUDIO:

                    get_wav_header(pb, st->codec, size);

                    if(ast->sample_size && st->codec->block_align && ast->sample_size % st->codec->block_align)

                        av_log(s, AV_LOG_DEBUG, "invalid sample size or block align detected\n");

                    if (size%2) /* 2-aligned (fix for Stargate SG-1 - 3x18 - Shades of Grey.avi) */

                        url_fskip(pb, 1);

                    /* special case time: To support Xan DPCM, hardcode

                     * the format if Xxan is the video codec */

                    st->need_parsing = 1;

                    /* ADTS header is in extradata, AAC without header must be stored as exact frames, parser not needed and it will fail */

                    if (st->codec->codec_id == CODEC_ID_AAC && st->codec->extradata_size)

                        st->need_parsing = 0;

                    /* force parsing as several audio frames can be in

                       one packet */

                    if (xan_video)

                        st->codec->codec_id = CODEC_ID_XAN_DPCM;

                    break;

                default:

                    st->codec->codec_type = CODEC_TYPE_DATA;

                    st->codec->codec_id= CODEC_ID_NONE;

                    st->codec->codec_tag= 0;

                    url_fskip(pb, size);

                    break;

                }

            }

            break;

        case MKTAG('i', 'n', 'd', 'x'):

            i= url_ftell(pb);

            if(!url_is_streamed(pb) && !(s->flags & AVFMT_FLAG_IGNIDX)){

                read_braindead_odml_indx(s, 0);

            }

            url_fseek(pb, i+size, SEEK_SET);

            break;

        case MKTAG('I', 'N', 'A', 'M'):

            avi_read_tag(pb, s->title, sizeof(s->title), size);

            break;

        case MKTAG('I', 'A', 'R', 'T'):

            avi_read_tag(pb, s->author, sizeof(s->author), size);

            break;

        case MKTAG('I', 'C', 'O', 'P'):

            avi_read_tag(pb, s->copyright, sizeof(s->copyright), size);

            break;

        case MKTAG('I', 'C', 'M', 'T'):

            avi_read_tag(pb, s->comment, sizeof(s->comment), size);

            break;

        case MKTAG('I', 'G', 'N', 'R'):

            avi_read_tag(pb, s->genre, sizeof(s->genre), size);

            break;

        case MKTAG('I', 'P', 'R', 'D'):

            avi_read_tag(pb, s->album, sizeof(s->album), size);

            break;

        case MKTAG('I', 'P', 'R', 'T'):

            avi_read_tag(pb, str_track, sizeof(str_track), size);

            sscanf(str_track, "%d", &s->track);

            break;

        default:

            /* skip tag */

            size += (size & 1);

            url_fskip(pb, size);

            break;

        }

    }

 end_of_header:

    /* check stream number */

    if (stream_index != s->nb_streams - 1) {

    fail:

        for(i=0;i<s->nb_streams;i++) {

            av_freep(&s->streams[i]->codec->extradata);

            av_freep(&s->streams[i]);

        }

        return -1;

    }



    if(!avi->index_loaded && !url_is_streamed(pb))

        avi_load_index(s);

    avi->index_loaded = 1;

    avi->non_interleaved |= guess_ni_flag(s);

    if(avi->non_interleaved)

        clean_index(s);



    return 0;

}
