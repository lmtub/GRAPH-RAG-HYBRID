static int avi_read_header(AVFormatContext *s)

{

    AVIContext *avi = s->priv_data;

    AVIOContext *pb = s->pb;

    unsigned int tag, tag1, handler;

    int codec_type, stream_index, frame_period;

    unsigned int size;

    int i;

    AVStream *st;

    AVIStream *ast      = NULL;

    int avih_width      = 0, avih_height = 0;

    int amv_file_format = 0;

    uint64_t list_end   = 0;

    int64_t pos;

    int ret;

    AVDictionaryEntry *dict_entry;



    avi->stream_index = -1;



    ret = get_riff(s, pb);

    if (ret < 0)

        return ret;



    av_log(avi, AV_LOG_DEBUG, "use odml:%d\n", avi->use_odml);



    avi->io_fsize = avi->fsize = avio_size(pb);

    if (avi->fsize <= 0 || avi->fsize < avi->riff_end)

        avi->fsize = avi->riff_end == 8 ? INT64_MAX : avi->riff_end;



    /* first list tag */

    stream_index = -1;

    codec_type   = -1;

    frame_period = 0;

    for (;;) {

        if (avio_feof(pb))

            goto fail;

        tag  = avio_rl32(pb);

        size = avio_rl32(pb);



        print_tag("tag", tag, size);



        switch (tag) {

        case MKTAG('L', 'I', 'S', 'T'):

            list_end = avio_tell(pb) + size;

            /* Ignored, except at start of video packets. */

            tag1 = avio_rl32(pb);



            print_tag("list", tag1, 0);



            if (tag1 == MKTAG('m', 'o', 'v', 'i')) {

                avi->movi_list = avio_tell(pb) - 4;

                if (size)

                    avi->movi_end = avi->movi_list + size + (size & 1);

                else

                    avi->movi_end = avi->fsize;

                av_log(NULL, AV_LOG_TRACE, "movi end=%"PRIx64"\n", avi->movi_end);

                goto end_of_header;

            } else if (tag1 == MKTAG('I', 'N', 'F', 'O'))

                ff_read_riff_info(s, size - 4);

            else if (tag1 == MKTAG('n', 'c', 'd', 't'))

                avi_read_nikon(s, list_end);



            break;

        case MKTAG('I', 'D', 'I', 'T'):

        {

            unsigned char date[64] = { 0 };

            size += (size & 1);

            size -= avio_read(pb, date, FFMIN(size, sizeof(date) - 1));

            avio_skip(pb, size);

            avi_metadata_creation_time(&s->metadata, date);

            break;


        case MKTAG('d', 'm', 'l', 'h'):

            avi->is_odml = 1;

            avio_skip(pb, size + (size & 1));

            break;

        case MKTAG('a', 'm', 'v', 'h'):

            amv_file_format = 1;

        case MKTAG('a', 'v', 'i', 'h'):

            /* AVI header */

            /* using frame_period is bad idea */

            frame_period = avio_rl32(pb);

            avio_rl32(pb); /* max. bytes per second */

            avio_rl32(pb);

            avi->non_interleaved |= avio_rl32(pb) & AVIF_MUSTUSEINDEX;



            avio_skip(pb, 2 * 4);

            avio_rl32(pb);

            avio_rl32(pb);

            avih_width  = avio_rl32(pb);

            avih_height = avio_rl32(pb);



            avio_skip(pb, size - 10 * 4);

            break;

        case MKTAG('s', 't', 'r', 'h'):

            /* stream header */



            tag1    = avio_rl32(pb);

            handler = avio_rl32(pb); /* codec tag */



            if (tag1 == MKTAG('p', 'a', 'd', 's')) {

                avio_skip(pb, size - 8);

                break;

            } else {

                stream_index++;

                st = avformat_new_stream(s, NULL);

                if (!st)

                    goto fail;



                st->id = stream_index;

                ast    = av_mallocz(sizeof(AVIStream));

                if (!ast)

                    goto fail;

                st->priv_data = ast;


            if (amv_file_format)

                tag1 = stream_index ? MKTAG('a', 'u', 'd', 's')

                                    : MKTAG('v', 'i', 'd', 's');



            print_tag("strh", tag1, -1);



            if (tag1 == MKTAG('i', 'a', 'v', 's') ||

                tag1 == MKTAG('i', 'v', 'a', 's')) {

                int64_t dv_dur;



                /* After some consideration -- I don't think we

                 * have to support anything but DV in type1 AVIs. */

                if (s->nb_streams != 1)

                    goto fail;



                if (handler != MKTAG('d', 'v', 's', 'd') &&

                    handler != MKTAG('d', 'v', 'h', 'd') &&

                    handler != MKTAG('d', 'v', 's', 'l'))

                    goto fail;



                ast = s->streams[0]->priv_data;

                av_freep(&s->streams[0]->codecpar->extradata);

                av_freep(&s->streams[0]->codecpar);

#if FF_API_LAVF_AVCTX

FF_DISABLE_DEPRECATION_WARNINGS

                av_freep(&s->streams[0]->codec);

FF_ENABLE_DEPRECATION_WARNINGS

#endif

                if (s->streams[0]->info)

                    av_freep(&s->streams[0]->info->duration_error);

                av_freep(&s->streams[0]->info);

                if (s->streams[0]->internal)

                    av_freep(&s->streams[0]->internal->avctx);

                av_freep(&s->streams[0]->internal);

                av_freep(&s->streams[0]);

                s->nb_streams = 0;

                if (CONFIG_DV_DEMUXER) {

                    avi->dv_demux = avpriv_dv_init_demux(s);

                    if (!avi->dv_demux)

                        goto fail;

                } else

                    goto fail;

                s->streams[0]->priv_data = ast;

                avio_skip(pb, 3 * 4);

                ast->scale = avio_rl32(pb);

                ast->rate  = avio_rl32(pb);

                avio_skip(pb, 4);  /* start time */



                dv_dur = avio_rl32(pb);

                if (ast->scale > 0 && ast->rate > 0 && dv_dur > 0) {

                    dv_dur     *= AV_TIME_BASE;

                    s->duration = av_rescale(dv_dur, ast->scale, ast->rate);


                /* else, leave duration alone; timing estimation in utils.c

                 * will make a guess based on bitrate. */



                stream_index = s->nb_streams - 1;

                avio_skip(pb, size - 9 * 4);

                break;




            av_assert0(stream_index < s->nb_streams);

            ast->handler = handler;



            avio_rl32(pb); /* flags */

            avio_rl16(pb); /* priority */

            avio_rl16(pb); /* language */

            avio_rl32(pb); /* initial frame */

            ast->scale = avio_rl32(pb);

            ast->rate  = avio_rl32(pb);

            if (!(ast->scale && ast->rate)) {

                av_log(s, AV_LOG_WARNING,

                       "scale/rate is %"PRIu32"/%"PRIu32" which is invalid. "

                       "(This file has been generated by broken software.)\n",

                       ast->scale,

                       ast->rate);

                if (frame_period) {

                    ast->rate  = 1000000;

                    ast->scale = frame_period;

                } else {

                    ast->rate  = 25;

                    ast->scale = 1;



            avpriv_set_pts_info(st, 64, ast->scale, ast->rate);



            ast->cum_len  = avio_rl32(pb); /* start */

            st->nb_frames = avio_rl32(pb);



            st->start_time = 0;

            avio_rl32(pb); /* buffer size */

            avio_rl32(pb); /* quality */

            if (ast->cum_len*ast->scale/ast->rate > 3600) {

                av_log(s, AV_LOG_ERROR, "crazy start time, iam scared, giving up\n");

                ast->cum_len = 0;


            ast->sample_size = avio_rl32(pb);

            ast->cum_len    *= FFMAX(1, ast->sample_size);

            av_log(s, AV_LOG_TRACE, "%"PRIu32" %"PRIu32" %d\n",

                    ast->rate, ast->scale, ast->sample_size);



            switch (tag1) {

            case MKTAG('v', 'i', 'd', 's'):

                codec_type = AVMEDIA_TYPE_VIDEO;



                ast->sample_size = 0;

                st->avg_frame_rate = av_inv_q(st->time_base);

                break;

            case MKTAG('a', 'u', 'd', 's'):

                codec_type = AVMEDIA_TYPE_AUDIO;

                break;

            case MKTAG('t', 'x', 't', 's'):

                codec_type = AVMEDIA_TYPE_SUBTITLE;

                break;

            case MKTAG('d', 'a', 't', 's'):

                codec_type = AVMEDIA_TYPE_DATA;

                break;

            default:

                av_log(s, AV_LOG_INFO, "unknown stream type %X\n", tag1);




            if (ast->sample_size < 0) {

                if (s->error_recognition & AV_EF_EXPLODE) {

                    av_log(s, AV_LOG_ERROR,

                           "Invalid sample_size %d at stream %d\n",

                           ast->sample_size,

                           stream_index);

                    goto fail;


                av_log(s, AV_LOG_WARNING,

                       "Invalid sample_size %d at stream %d "

                       "setting it to 0\n",

                       ast->sample_size,

                       stream_index);

                ast->sample_size = 0;




            if (ast->sample_size == 0) {

                st->duration = st->nb_frames;

                if (st->duration > 0 && avi->io_fsize > 0 && avi->riff_end > avi->io_fsize) {

                    av_log(s, AV_LOG_DEBUG, "File is truncated adjusting duration\n");

                    st->duration = av_rescale(st->duration, avi->io_fsize, avi->riff_end);



            ast->frame_offset = ast->cum_len;

            avio_skip(pb, size - 12 * 4);

            break;

        case MKTAG('s', 't', 'r', 'f'):

            /* stream header */

            if (!size)

                break;

            if (stream_index >= (unsigned)s->nb_streams || avi->dv_demux) {

                avio_skip(pb, size);

            } else {

                uint64_t cur_pos = avio_tell(pb);

                unsigned esize;

                if (cur_pos < list_end)

                    size = FFMIN(size, list_end - cur_pos);

                st = s->streams[stream_index];

                if (st->codecpar->codec_type != AVMEDIA_TYPE_UNKNOWN) {

                    avio_skip(pb, size);

                    break;


                switch (codec_type) {

                case AVMEDIA_TYPE_VIDEO:

                    if (amv_file_format) {

                        st->codecpar->width      = avih_width;

                        st->codecpar->height     = avih_height;

                        st->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;

                        st->codecpar->codec_id   = AV_CODEC_ID_AMV;

                        avio_skip(pb, size);

                        break;


                    tag1 = ff_get_bmp_header(pb, st, &esize);



                    if (tag1 == MKTAG('D', 'X', 'S', 'B') ||

                        tag1 == MKTAG('D', 'X', 'S', 'A')) {

                        st->codecpar->codec_type = AVMEDIA_TYPE_SUBTITLE;

                        st->codecpar->codec_tag  = tag1;

                        st->codecpar->codec_id   = AV_CODEC_ID_XSUB;

                        break;




                    if (size > 10 * 4 && size < (1 << 30) && size < avi->fsize) {

                        if (esize == size-1 && (esize&1)) {

                            st->codecpar->extradata_size = esize - 10 * 4;

                        } else

                            st->codecpar->extradata_size =  size - 10 * 4;





                        if (ff_get_extradata(s, st->codecpar, pb, st->codecpar->extradata_size) < 0)

                            return AVERROR(ENOMEM);




                    // FIXME: check if the encoder really did this correctly

                    if (st->codecpar->extradata_size & 1)

                        avio_r8(pb);



                    /* Extract palette from extradata if bpp <= 8.

                     * This code assumes that extradata contains only palette.

                     * This is true for all paletted codecs implemented in

                     * FFmpeg. */

                    if (st->codecpar->extradata_size &&

                        (st->codecpar->bits_per_coded_sample <= 8)) {

                        int pal_size = (1 << st->codecpar->bits_per_coded_sample) << 2;

                        const uint8_t *pal_src;



                        pal_size = FFMIN(pal_size, st->codecpar->extradata_size);

                        pal_src  = st->codecpar->extradata +

                                   st->codecpar->extradata_size - pal_size;

                        /* Exclude the "BottomUp" field from the palette */

                        if (pal_src - st->codecpar->extradata >= 9 &&

                            !memcmp(st->codecpar->extradata + st->codecpar->extradata_size - 9, "BottomUp", 9))

                            pal_src -= 9;

                        for (i = 0; i < pal_size / 4; i++)

                            ast->pal[i] = 0xFFU<<24 | AV_RL32(pal_src+4*i);

                        ast->has_pal = 1;




                    print_tag("video", tag1, 0);



                    st->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;

                    st->codecpar->codec_tag  = tag1;

                    st->codecpar->codec_id   = ff_codec_get_id(ff_codec_bmp_tags,

                                                            tag1);

                    /* If codec is not found yet, try with the mov tags. */

                    if (!st->codecpar->codec_id) {

                        char tag_buf[32];

                        av_get_codec_tag_string(tag_buf, sizeof(tag_buf), tag1);

                        st->codecpar->codec_id =

                            ff_codec_get_id(ff_codec_movvideo_tags, tag1);

                        if (st->codecpar->codec_id)

                           av_log(s, AV_LOG_WARNING,

                                  "mov tag found in avi (fourcc %s)\n",

                                  tag_buf);


                    /* This is needed to get the pict type which is necessary

                     * for generating correct pts. */

                    st->need_parsing = AVSTREAM_PARSE_HEADERS;



                    if (st->codecpar->codec_id == AV_CODEC_ID_MPEG4 &&

                        ast->handler == MKTAG('X', 'V', 'I', 'D'))

                        st->codecpar->codec_tag = MKTAG('X', 'V', 'I', 'D');



                    if (st->codecpar->codec_tag == MKTAG('V', 'S', 'S', 'H'))

                        st->need_parsing = AVSTREAM_PARSE_FULL;

                    if (st->codecpar->codec_id == AV_CODEC_ID_RV40)

                        st->need_parsing = AVSTREAM_PARSE_NONE;



                    if (st->codecpar->codec_tag == 0 && st->codecpar->height > 0 &&

                        st->codecpar->extradata_size < 1U << 30) {

                        st->codecpar->extradata_size += 9;

                        if ((ret = av_reallocp(&st->codecpar->extradata,

                                               st->codecpar->extradata_size +

                                               AV_INPUT_BUFFER_PADDING_SIZE)) < 0) {

                            st->codecpar->extradata_size = 0;

                            return ret;

                        } else

                            memcpy(st->codecpar->extradata + st->codecpar->extradata_size - 9,

                                   "BottomUp", 9);


                    st->codecpar->height = FFABS(st->codecpar->height);



//                    avio_skip(pb, size - 5 * 4);

                    break;

                case AVMEDIA_TYPE_AUDIO:

                    ret = ff_get_wav_header(s, pb, st->codecpar, size, 0);

                    if (ret < 0)

                        return ret;

                    ast->dshow_block_align = st->codecpar->block_align;

                    if (ast->sample_size && st->codecpar->block_align &&

                        ast->sample_size != st->codecpar->block_align) {

                        av_log(s,

                               AV_LOG_WARNING,

                               "sample size (%d) != block align (%d)\n",

                               ast->sample_size,

                               st->codecpar->block_align);

                        ast->sample_size = st->codecpar->block_align;


                    /* 2-aligned

                     * (fix for Stargate SG-1 - 3x18 - Shades of Grey.avi) */

                    if (size & 1)

                        avio_skip(pb, 1);

                    /* Force parsing as several audio frames can be in

                     * one packet and timestamps refer to packet start. */

                    st->need_parsing = AVSTREAM_PARSE_TIMESTAMPS;

                    /* ADTS header is in extradata, AAC without header must be

                     * stored as exact frames. Parser not needed and it will

                     * fail. */

                    if (st->codecpar->codec_id == AV_CODEC_ID_AAC &&

                        st->codecpar->extradata_size)

                        st->need_parsing = AVSTREAM_PARSE_NONE;

                    // The flac parser does not work with AVSTREAM_PARSE_TIMESTAMPS

                    if (st->codecpar->codec_id == AV_CODEC_ID_FLAC)

                        st->need_parsing = AVSTREAM_PARSE_NONE;

                    /* AVI files with Xan DPCM audio (wrongly) declare PCM

                     * audio in the header but have Axan as stream_code_tag. */

                    if (ast->handler == AV_RL32("Axan")) {

                        st->codecpar->codec_id  = AV_CODEC_ID_XAN_DPCM;

                        st->codecpar->codec_tag = 0;

                        ast->dshow_block_align = 0;


                    if (amv_file_format) {

                        st->codecpar->codec_id    = AV_CODEC_ID_ADPCM_IMA_AMV;

                        ast->dshow_block_align = 0;


                    if ((st->codecpar->codec_id == AV_CODEC_ID_AAC  ||

                         st->codecpar->codec_id == AV_CODEC_ID_FLAC ||

                         st->codecpar->codec_id == AV_CODEC_ID_MP2 ) && ast->dshow_block_align <= 4 && ast->dshow_block_align) {

                        av_log(s, AV_LOG_DEBUG, "overriding invalid dshow_block_align of %d\n", ast->dshow_block_align);

                        ast->dshow_block_align = 0;


                    if (st->codecpar->codec_id == AV_CODEC_ID_AAC && ast->dshow_block_align == 1024 && ast->sample_size == 1024 ||

                       st->codecpar->codec_id == AV_CODEC_ID_AAC && ast->dshow_block_align == 4096 && ast->sample_size == 4096 ||

                       st->codecpar->codec_id == AV_CODEC_ID_MP3 && ast->dshow_block_align == 1152 && ast->sample_size == 1152) {

                        av_log(s, AV_LOG_DEBUG, "overriding sample_size\n");

                        ast->sample_size = 0;


                    break;

                case AVMEDIA_TYPE_SUBTITLE:

                    st->codecpar->codec_type = AVMEDIA_TYPE_SUBTITLE;

                    st->request_probe= 1;

                    avio_skip(pb, size);

                    break;

                default:

                    st->codecpar->codec_type = AVMEDIA_TYPE_DATA;

                    st->codecpar->codec_id   = AV_CODEC_ID_NONE;

                    st->codecpar->codec_tag  = 0;

                    avio_skip(pb, size);

                    break;



            break;

        case MKTAG('s', 't', 'r', 'd'):

            if (stream_index >= (unsigned)s->nb_streams

                || s->streams[stream_index]->codecpar->extradata_size

                || s->streams[stream_index]->codecpar->codec_tag == MKTAG('H','2','6','4')) {

                avio_skip(pb, size);

            } else {

                uint64_t cur_pos = avio_tell(pb);

                if (cur_pos < list_end)

                    size = FFMIN(size, list_end - cur_pos);

                st = s->streams[stream_index];



                if (size<(1<<30)) {


                        av_log(s, AV_LOG_WARNING, "New extradata in strd chunk, freeing previous one.\n");



                    if (ff_get_extradata(s, st->codecpar, pb, size) < 0)

                        return AVERROR(ENOMEM);




                if (st->codecpar->extradata_size & 1) //FIXME check if the encoder really did this correctly

                    avio_r8(pb);



                ret = avi_extract_stream_metadata(s, st);

                if (ret < 0) {

                    av_log(s, AV_LOG_WARNING, "could not decoding EXIF data in stream header.\n");



            break;

        case MKTAG('i', 'n', 'd', 'x'):

            pos = avio_tell(pb);

            if (pb->seekable && !(s->flags & AVFMT_FLAG_IGNIDX) &&

                avi->use_odml &&

                read_braindead_odml_indx(s, 0) < 0 &&

                (s->error_recognition & AV_EF_EXPLODE))

                goto fail;

            avio_seek(pb, pos + size, SEEK_SET);

            break;

        case MKTAG('v', 'p', 'r', 'p'):

            if (stream_index < (unsigned)s->nb_streams && size > 9 * 4) {

                AVRational active, active_aspect;



                st = s->streams[stream_index];

                avio_rl32(pb);

                avio_rl32(pb);

                avio_rl32(pb);

                avio_rl32(pb);

                avio_rl32(pb);



                active_aspect.den = avio_rl16(pb);

                active_aspect.num = avio_rl16(pb);

                active.num        = avio_rl32(pb);

                active.den        = avio_rl32(pb);

                avio_rl32(pb); // nbFieldsPerFrame



                if (active_aspect.num && active_aspect.den &&

                    active.num && active.den) {

                    st->sample_aspect_ratio = av_div_q(active_aspect, active);

                    av_log(s, AV_LOG_TRACE, "vprp %d/%d %d/%d\n",

                            active_aspect.num, active_aspect.den,

                            active.num, active.den);


                size -= 9 * 4;


            avio_skip(pb, size);

            break;

        case MKTAG('s', 't', 'r', 'n'):

            if (s->nb_streams) {

                ret = avi_read_tag(s, s->streams[s->nb_streams - 1], tag, size);

                if (ret < 0)

                    return ret;

                break;


        default:

            if (size > 1000000) {

                char tag_buf[32];

                av_get_codec_tag_string(tag_buf, sizeof(tag_buf), tag);

                av_log(s, AV_LOG_ERROR,

                       "Something went wrong during header parsing, "

                       "tag %s has size %u, "

                       "I will ignore it and try to continue anyway.\n",

                       tag_buf, size);

                if (s->error_recognition & AV_EF_EXPLODE)

                    goto fail;

                avi->movi_list = avio_tell(pb) - 4;

                avi->movi_end  = avi->fsize;

                goto end_of_header;


        /* Do not fail for very large idx1 tags */

        case MKTAG('i', 'd', 'x', '1'):

            /* skip tag */

            size += (size & 1);

            avio_skip(pb, size);

            break;





end_of_header:

    /* check stream number */

    if (stream_index != s->nb_streams - 1) {



fail:

        return AVERROR_INVALIDDATA;




    if (!avi->index_loaded && pb->seekable)

        avi_load_index(s);

    calculate_bitrate(s);

    avi->index_loaded    |= 1;



    if ((ret = guess_ni_flag(s)) < 0)

        return ret;



    avi->non_interleaved |= ret | (s->flags & AVFMT_FLAG_SORT_DTS);



    dict_entry = av_dict_get(s->metadata, "ISFT", NULL, 0);

    if (dict_entry && !strcmp(dict_entry->value, "PotEncoder"))

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            if (   st->codecpar->codec_id == AV_CODEC_ID_MPEG1VIDEO

                || st->codecpar->codec_id == AV_CODEC_ID_MPEG2VIDEO)

                st->need_parsing = AVSTREAM_PARSE_FULL;




    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        if (st->nb_index_entries)

            break;


    // DV-in-AVI cannot be non-interleaved, if set this must be

    // a mis-detection.

    if (avi->dv_demux)

        avi->non_interleaved = 0;

    if (i == s->nb_streams && avi->non_interleaved) {

        av_log(s, AV_LOG_WARNING,

               "Non-interleaved AVI without index, switching to interleaved\n");

        avi->non_interleaved = 0;




    if (avi->non_interleaved) {

        av_log(s, AV_LOG_INFO, "non-interleaved AVI\n");

        clean_index(s);




    ff_metadata_conv_ctx(s, NULL, avi_metadata_conv);

    ff_metadata_conv_ctx(s, NULL, ff_riff_info_conv);



    return 0;
