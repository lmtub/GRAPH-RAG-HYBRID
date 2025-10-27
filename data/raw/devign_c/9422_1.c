static int transcode(AVFormatContext **output_files,

                     int nb_output_files,

                     AVFormatContext **input_files,

                     int nb_input_files,

                     AVStreamMap *stream_maps, int nb_stream_maps)

{

    int ret = 0, i, j, k, n, nb_istreams = 0, nb_ostreams = 0, step;

    AVFormatContext *is, *os;

    AVCodecContext *codec, *icodec;

    AVOutputStream *ost, **ost_table = NULL;

    AVInputStream *ist, **ist_table = NULL;

    AVInputFile *file_table;

    char error[1024];

    int key;

    int want_sdp = 1;

    uint8_t no_packet[MAX_FILES]={0};

    int no_packet_count=0;

    int nb_frame_threshold[AVMEDIA_TYPE_NB]={0};

    int nb_streams[AVMEDIA_TYPE_NB]={0};



    file_table= av_mallocz(nb_input_files * sizeof(AVInputFile));

    if (!file_table)

        goto fail;



    /* input stream init */

    j = 0;

    for(i=0;i<nb_input_files;i++) {

        is = input_files[i];

        file_table[i].ist_index = j;

        file_table[i].nb_streams = is->nb_streams;

        j += is->nb_streams;

    }

    nb_istreams = j;



    ist_table = av_mallocz(nb_istreams * sizeof(AVInputStream *));

    if (!ist_table)

        goto fail;



    for(i=0;i<nb_istreams;i++) {

        ist = av_mallocz(sizeof(AVInputStream));

        if (!ist)

            goto fail;

        ist_table[i] = ist;

    }

    j = 0;

    for(i=0;i<nb_input_files;i++) {

        is = input_files[i];

        for(k=0;k<is->nb_streams;k++) {

            ist = ist_table[j++];

            ist->st = is->streams[k];

            ist->file_index = i;

            ist->index = k;

            ist->discard = 1; /* the stream is discarded by default

                                 (changed later) */



            if (rate_emu) {

                ist->start = av_gettime();

            }

        }

    }



    /* output stream init */

    nb_ostreams = 0;

    for(i=0;i<nb_output_files;i++) {

        os = output_files[i];

        if (!os->nb_streams && !(os->oformat->flags & AVFMT_NOSTREAMS)) {

            av_dump_format(output_files[i], i, output_files[i]->filename, 1);

            fprintf(stderr, "Output file #%d does not contain any stream\n", i);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        nb_ostreams += os->nb_streams;

    }

    if (nb_stream_maps > 0 && nb_stream_maps != nb_ostreams) {

        fprintf(stderr, "Number of stream maps must match number of output streams\n");

        ret = AVERROR(EINVAL);

        goto fail;

    }



    /* Sanity check the mapping args -- do the input files & streams exist? */

    for(i=0;i<nb_stream_maps;i++) {

        int fi = stream_maps[i].file_index;

        int si = stream_maps[i].stream_index;



        if (fi < 0 || fi > nb_input_files - 1 ||

            si < 0 || si > file_table[fi].nb_streams - 1) {

            fprintf(stderr,"Could not find input stream #%d.%d\n", fi, si);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        fi = stream_maps[i].sync_file_index;

        si = stream_maps[i].sync_stream_index;

        if (fi < 0 || fi > nb_input_files - 1 ||

            si < 0 || si > file_table[fi].nb_streams - 1) {

            fprintf(stderr,"Could not find sync stream #%d.%d\n", fi, si);

            ret = AVERROR(EINVAL);

            goto fail;

        }

    }



    ost_table = av_mallocz(sizeof(AVOutputStream *) * nb_ostreams);

    if (!ost_table)

        goto fail;



    for(k=0;k<nb_output_files;k++) {

        os = output_files[k];

        for(i=0;i<os->nb_streams;i++,n++) {

            nb_streams[os->streams[i]->codec->codec_type]++;

        }

    }

    for(step=1<<30; step; step>>=1){

        int found_streams[AVMEDIA_TYPE_NB]={0};

        for(j=0; j<AVMEDIA_TYPE_NB; j++)

            nb_frame_threshold[j] += step;



        for(j=0; j<nb_istreams; j++) {

            int skip=0;

            ist = ist_table[j];

            if(opt_programid){

                int pi,si;

                AVFormatContext *f= input_files[ ist->file_index ];

                skip=1;

                for(pi=0; pi<f->nb_programs; pi++){

                    AVProgram *p= f->programs[pi];

                    if(p->id == opt_programid)

                        for(si=0; si<p->nb_stream_indexes; si++){

                            if(f->streams[ p->stream_index[si] ] == ist->st)

                                skip=0;

                        }

                }

            }

            if (ist->discard && ist->st->discard != AVDISCARD_ALL && !skip

                && nb_frame_threshold[ist->st->codec->codec_type] <= ist->st->codec_info_nb_frames){

                found_streams[ist->st->codec->codec_type]++;

            }

        }

        for(j=0; j<AVMEDIA_TYPE_NB; j++)

            if(found_streams[j] < nb_streams[j])

                nb_frame_threshold[j] -= step;

    }

    n = 0;

    for(k=0;k<nb_output_files;k++) {

        os = output_files[k];

        for(i=0;i<os->nb_streams;i++,n++) {

            int found;

            ost = ost_table[n] = output_streams_for_file[k][i];

            ost->st = os->streams[i];

            if (nb_stream_maps > 0) {

                ost->source_index = file_table[stream_maps[n].file_index].ist_index +

                    stream_maps[n].stream_index;



                /* Sanity check that the stream types match */

                if (ist_table[ost->source_index]->st->codec->codec_type != ost->st->codec->codec_type) {

                    int i= ost->file_index;

                    av_dump_format(output_files[i], i, output_files[i]->filename, 1);

                    fprintf(stderr, "Codec type mismatch for mapping #%d.%d -> #%d.%d\n",

                        stream_maps[n].file_index, stream_maps[n].stream_index,

                        ost->file_index, ost->index);

                    ffmpeg_exit(1);

                }



            } else {

                /* get corresponding input stream index : we select the first one with the right type */

                found = 0;

                for(j=0;j<nb_istreams;j++) {

                    int skip=0;

                    ist = ist_table[j];

                    if(opt_programid){

                        int pi,si;

                        AVFormatContext *f= input_files[ ist->file_index ];

                        skip=1;

                        for(pi=0; pi<f->nb_programs; pi++){

                            AVProgram *p= f->programs[pi];

                            if(p->id == opt_programid)

                                for(si=0; si<p->nb_stream_indexes; si++){

                                    if(f->streams[ p->stream_index[si] ] == ist->st)

                                        skip=0;

                                }

                        }

                    }

                    if (ist->discard && ist->st->discard != AVDISCARD_ALL && !skip &&

                        ist->st->codec->codec_type == ost->st->codec->codec_type &&

                        nb_frame_threshold[ist->st->codec->codec_type] <= ist->st->codec_info_nb_frames) {

                            ost->source_index = j;

                            found = 1;

                            break;

                    }

                }



                if (!found) {

                    if(! opt_programid) {

                        /* try again and reuse existing stream */

                        for(j=0;j<nb_istreams;j++) {

                            ist = ist_table[j];

                            if (   ist->st->codec->codec_type == ost->st->codec->codec_type

                                && ist->st->discard != AVDISCARD_ALL) {

                                ost->source_index = j;

                                found = 1;

                            }

                        }

                    }

                    if (!found) {

                        int i= ost->file_index;

                        av_dump_format(output_files[i], i, output_files[i]->filename, 1);

                        fprintf(stderr, "Could not find input stream matching output stream #%d.%d\n",

                                ost->file_index, ost->index);

                        ffmpeg_exit(1);

                    }

                }

            }

            ist = ist_table[ost->source_index];

            ist->discard = 0;

            ost->sync_ist = (nb_stream_maps > 0) ?

                ist_table[file_table[stream_maps[n].sync_file_index].ist_index +

                         stream_maps[n].sync_stream_index] : ist;

        }

    }



    /* for each output stream, we compute the right encoding parameters */

    for(i=0;i<nb_ostreams;i++) {

        ost = ost_table[i];

        os = output_files[ost->file_index];

        ist = ist_table[ost->source_index];



        codec = ost->st->codec;

        icodec = ist->st->codec;



        if (metadata_streams_autocopy)

            av_metadata_copy(&ost->st->metadata, ist->st->metadata,

                             AV_METADATA_DONT_OVERWRITE);



        ost->st->disposition = ist->st->disposition;

        codec->bits_per_raw_sample= icodec->bits_per_raw_sample;

        codec->chroma_sample_location = icodec->chroma_sample_location;



        if (ost->st->stream_copy) {

            uint64_t extra_size = (uint64_t)icodec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE;



            if (extra_size > INT_MAX)

                goto fail;



            /* if stream_copy is selected, no need to decode or encode */

            codec->codec_id = icodec->codec_id;

            codec->codec_type = icodec->codec_type;



            if(!codec->codec_tag){

                if(   !os->oformat->codec_tag

                   || av_codec_get_id (os->oformat->codec_tag, icodec->codec_tag) == codec->codec_id

                   || av_codec_get_tag(os->oformat->codec_tag, icodec->codec_id) <= 0)

                    codec->codec_tag = icodec->codec_tag;

            }



            codec->bit_rate = icodec->bit_rate;

            codec->rc_max_rate    = icodec->rc_max_rate;

            codec->rc_buffer_size = icodec->rc_buffer_size;

            codec->extradata= av_mallocz(extra_size);

            if (!codec->extradata)

                goto fail;

            memcpy(codec->extradata, icodec->extradata, icodec->extradata_size);

            codec->extradata_size= icodec->extradata_size;

            if(!copy_tb && av_q2d(icodec->time_base)*icodec->ticks_per_frame > av_q2d(ist->st->time_base) && av_q2d(ist->st->time_base) < 1.0/500){

                codec->time_base = icodec->time_base;

                codec->time_base.num *= icodec->ticks_per_frame;

                av_reduce(&codec->time_base.num, &codec->time_base.den,

                          codec->time_base.num, codec->time_base.den, INT_MAX);

            }else

                codec->time_base = ist->st->time_base;

            switch(codec->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                if(audio_volume != 256) {

                    fprintf(stderr,"-acodec copy and -vol are incompatible (frames are not decoded)\n");

                    ffmpeg_exit(1);

                }

                codec->channel_layout = icodec->channel_layout;

                codec->sample_rate = icodec->sample_rate;

                codec->channels = icodec->channels;

                codec->frame_size = icodec->frame_size;

                codec->audio_service_type = icodec->audio_service_type;

                codec->block_align= icodec->block_align;

                if(codec->block_align == 1 && codec->codec_id == CODEC_ID_MP3)

                    codec->block_align= 0;

                if(codec->codec_id == CODEC_ID_AC3)

                    codec->block_align= 0;

                break;

            case AVMEDIA_TYPE_VIDEO:

                codec->pix_fmt = icodec->pix_fmt;

                codec->width = icodec->width;

                codec->height = icodec->height;

                codec->has_b_frames = icodec->has_b_frames;

                break;

            case AVMEDIA_TYPE_SUBTITLE:

                codec->width = icodec->width;

                codec->height = icodec->height;

                break;

            default:

                abort();

            }

        } else {

            switch(codec->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                ost->fifo= av_fifo_alloc(1024);

                if(!ost->fifo)

                    goto fail;

                ost->reformat_pair = MAKE_SFMT_PAIR(AV_SAMPLE_FMT_NONE,AV_SAMPLE_FMT_NONE);

                ost->audio_resample = codec->sample_rate != icodec->sample_rate || audio_sync_method > 1;

                icodec->request_channels = codec->channels;

                ist->decoding_needed = 1;

                ost->encoding_needed = 1;

                ost->resample_sample_fmt  = icodec->sample_fmt;

                ost->resample_sample_rate = icodec->sample_rate;

                ost->resample_channels    = icodec->channels;

                break;

            case AVMEDIA_TYPE_VIDEO:

                if (ost->st->codec->pix_fmt == PIX_FMT_NONE) {

                    fprintf(stderr, "Video pixel format is unknown, stream cannot be encoded\n");

                    ffmpeg_exit(1);

                }

                ost->video_resample = (codec->width != icodec->width   ||

                                       codec->height != icodec->height ||

                        (codec->pix_fmt != icodec->pix_fmt));

                if (ost->video_resample) {

#if !CONFIG_AVFILTER

                    avcodec_get_frame_defaults(&ost->pict_tmp);

                    if(avpicture_alloc((AVPicture*)&ost->pict_tmp, codec->pix_fmt,

                                         codec->width, codec->height)) {

                        fprintf(stderr, "Cannot allocate temp picture, check pix fmt\n");

                        ffmpeg_exit(1);

                    }

                    sws_flags = av_get_int(sws_opts, "sws_flags", NULL);

                    ost->img_resample_ctx = sws_getContext(

                        icodec->width,

                        icodec->height,

                            icodec->pix_fmt,

                            codec->width,

                            codec->height,

                            codec->pix_fmt,

                            sws_flags, NULL, NULL, NULL);

                    if (ost->img_resample_ctx == NULL) {

                        fprintf(stderr, "Cannot get resampling context\n");

                        ffmpeg_exit(1);

                    }



                    ost->original_height = icodec->height;

                    ost->original_width  = icodec->width;

#endif

                    codec->bits_per_raw_sample= 0;

                }

                ost->resample_height = icodec->height;

                ost->resample_width  = icodec->width;

                ost->resample_pix_fmt= icodec->pix_fmt;

                ost->encoding_needed = 1;

                ist->decoding_needed = 1;



#if CONFIG_AVFILTER

                if (configure_filters(ist, ost)) {

                    fprintf(stderr, "Error opening filters!\n");

                    exit(1);

                }

#endif

                break;

            case AVMEDIA_TYPE_SUBTITLE:

                ost->encoding_needed = 1;

                ist->decoding_needed = 1;

                break;

            default:

                abort();

                break;

            }

            /* two pass mode */

            if (ost->encoding_needed &&

                (codec->flags & (CODEC_FLAG_PASS1 | CODEC_FLAG_PASS2))) {

                char logfilename[1024];

                FILE *f;



                snprintf(logfilename, sizeof(logfilename), "%s-%d.log",

                         pass_logfilename_prefix ? pass_logfilename_prefix : DEFAULT_PASS_LOGFILENAME_PREFIX,

                         i);

                if (codec->flags & CODEC_FLAG_PASS1) {

                    f = fopen(logfilename, "wb");

                    if (!f) {

                        fprintf(stderr, "Cannot write log file '%s' for pass-1 encoding: %s\n", logfilename, strerror(errno));

                        ffmpeg_exit(1);

                    }

                    ost->logfile = f;

                } else {

                    char  *logbuffer;

                    size_t logbuffer_size;

                    if (read_file(logfilename, &logbuffer, &logbuffer_size) < 0) {

                        fprintf(stderr, "Error reading log file '%s' for pass-2 encoding\n", logfilename);

                        ffmpeg_exit(1);

                    }

                    codec->stats_in = logbuffer;

                }

            }

        }

        if(codec->codec_type == AVMEDIA_TYPE_VIDEO){

            /* maximum video buffer size is 6-bytes per pixel, plus DPX header size */

            int size= codec->width * codec->height;

            bit_buffer_size= FFMAX(bit_buffer_size, 6*size + 1664);

        }

    }



    if (!bit_buffer)

        bit_buffer = av_malloc(bit_buffer_size);

    if (!bit_buffer) {

        fprintf(stderr, "Cannot allocate %d bytes output buffer\n",

                bit_buffer_size);

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    /* open each encoder */

    for(i=0;i<nb_ostreams;i++) {

        ost = ost_table[i];

        if (ost->encoding_needed) {

            AVCodec *codec = i < nb_output_codecs ? output_codecs[i] : NULL;

            AVCodecContext *dec = ist_table[ost->source_index]->st->codec;

            if (!codec)

                codec = avcodec_find_encoder(ost->st->codec->codec_id);

            if (!codec) {

                snprintf(error, sizeof(error), "Encoder (codec id %d) not found for output stream #%d.%d",

                         ost->st->codec->codec_id, ost->file_index, ost->index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            if (dec->subtitle_header) {

                ost->st->codec->subtitle_header = av_malloc(dec->subtitle_header_size);

                if (!ost->st->codec->subtitle_header) {

                    ret = AVERROR(ENOMEM);

                    goto dump_format;

                }

                memcpy(ost->st->codec->subtitle_header, dec->subtitle_header, dec->subtitle_header_size);

                ost->st->codec->subtitle_header_size = dec->subtitle_header_size;

            }

            if (avcodec_open(ost->st->codec, codec) < 0) {

                snprintf(error, sizeof(error), "Error while opening encoder for output stream #%d.%d - maybe incorrect parameters such as bit_rate, rate, width or height",

                        ost->file_index, ost->index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            extra_size += ost->st->codec->extradata_size;

        }

    }



    /* open each decoder */

    for(i=0;i<nb_istreams;i++) {

        ist = ist_table[i];

        if (ist->decoding_needed) {

            AVCodec *codec = i < nb_input_codecs ? input_codecs[i] : NULL;

            if (!codec)

                codec = avcodec_find_decoder(ist->st->codec->codec_id);

            if (!codec) {

                snprintf(error, sizeof(error), "Decoder (codec id %d) not found for input stream #%d.%d",

                        ist->st->codec->codec_id, ist->file_index, ist->index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            if (avcodec_open(ist->st->codec, codec) < 0) {

                snprintf(error, sizeof(error), "Error while opening decoder for input stream #%d.%d",

                        ist->file_index, ist->index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            //if (ist->st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

            //    ist->st->codec->flags |= CODEC_FLAG_REPEAT_FIELD;

        }

    }



    /* init pts */

    for(i=0;i<nb_istreams;i++) {

        AVStream *st;

        ist = ist_table[i];

        st= ist->st;

        ist->pts = st->avg_frame_rate.num ? - st->codec->has_b_frames*AV_TIME_BASE / av_q2d(st->avg_frame_rate) : 0;

        ist->next_pts = AV_NOPTS_VALUE;

        ist->is_start = 1;

    }



    /* set meta data information from input file if required */

    for (i=0;i<nb_meta_data_maps;i++) {

        AVFormatContext *files[2];

        AVMetadata      **meta[2];

        int j;



#define METADATA_CHECK_INDEX(index, nb_elems, desc)\

        if ((index) < 0 || (index) >= (nb_elems)) {\

            snprintf(error, sizeof(error), "Invalid %s index %d while processing metadata maps\n",\

                     (desc), (index));\

            ret = AVERROR(EINVAL);\

            goto dump_format;\

        }



        int out_file_index = meta_data_maps[i][0].file;

        int in_file_index = meta_data_maps[i][1].file;

        if (in_file_index < 0 || out_file_index < 0)

            continue;

        METADATA_CHECK_INDEX(out_file_index, nb_output_files, "output file")

        METADATA_CHECK_INDEX(in_file_index, nb_input_files, "input file")



        files[0] = output_files[out_file_index];

        files[1] = input_files[in_file_index];



        for (j = 0; j < 2; j++) {

            AVMetaDataMap *map = &meta_data_maps[i][j];



            switch (map->type) {

            case 'g':

                meta[j] = &files[j]->metadata;

                break;

            case 's':

                METADATA_CHECK_INDEX(map->index, files[j]->nb_streams, "stream")

                meta[j] = &files[j]->streams[map->index]->metadata;

                break;

            case 'c':

                METADATA_CHECK_INDEX(map->index, files[j]->nb_chapters, "chapter")

                meta[j] = &files[j]->chapters[map->index]->metadata;

                break;

            case 'p':

                METADATA_CHECK_INDEX(map->index, files[j]->nb_programs, "program")

                meta[j] = &files[j]->programs[map->index]->metadata;

                break;

            }

        }



        av_metadata_copy(meta[0], *meta[1], AV_METADATA_DONT_OVERWRITE);

    }



    /* copy global metadata by default */

    if (metadata_global_autocopy) {



        for (i = 0; i < nb_output_files; i++)

            av_metadata_copy(&output_files[i]->metadata, input_files[0]->metadata,

                             AV_METADATA_DONT_OVERWRITE);

    }



    /* copy chapters according to chapter maps */

    for (i = 0; i < nb_chapter_maps; i++) {

        int infile  = chapter_maps[i].in_file;

        int outfile = chapter_maps[i].out_file;



        if (infile < 0 || outfile < 0)

            continue;

        if (infile >= nb_input_files) {

            snprintf(error, sizeof(error), "Invalid input file index %d in chapter mapping.\n", infile);

            ret = AVERROR(EINVAL);

            goto dump_format;

        }

        if (outfile >= nb_output_files) {

            snprintf(error, sizeof(error), "Invalid output file index %d in chapter mapping.\n",outfile);

            ret = AVERROR(EINVAL);

            goto dump_format;

        }

        copy_chapters(infile, outfile);

    }



    /* copy chapters from the first input file that has them*/

    if (!nb_chapter_maps)

        for (i = 0; i < nb_input_files; i++) {

            if (!input_files[i]->nb_chapters)

                continue;



            for (j = 0; j < nb_output_files; j++)

                if ((ret = copy_chapters(i, j)) < 0)

                    goto dump_format;

            break;

        }



    /* open files and write file headers */

    for(i=0;i<nb_output_files;i++) {

        os = output_files[i];

        if (av_write_header(os) < 0) {

            snprintf(error, sizeof(error), "Could not write header for output file #%d (incorrect codec parameters ?)", i);

            ret = AVERROR(EINVAL);

            goto dump_format;

        }

        if (strcmp(output_files[i]->oformat->name, "rtp")) {

            want_sdp = 0;

        }

    }



 dump_format:

    /* dump the file output parameters - cannot be done before in case

       of stream copy */

    for(i=0;i<nb_output_files;i++) {

        av_dump_format(output_files[i], i, output_files[i]->filename, 1);

    }



    /* dump the stream mapping */

    if (verbose >= 0) {

        fprintf(stderr, "Stream mapping:\n");

        for(i=0;i<nb_ostreams;i++) {

            ost = ost_table[i];

            fprintf(stderr, "  Stream #%d.%d -> #%d.%d",

                    ist_table[ost->source_index]->file_index,

                    ist_table[ost->source_index]->index,

                    ost->file_index,

                    ost->index);

            if (ost->sync_ist != ist_table[ost->source_index])

                fprintf(stderr, " [sync #%d.%d]",

                        ost->sync_ist->file_index,

                        ost->sync_ist->index);

            fprintf(stderr, "\n");

        }

    }



    if (ret) {

        fprintf(stderr, "%s\n", error);

        goto fail;

    }



    if (want_sdp) {

        print_sdp(output_files, nb_output_files);

    }



    if (!using_stdin) {

        if(verbose >= 0)

            fprintf(stderr, "Press [q] to stop encoding\n");

        url_set_interrupt_cb(decode_interrupt_cb);

    }

    term_init();



    timer_start = av_gettime();



    for(; received_sigterm == 0;) {

        int file_index, ist_index;

        AVPacket pkt;

        double ipts_min;

        double opts_min;



    redo:

        ipts_min= 1e100;

        opts_min= 1e100;

        /* if 'q' pressed, exits */

        if (!using_stdin) {

            if (q_pressed)

                break;

            /* read_key() returns 0 on EOF */

            key = read_key();

            if (key == 'q')

                break;

        }



        /* select the stream that we must read now by looking at the

           smallest output pts */

        file_index = -1;

        for(i=0;i<nb_ostreams;i++) {

            double ipts, opts;

            ost = ost_table[i];

            os = output_files[ost->file_index];

            ist = ist_table[ost->source_index];

            if(ist->is_past_recording_time || no_packet[ist->file_index])

                continue;

                opts = ost->st->pts.val * av_q2d(ost->st->time_base);

            ipts = (double)ist->pts;

            if (!file_table[ist->file_index].eof_reached){

                if(ipts < ipts_min) {

                    ipts_min = ipts;

                    if(input_sync ) file_index = ist->file_index;

                }

                if(opts < opts_min) {

                    opts_min = opts;

                    if(!input_sync) file_index = ist->file_index;

                }

            }

            if(ost->frame_number >= max_frames[ost->st->codec->codec_type]){

                file_index= -1;

                break;

            }

        }

        /* if none, if is finished */

        if (file_index < 0) {

            if(no_packet_count){

                no_packet_count=0;

                memset(no_packet, 0, sizeof(no_packet));

                usleep(10000);

                continue;

            }

            break;

        }



        /* finish if limit size exhausted */

        if (limit_filesize != 0 && limit_filesize <= avio_tell(output_files[0]->pb))

            break;



        /* read a frame from it and output it in the fifo */

        is = input_files[file_index];

        ret= av_read_frame(is, &pkt);

        if(ret == AVERROR(EAGAIN)){

            no_packet[file_index]=1;

            no_packet_count++;

            continue;

        }

        if (ret < 0) {

            file_table[file_index].eof_reached = 1;

            if (opt_shortest)

                break;

            else

                continue;

        }



        no_packet_count=0;

        memset(no_packet, 0, sizeof(no_packet));



        if (do_pkt_dump) {

            av_pkt_dump_log2(NULL, AV_LOG_DEBUG, &pkt, do_hex_dump,

                             is->streams[pkt.stream_index]);

        }

        /* the following test is needed in case new streams appear

           dynamically in stream : we ignore them */

        if (pkt.stream_index >= file_table[file_index].nb_streams)

            goto discard_packet;

        ist_index = file_table[file_index].ist_index + pkt.stream_index;

        ist = ist_table[ist_index];

        if (ist->discard)

            goto discard_packet;



        if (pkt.dts != AV_NOPTS_VALUE)

            pkt.dts += av_rescale_q(input_files_ts_offset[ist->file_index], AV_TIME_BASE_Q, ist->st->time_base);

        if (pkt.pts != AV_NOPTS_VALUE)

            pkt.pts += av_rescale_q(input_files_ts_offset[ist->file_index], AV_TIME_BASE_Q, ist->st->time_base);



        if (pkt.stream_index < nb_input_files_ts_scale[file_index]

            && input_files_ts_scale[file_index][pkt.stream_index]){

            if(pkt.pts != AV_NOPTS_VALUE)

                pkt.pts *= input_files_ts_scale[file_index][pkt.stream_index];

            if(pkt.dts != AV_NOPTS_VALUE)

                pkt.dts *= input_files_ts_scale[file_index][pkt.stream_index];

        }



//        fprintf(stderr, "next:%"PRId64" dts:%"PRId64" off:%"PRId64" %d\n", ist->next_pts, pkt.dts, input_files_ts_offset[ist->file_index], ist->st->codec->codec_type);

        if (pkt.dts != AV_NOPTS_VALUE && ist->next_pts != AV_NOPTS_VALUE

            && (is->iformat->flags & AVFMT_TS_DISCONT)) {

            int64_t pkt_dts= av_rescale_q(pkt.dts, ist->st->time_base, AV_TIME_BASE_Q);

            int64_t delta= pkt_dts - ist->next_pts;

            if((FFABS(delta) > 1LL*dts_delta_threshold*AV_TIME_BASE || pkt_dts+1<ist->pts)&& !copy_ts){

                input_files_ts_offset[ist->file_index]-= delta;

                if (verbose > 2)

                    fprintf(stderr, "timestamp discontinuity %"PRId64", new offset= %"PRId64"\n", delta, input_files_ts_offset[ist->file_index]);

                pkt.dts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

                if(pkt.pts != AV_NOPTS_VALUE)

                    pkt.pts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

            }

        }



        /* finish if recording time exhausted */

        if (recording_time != INT64_MAX &&

            av_compare_ts(pkt.pts, ist->st->time_base, recording_time + start_time, (AVRational){1, 1000000}) >= 0) {

            ist->is_past_recording_time = 1;

            goto discard_packet;

        }



        //fprintf(stderr,"read #%d.%d size=%d\n", ist->file_index, ist->index, pkt.size);

        if (output_packet(ist, ist_index, ost_table, nb_ostreams, &pkt) < 0) {



            if (verbose >= 0)

                fprintf(stderr, "Error while decoding stream #%d.%d\n",

                        ist->file_index, ist->index);

            if (exit_on_error)

                ffmpeg_exit(1);

            av_free_packet(&pkt);

            goto redo;

        }



    discard_packet:

        av_free_packet(&pkt);



        /* dump report by using the output first video and audio streams */

        print_report(output_files, ost_table, nb_ostreams, 0);

    }



    /* at the end of stream, we must flush the decoder buffers */

    for(i=0;i<nb_istreams;i++) {

        ist = ist_table[i];

        if (ist->decoding_needed) {

            output_packet(ist, i, ost_table, nb_ostreams, NULL);

        }

    }



    term_exit();



    /* write the trailer if needed and close file */

    for(i=0;i<nb_output_files;i++) {

        os = output_files[i];

        av_write_trailer(os);

    }



    /* dump report by using the first video and audio streams */

    print_report(output_files, ost_table, nb_ostreams, 1);



    /* close each encoder */

    for(i=0;i<nb_ostreams;i++) {

        ost = ost_table[i];

        if (ost->encoding_needed) {

            av_freep(&ost->st->codec->stats_in);

            avcodec_close(ost->st->codec);

        }

#if CONFIG_AVFILTER

        avfilter_graph_free(&ost->graph);

#endif

    }



    /* close each decoder */

    for(i=0;i<nb_istreams;i++) {

        ist = ist_table[i];

        if (ist->decoding_needed) {

            avcodec_close(ist->st->codec);

        }

    }



    /* finished ! */

    ret = 0;



 fail:

    av_freep(&bit_buffer);

    av_free(file_table);



    if (ist_table) {

        for(i=0;i<nb_istreams;i++) {

            ist = ist_table[i];

            av_free(ist);

        }

        av_free(ist_table);

    }

    if (ost_table) {

        for(i=0;i<nb_ostreams;i++) {

            ost = ost_table[i];

            if (ost) {

                if (ost->st->stream_copy)

                    av_freep(&ost->st->codec->extradata);

                if (ost->logfile) {

                    fclose(ost->logfile);

                    ost->logfile = NULL;

                }

                av_fifo_free(ost->fifo); /* works even if fifo is not

                                             initialized but set to zero */

                av_freep(&ost->st->codec->subtitle_header);

                av_free(ost->pict_tmp.data[0]);

                av_free(ost->forced_kf_pts);

                if (ost->video_resample)

                    sws_freeContext(ost->img_resample_ctx);

                if (ost->resample)

                    audio_resample_close(ost->resample);

                if (ost->reformat_ctx)

                    av_audio_convert_free(ost->reformat_ctx);

                av_free(ost);

            }

        }

        av_free(ost_table);

    }

    return ret;

}
