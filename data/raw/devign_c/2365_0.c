static int open_output_file(OptionsContext *o, const char *filename)

{

    AVFormatContext *oc;

    int i, j, err;

    AVOutputFormat *file_oformat;

    OutputFile *of;

    OutputStream *ost;

    InputStream  *ist;

    AVDictionary *unused_opts = NULL;

    AVDictionaryEntry *e = NULL;



    if (configure_complex_filters() < 0) {

        av_log(NULL, AV_LOG_FATAL, "Error configuring filters.\n");

        exit_program(1);

    }



    if (o->stop_time != INT64_MAX && o->recording_time != INT64_MAX) {

        o->stop_time = INT64_MAX;

        av_log(NULL, AV_LOG_WARNING, "-t and -to cannot be used together; using -t.\n");

    }



    if (o->stop_time != INT64_MAX && o->recording_time == INT64_MAX) {

        int64_t start_time = o->start_time == AV_NOPTS_VALUE ? 0 : o->start_time;

        if (o->stop_time <= start_time) {

            av_log(NULL, AV_LOG_WARNING, "-to value smaller than -ss; ignoring -to.\n");

            o->stop_time = INT64_MAX;

        } else {

            o->recording_time = o->stop_time - start_time;

        }

    }



    GROW_ARRAY(output_files, nb_output_files);

    of = av_mallocz(sizeof(*of));

    if (!of)

        exit_program(1);

    output_files[nb_output_files - 1] = of;



    of->ost_index      = nb_output_streams;

    of->recording_time = o->recording_time;

    of->start_time     = o->start_time;

    of->limit_filesize = o->limit_filesize;

    of->shortest       = o->shortest;

    av_dict_copy(&of->opts, o->g->format_opts, 0);



    if (!strcmp(filename, "-"))

        filename = "pipe:";



    err = avformat_alloc_output_context2(&oc, NULL, o->format, filename);

    if (!oc) {

        print_error(filename, err);

        exit_program(1);

    }



    of->ctx = oc;

    if (o->recording_time != INT64_MAX)

        oc->duration = o->recording_time;



    file_oformat= oc->oformat;

    oc->interrupt_callback = int_cb;



    /* create streams for all unlabeled output pads */

    for (i = 0; i < nb_filtergraphs; i++) {

        FilterGraph *fg = filtergraphs[i];

        for (j = 0; j < fg->nb_outputs; j++) {

            OutputFilter *ofilter = fg->outputs[j];



            if (!ofilter->out_tmp || ofilter->out_tmp->name)

                continue;



            switch (avfilter_pad_get_type(ofilter->out_tmp->filter_ctx->output_pads,

                                          ofilter->out_tmp->pad_idx)) {

            case AVMEDIA_TYPE_VIDEO:    o->video_disable    = 1; break;

            case AVMEDIA_TYPE_AUDIO:    o->audio_disable    = 1; break;

            case AVMEDIA_TYPE_SUBTITLE: o->subtitle_disable = 1; break;

            }

            init_output_filter(ofilter, o, oc);

        }

    }



    /* ffserver seeking with date=... needs a date reference */

    if (!strcmp(file_oformat->name, "ffm") &&

        av_strstart(filename, "http:", NULL)) {

        int err = parse_option(o, "metadata", "creation_time=now", options);

        if (err < 0) {

            print_error(filename, err);

            exit_program(1);

        }

    }



    if (!strcmp(file_oformat->name, "ffm") && !override_ffserver &&

        av_strstart(filename, "http:", NULL)) {

        int j;

        /* special case for files sent to ffserver: we get the stream

           parameters from ffserver */

        int err = read_ffserver_streams(o, oc, filename);

        if (err < 0) {

            print_error(filename, err);

            exit_program(1);

        }

        for(j = nb_output_streams - oc->nb_streams; j < nb_output_streams; j++) {

            ost = output_streams[j];

            for (i = 0; i < nb_input_streams; i++) {

                ist = input_streams[i];

                if(ist->st->codec->codec_type == ost->st->codec->codec_type){

                    ost->sync_ist= ist;

                    ost->source_index= i;

                    if(ost->st->codec->codec_type == AVMEDIA_TYPE_AUDIO) ost->avfilter = av_strdup("anull");

                    if(ost->st->codec->codec_type == AVMEDIA_TYPE_VIDEO) ost->avfilter = av_strdup("null");

                    ist->discard = 0;

                    ist->st->discard = AVDISCARD_NONE;

                    break;

                }

            }

            if(!ost->sync_ist){

                av_log(NULL, AV_LOG_FATAL, "Missing %s stream which is required by this ffm\n", av_get_media_type_string(ost->st->codec->codec_type));

                exit_program(1);

            }

        }

    } else if (!o->nb_stream_maps) {

        char *subtitle_codec_name = NULL;

        /* pick the "best" stream of each type */



        /* video: highest resolution */

        if (!o->video_disable && oc->oformat->video_codec != AV_CODEC_ID_NONE) {

            int area = 0, idx = -1;

            int qcr = avformat_query_codec(oc->oformat, oc->oformat->video_codec, 0);

            for (i = 0; i < nb_input_streams; i++) {

                int new_area;

                ist = input_streams[i];

                new_area = ist->st->codec->width * ist->st->codec->height;

                if((qcr!=MKTAG('A', 'P', 'I', 'C')) && (ist->st->disposition & AV_DISPOSITION_ATTACHED_PIC))

                    new_area = 1;

                if (ist->st->codec->codec_type == AVMEDIA_TYPE_VIDEO &&

                    new_area > area) {

                    if((qcr==MKTAG('A', 'P', 'I', 'C')) && !(ist->st->disposition & AV_DISPOSITION_ATTACHED_PIC))

                        continue;

                    area = new_area;

                    idx = i;

                }

            }

            if (idx >= 0)

                new_video_stream(o, oc, idx);

        }



        /* audio: most channels */

        if (!o->audio_disable && oc->oformat->audio_codec != AV_CODEC_ID_NONE) {

            int channels = 0, idx = -1;

            for (i = 0; i < nb_input_streams; i++) {

                ist = input_streams[i];

                if (ist->st->codec->codec_type == AVMEDIA_TYPE_AUDIO &&

                    ist->st->codec->channels > channels) {

                    channels = ist->st->codec->channels;

                    idx = i;

                }

            }

            if (idx >= 0)

                new_audio_stream(o, oc, idx);

        }



        /* subtitles: pick first */

        MATCH_PER_TYPE_OPT(codec_names, str, subtitle_codec_name, oc, "s");

        if (!o->subtitle_disable && (oc->oformat->subtitle_codec != AV_CODEC_ID_NONE || subtitle_codec_name)) {

            for (i = 0; i < nb_input_streams; i++)

                if (input_streams[i]->st->codec->codec_type == AVMEDIA_TYPE_SUBTITLE) {

                    new_subtitle_stream(o, oc, i);

                    break;

                }

        }

        /* do something with data? */

    } else {

        for (i = 0; i < o->nb_stream_maps; i++) {

            StreamMap *map = &o->stream_maps[i];



            if (map->disabled)

                continue;



            if (map->linklabel) {

                FilterGraph *fg;

                OutputFilter *ofilter = NULL;

                int j, k;



                for (j = 0; j < nb_filtergraphs; j++) {

                    fg = filtergraphs[j];

                    for (k = 0; k < fg->nb_outputs; k++) {

                        AVFilterInOut *out = fg->outputs[k]->out_tmp;

                        if (out && !strcmp(out->name, map->linklabel)) {

                            ofilter = fg->outputs[k];

                            goto loop_end;

                        }

                    }

                }

loop_end:

                if (!ofilter) {

                    av_log(NULL, AV_LOG_FATAL, "Output with label '%s' does not exist "

                           "in any defined filter graph, or was already used elsewhere.\n", map->linklabel);

                    exit_program(1);

                }

                init_output_filter(ofilter, o, oc);

            } else {

                int src_idx = input_files[map->file_index]->ist_index + map->stream_index;



                ist = input_streams[input_files[map->file_index]->ist_index + map->stream_index];

                if(o->subtitle_disable && ist->st->codec->codec_type == AVMEDIA_TYPE_SUBTITLE)

                    continue;

                if(o->   audio_disable && ist->st->codec->codec_type == AVMEDIA_TYPE_AUDIO)

                    continue;

                if(o->   video_disable && ist->st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

                    continue;

                if(o->    data_disable && ist->st->codec->codec_type == AVMEDIA_TYPE_DATA)

                    continue;



                switch (ist->st->codec->codec_type) {

                case AVMEDIA_TYPE_VIDEO:      ost = new_video_stream     (o, oc, src_idx); break;

                case AVMEDIA_TYPE_AUDIO:      ost = new_audio_stream     (o, oc, src_idx); break;

                case AVMEDIA_TYPE_SUBTITLE:   ost = new_subtitle_stream  (o, oc, src_idx); break;

                case AVMEDIA_TYPE_DATA:       ost = new_data_stream      (o, oc, src_idx); break;

                case AVMEDIA_TYPE_ATTACHMENT: ost = new_attachment_stream(o, oc, src_idx); break;

                default:

                    av_log(NULL, AV_LOG_FATAL, "Cannot map stream #%d:%d - unsupported type.\n",

                           map->file_index, map->stream_index);

                    exit_program(1);

                }

            }

        }

    }



    /* handle attached files */

    for (i = 0; i < o->nb_attachments; i++) {

        AVIOContext *pb;

        uint8_t *attachment;

        const char *p;

        int64_t len;



        if ((err = avio_open2(&pb, o->attachments[i], AVIO_FLAG_READ, &int_cb, NULL)) < 0) {

            av_log(NULL, AV_LOG_FATAL, "Could not open attachment file %s.\n",

                   o->attachments[i]);

            exit_program(1);

        }

        if ((len = avio_size(pb)) <= 0) {

            av_log(NULL, AV_LOG_FATAL, "Could not get size of the attachment %s.\n",

                   o->attachments[i]);

            exit_program(1);

        }

        if (!(attachment = av_malloc(len))) {

            av_log(NULL, AV_LOG_FATAL, "Attachment %s too large to fit into memory.\n",

                   o->attachments[i]);

            exit_program(1);

        }

        avio_read(pb, attachment, len);



        ost = new_attachment_stream(o, oc, -1);

        ost->stream_copy               = 0;

        ost->attachment_filename       = o->attachments[i];

        ost->finished                  = 1;

        ost->st->codec->extradata      = attachment;

        ost->st->codec->extradata_size = len;



        p = strrchr(o->attachments[i], '/');

        av_dict_set(&ost->st->metadata, "filename", (p && *p) ? p + 1 : o->attachments[i], AV_DICT_DONT_OVERWRITE);

        avio_close(pb);

    }



    for (i = nb_output_streams - oc->nb_streams; i < nb_output_streams; i++) { //for all streams of this output file

        AVDictionaryEntry *e;

        ost = output_streams[i];



        if ((ost->stream_copy || ost->attachment_filename)

            && (e = av_dict_get(o->g->codec_opts, "flags", NULL, AV_DICT_IGNORE_SUFFIX))

            && (!e->key[5] || check_stream_specifier(oc, ost->st, e->key+6)))

            if (av_opt_set(ost->st->codec, "flags", e->value, 0) < 0)

                exit_program(1);

    }



    /* check if all codec options have been used */

    unused_opts = strip_specifiers(o->g->codec_opts);

    for (i = of->ost_index; i < nb_output_streams; i++) {

        e = NULL;

        while ((e = av_dict_get(output_streams[i]->opts, "", e,

                                AV_DICT_IGNORE_SUFFIX)))

            av_dict_set(&unused_opts, e->key, NULL, 0);

    }



    e = NULL;

    while ((e = av_dict_get(unused_opts, "", e, AV_DICT_IGNORE_SUFFIX))) {

        const AVClass *class = avcodec_get_class();

        const AVOption *option = av_opt_find(&class, e->key, NULL, 0,

                                             AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ);

        if (!option)

            continue;

        if (!(option->flags & AV_OPT_FLAG_ENCODING_PARAM)) {

            av_log(NULL, AV_LOG_ERROR, "Codec AVOption %s (%s) specified for "

                   "output file #%d (%s) is not an encoding option.\n", e->key,

                   option->help ? option->help : "", nb_output_files - 1,

                   filename);

            exit_program(1);

        }



        // gop_timecode is injected by generic code but not always used

        if (!strcmp(e->key, "gop_timecode"))

            continue;



        av_log(NULL, AV_LOG_WARNING, "Codec AVOption %s (%s) specified for "

               "output file #%d (%s) has not been used for any stream. The most "

               "likely reason is either wrong type (e.g. a video option with "

               "no video streams) or that it is a private option of some encoder "

               "which was not actually used for any stream.\n", e->key,

               option->help ? option->help : "", nb_output_files - 1, filename);

    }

    av_dict_free(&unused_opts);



    /* check filename in case of an image number is expected */

    if (oc->oformat->flags & AVFMT_NEEDNUMBER) {

        if (!av_filename_number_test(oc->filename)) {

            print_error(oc->filename, AVERROR(EINVAL));

            exit_program(1);

        }

    }



    if (!(oc->oformat->flags & AVFMT_NOFILE)) {

        /* test if it already exists to avoid losing precious files */

        assert_file_overwrite(filename);



        /* open the file */

        if ((err = avio_open2(&oc->pb, filename, AVIO_FLAG_WRITE,

                              &oc->interrupt_callback,

                              &of->opts)) < 0) {

            print_error(filename, err);

            exit_program(1);

        }

    } else if (strcmp(oc->oformat->name, "image2")==0 && !av_filename_number_test(filename))

        assert_file_overwrite(filename);



    if (o->mux_preload) {

        uint8_t buf[64];

        snprintf(buf, sizeof(buf), "%d", (int)(o->mux_preload*AV_TIME_BASE));

        av_dict_set(&of->opts, "preload", buf, 0);

    }

    oc->max_delay = (int)(o->mux_max_delay * AV_TIME_BASE);



    /* copy metadata */

    for (i = 0; i < o->nb_metadata_map; i++) {

        char *p;

        int in_file_index = strtol(o->metadata_map[i].u.str, &p, 0);



        if (in_file_index >= nb_input_files) {

            av_log(NULL, AV_LOG_FATAL, "Invalid input file index %d while processing metadata maps\n", in_file_index);

            exit_program(1);

        }

        copy_metadata(o->metadata_map[i].specifier, *p ? p + 1 : p, oc,

                      in_file_index >= 0 ?

                      input_files[in_file_index]->ctx : NULL, o);

    }



    /* copy chapters */

    if (o->chapters_input_file >= nb_input_files) {

        if (o->chapters_input_file == INT_MAX) {

            /* copy chapters from the first input file that has them*/

            o->chapters_input_file = -1;

            for (i = 0; i < nb_input_files; i++)

                if (input_files[i]->ctx->nb_chapters) {

                    o->chapters_input_file = i;

                    break;

                }

        } else {

            av_log(NULL, AV_LOG_FATAL, "Invalid input file index %d in chapter mapping.\n",

                   o->chapters_input_file);

            exit_program(1);

        }

    }

    if (o->chapters_input_file >= 0)

        copy_chapters(input_files[o->chapters_input_file], of,

                      !o->metadata_chapters_manual);



    /* copy global metadata by default */

    if (!o->metadata_global_manual && nb_input_files){

        av_dict_copy(&oc->metadata, input_files[0]->ctx->metadata,

                     AV_DICT_DONT_OVERWRITE);

        if(o->recording_time != INT64_MAX)

            av_dict_set(&oc->metadata, "duration", NULL, 0);

        av_dict_set(&oc->metadata, "creation_time", NULL, 0);

    }

    if (!o->metadata_streams_manual)

        for (i = of->ost_index; i < nb_output_streams; i++) {

            InputStream *ist;

            if (output_streams[i]->source_index < 0)         /* this is true e.g. for attached files */

                continue;

            ist = input_streams[output_streams[i]->source_index];

            av_dict_copy(&output_streams[i]->st->metadata, ist->st->metadata, AV_DICT_DONT_OVERWRITE);

        }



    /* process manually set metadata */

    for (i = 0; i < o->nb_metadata; i++) {

        AVDictionary **m;

        char type, *val;

        const char *stream_spec;

        int index = 0, j, ret = 0;



        val = strchr(o->metadata[i].u.str, '=');

        if (!val) {

            av_log(NULL, AV_LOG_FATAL, "No '=' character in metadata string %s.\n",

                   o->metadata[i].u.str);

            exit_program(1);

        }

        *val++ = 0;



        parse_meta_type(o->metadata[i].specifier, &type, &index, &stream_spec);

        if (type == 's') {

            for (j = 0; j < oc->nb_streams; j++) {

                if ((ret = check_stream_specifier(oc, oc->streams[j], stream_spec)) > 0) {

                    av_dict_set(&oc->streams[j]->metadata, o->metadata[i].u.str, *val ? val : NULL, 0);

                } else if (ret < 0)

                    exit_program(1);

            }

        }

        else {

            switch (type) {

            case 'g':

                m = &oc->metadata;

                break;

            case 'c':

                if (index < 0 || index >= oc->nb_chapters) {

                    av_log(NULL, AV_LOG_FATAL, "Invalid chapter index %d in metadata specifier.\n", index);

                    exit_program(1);

                }

                m = &oc->chapters[index]->metadata;

                break;

            default:

                av_log(NULL, AV_LOG_FATAL, "Invalid metadata specifier %s.\n", o->metadata[i].specifier);

                exit_program(1);

            }

            av_dict_set(m, o->metadata[i].u.str, *val ? val : NULL, 0);

        }

    }



    return 0;

}
