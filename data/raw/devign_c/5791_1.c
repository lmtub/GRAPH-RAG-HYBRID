static void opt_output_file(void *optctx, const char *filename)

{

    OptionsContext *o = optctx;

    AVFormatContext *oc;

    int i, err;

    AVOutputFormat *file_oformat;

    OutputStream *ost;

    InputStream  *ist;



    if (!strcmp(filename, "-"))

        filename = "pipe:";



    err = avformat_alloc_output_context2(&oc, NULL, o->format, filename);

    if (!oc) {

        print_error(filename, err);

        exit_program(1);

    }

    file_oformat= oc->oformat;

    oc->interrupt_callback = int_cb;



    if (!strcmp(file_oformat->name, "ffm") &&

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

            ost = &output_streams[j];

            for (i = 0; i < nb_input_streams; i++) {

                ist = &input_streams[i];

                if(ist->st->codec->codec_type == ost->st->codec->codec_type){

                    ost->sync_ist= ist;

                    ost->source_index= i;

                    ist->discard = 0;

                    break;

                }

            }

            if(!ost->sync_ist){

                av_log(NULL, AV_LOG_FATAL, "Missing %s stream which is required by this ffm\n", av_get_media_type_string(ost->st->codec->codec_type));

                exit_program(1);

            }

        }

    } else if (!o->nb_stream_maps) {

        /* pick the "best" stream of each type */

#define NEW_STREAM(type, index)\

        if (index >= 0) {\

            ost = new_ ## type ## _stream(o, oc);\

            ost->source_index = index;\

            ost->sync_ist     = &input_streams[index];\

            input_streams[index].discard = 0;\

        }



        /* video: highest resolution */

        if (!o->video_disable && oc->oformat->video_codec != CODEC_ID_NONE) {

            int area = 0, idx = -1;

            for (i = 0; i < nb_input_streams; i++) {

                ist = &input_streams[i];

                if (ist->st->codec->codec_type == AVMEDIA_TYPE_VIDEO &&

                    ist->st->codec->width * ist->st->codec->height > area) {

                    area = ist->st->codec->width * ist->st->codec->height;

                    idx = i;

                }

            }

            NEW_STREAM(video, idx);

        }



        /* audio: most channels */

        if (!o->audio_disable && oc->oformat->audio_codec != CODEC_ID_NONE) {

            int channels = 0, idx = -1;

            for (i = 0; i < nb_input_streams; i++) {

                ist = &input_streams[i];

                if (ist->st->codec->codec_type == AVMEDIA_TYPE_AUDIO &&

                    ist->st->codec->channels > channels) {

                    channels = ist->st->codec->channels;

                    idx = i;

                }

            }

            NEW_STREAM(audio, idx);

        }



        /* subtitles: pick first */

        if (!o->subtitle_disable && (oc->oformat->subtitle_codec != CODEC_ID_NONE || subtitle_codec_name)) {

            for (i = 0; i < nb_input_streams; i++)

                if (input_streams[i].st->codec->codec_type == AVMEDIA_TYPE_SUBTITLE) {

                    NEW_STREAM(subtitle, i);

                    break;

                }

        }

        /* do something with data? */

    } else {

        for (i = 0; i < o->nb_stream_maps; i++) {

            StreamMap *map = &o->stream_maps[i];



            if (map->disabled)

                continue;



            ist = &input_streams[input_files[map->file_index].ist_index + map->stream_index];

            if(o->subtitle_disable && ist->st->codec->codec_type == AVMEDIA_TYPE_SUBTITLE)

                continue;

            if(o->   audio_disable && ist->st->codec->codec_type == AVMEDIA_TYPE_AUDIO)

                continue;

            if(o->   video_disable && ist->st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

                continue;

            if(o->    data_disable && ist->st->codec->codec_type == AVMEDIA_TYPE_DATA)

                continue;



            switch (ist->st->codec->codec_type) {

            case AVMEDIA_TYPE_VIDEO:    ost = new_video_stream(o, oc);    break;

            case AVMEDIA_TYPE_AUDIO:    ost = new_audio_stream(o, oc);    break;

            case AVMEDIA_TYPE_SUBTITLE: ost = new_subtitle_stream(o, oc); break;

            case AVMEDIA_TYPE_DATA:     ost = new_data_stream(o, oc);     break;

            case AVMEDIA_TYPE_ATTACHMENT: ost = new_attachment_stream(o, oc); break;

            default:

                av_log(NULL, AV_LOG_FATAL, "Cannot map stream #%d:%d - unsupported type.\n",

                       map->file_index, map->stream_index);

                exit_program(1);

            }



            ost->source_index = input_files[map->file_index].ist_index + map->stream_index;

            ost->sync_ist     = &input_streams[input_files[map->sync_file_index].ist_index +

                                           map->sync_stream_index];

            ist->discard = 0;

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



        ost = new_attachment_stream(o, oc);

        ost->stream_copy               = 0;

        ost->source_index              = -1;

        ost->attachment_filename       = o->attachments[i];

        ost->st->codec->extradata      = attachment;

        ost->st->codec->extradata_size = len;



        p = strrchr(o->attachments[i], '/');

        av_dict_set(&ost->st->metadata, "filename", (p && *p) ? p + 1 : o->attachments[i], AV_DICT_DONT_OVERWRITE);

        avio_close(pb);

    }



    output_files = grow_array(output_files, sizeof(*output_files), &nb_output_files, nb_output_files + 1);

    output_files[nb_output_files - 1].ctx       = oc;

    output_files[nb_output_files - 1].ost_index = nb_output_streams - oc->nb_streams;

    output_files[nb_output_files - 1].recording_time = o->recording_time;

    output_files[nb_output_files - 1].start_time     = o->start_time;

    output_files[nb_output_files - 1].limit_filesize = o->limit_filesize;

    av_dict_copy(&output_files[nb_output_files - 1].opts, format_opts, 0);



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

                              &output_files[nb_output_files - 1].opts)) < 0) {

            print_error(filename, err);

            exit_program(1);

        }

    }



    if (o->mux_preload) {

        uint8_t buf[64];

        snprintf(buf, sizeof(buf), "%d", (int)(o->mux_preload*AV_TIME_BASE));

        av_dict_set(&output_files[nb_output_files - 1].opts, "preload", buf, 0);

    }

    oc->max_delay = (int)(o->mux_max_delay * AV_TIME_BASE);



    if (loop_output >= 0) {

        av_log(NULL, AV_LOG_WARNING, "-loop_output is deprecated, use -loop\n");

        oc->loop_output = loop_output;

    }



    /* copy metadata */

    for (i = 0; i < o->nb_metadata_map; i++) {

        char *p;

        int in_file_index = strtol(o->metadata_map[i].u.str, &p, 0);



        if (in_file_index < 0)

            continue;

        if (in_file_index >= nb_input_files) {

            av_log(NULL, AV_LOG_FATAL, "Invalid input file index %d while processing metadata maps\n", in_file_index);

            exit_program(1);

        }

        copy_metadata(o->metadata_map[i].specifier, *p ? p + 1 : p, oc, input_files[in_file_index].ctx, o);

    }



    /* copy chapters */

    if (o->chapters_input_file >= nb_input_files) {

        if (o->chapters_input_file == INT_MAX) {

            /* copy chapters from the first input file that has them*/

            o->chapters_input_file = -1;

            for (i = 0; i < nb_input_files; i++)

                if (input_files[i].ctx->nb_chapters) {

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

        copy_chapters(&input_files[o->chapters_input_file], &output_files[nb_output_files - 1],

                      !o->metadata_chapters_manual);



    /* copy global metadata by default */

    if (!o->metadata_global_manual && nb_input_files){

        av_dict_copy(&oc->metadata, input_files[0].ctx->metadata,

                     AV_DICT_DONT_OVERWRITE);

        if(o->recording_time != INT64_MAX)

            av_dict_set(&oc->metadata, "duration", NULL, 0);

    }

    if (!o->metadata_streams_manual)

        for (i = output_files[nb_output_files - 1].ost_index; i < nb_output_streams; i++) {

            InputStream *ist;

            if (output_streams[i].source_index < 0)         /* this is true e.g. for attached files */

                continue;

            ist = &input_streams[output_streams[i].source_index];

            av_dict_copy(&output_streams[i].st->metadata, ist->st->metadata, AV_DICT_DONT_OVERWRITE);

        }



    /* process manually set metadata */

    for (i = 0; i < o->nb_metadata; i++) {

        AVDictionary **m;

        char type, *val;

        const char *stream_spec;

        int index = 0, j, ret;



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

            printf("ret %d, stream_spec %s\n", ret, stream_spec);

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



    reset_options(o, 0);

}
