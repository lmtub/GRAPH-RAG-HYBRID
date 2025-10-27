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



    oc = avformat_alloc_context();

    if (!oc) {

        print_error(filename, AVERROR(ENOMEM));

        exit_program(1);

    }



    if (last_asked_format) {

        file_oformat = av_guess_format(last_asked_format, NULL, NULL);

        if (!file_oformat) {

            fprintf(stderr, "Requested output format '%s' is not a suitable output format\n", last_asked_format);

            exit_program(1);

        }

        last_asked_format = NULL;

    } else {

        file_oformat = av_guess_format(NULL, filename, NULL);

        if (!file_oformat) {

            fprintf(stderr, "Unable to find a suitable output format for '%s'\n",

                    filename);

            exit_program(1);

        }

    }



    oc->oformat = file_oformat;

    av_strlcpy(oc->filename, filename, sizeof(oc->filename));



    if (!strcmp(file_oformat->name, "ffm") &&

        av_strstart(filename, "http:", NULL)) {

        /* special case for files sent to avserver: we get the stream

           parameters from avserver */

        int err = read_avserver_streams(oc, filename);

        if (err < 0) {

            print_error(filename, err);

            exit_program(1);

        }

    } else if (!o->nb_stream_maps) {

        /* pick the "best" stream of each type */

#define NEW_STREAM(type, index)\

        if (index >= 0) {\

            ost = new_ ## type ## _stream(oc);\

            ost->source_index = index;\

            ost->sync_ist     = &input_streams[index];\

            input_streams[index].discard = 0;\

        }



        /* video: highest resolution */

        if (!video_disable && oc->oformat->video_codec != CODEC_ID_NONE) {

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

        if (!audio_disable && oc->oformat->audio_codec != CODEC_ID_NONE) {

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

        if (!subtitle_disable && oc->oformat->subtitle_codec != CODEC_ID_NONE) {

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

            switch (ist->st->codec->codec_type) {

            case AVMEDIA_TYPE_VIDEO:    ost = new_video_stream(oc);    break;

            case AVMEDIA_TYPE_AUDIO:    ost = new_audio_stream(oc);    break;

            case AVMEDIA_TYPE_SUBTITLE: ost = new_subtitle_stream(oc); break;

            case AVMEDIA_TYPE_DATA:     ost = new_data_stream(oc);     break;

            default:

                av_log(NULL, AV_LOG_ERROR, "Cannot map stream #%d.%d - unsupported type.\n",

                       map->file_index, map->stream_index);

                exit_program(1);

            }



            ost->source_index = input_files[map->file_index].ist_index + map->stream_index;

            ost->sync_ist = &input_streams[input_files[map->sync_file_index].ist_index +

                                           map->sync_stream_index];

            ist->discard = 0;

        }

    }



    av_dict_copy(&oc->metadata, metadata, 0);

    av_dict_free(&metadata);





    output_files = grow_array(output_files, sizeof(*output_files), &nb_output_files, nb_output_files + 1);

    output_files[nb_output_files - 1].ctx       = oc;

    output_files[nb_output_files - 1].ost_index = nb_output_streams - oc->nb_streams;

    output_files[nb_output_files - 1].recording_time = o->recording_time;

    output_files[nb_output_files - 1].start_time     = o->start_time;

    output_files[nb_output_files - 1].limit_filesize = limit_filesize;

    av_dict_copy(&output_files[nb_output_files - 1].opts, format_opts, 0);



    /* check filename in case of an image number is expected */

    if (oc->oformat->flags & AVFMT_NEEDNUMBER) {

        if (!av_filename_number_test(oc->filename)) {

            print_error(oc->filename, AVERROR(EINVAL));

            exit_program(1);

        }

    }



    if (!(oc->oformat->flags & AVFMT_NOFILE)) {

        /* test if it already exists to avoid loosing precious files */

        if (!file_overwrite &&

            (strchr(filename, ':') == NULL ||

             filename[1] == ':' ||

             av_strstart(filename, "file:", NULL))) {

            if (avio_check(filename, 0) == 0) {

                if (!using_stdin) {

                    fprintf(stderr,"File '%s' already exists. Overwrite ? [y/N] ", filename);

                    fflush(stderr);

                    if (!read_yesno()) {

                        fprintf(stderr, "Not overwriting - exiting\n");

                        exit_program(1);

                    }

                }

                else {

                    fprintf(stderr,"File '%s' already exists. Exiting.\n", filename);

                    exit_program(1);

                }

            }

        }



        /* open the file */

        if ((err = avio_open(&oc->pb, filename, AVIO_FLAG_WRITE)) < 0) {

            print_error(filename, err);

            exit_program(1);

        }

    }



    oc->preload= (int)(mux_preload*AV_TIME_BASE);

    oc->max_delay= (int)(mux_max_delay*AV_TIME_BASE);

    oc->flags |= AVFMT_FLAG_NONBLOCK;



    /* copy chapters */

    if (chapters_input_file >= nb_input_files) {

        if (chapters_input_file == INT_MAX) {

            /* copy chapters from the first input file that has them*/

            chapters_input_file = -1;

            for (i = 0; i < nb_input_files; i++)

                if (input_files[i].ctx->nb_chapters) {

                    chapters_input_file = i;

                    break;

                }

        } else {

            av_log(NULL, AV_LOG_ERROR, "Invalid input file index %d in chapter mapping.\n",

                   chapters_input_file);

            exit_program(1);

        }

    }

    if (chapters_input_file >= 0)

        copy_chapters(&input_files[chapters_input_file], &output_files[nb_output_files - 1]);



    /* copy metadata */

    for (i = 0; i < nb_meta_data_maps; i++) {

        AVFormatContext *files[2];

        AVDictionary    **meta[2];

        int j;



#define METADATA_CHECK_INDEX(index, nb_elems, desc)\

        if ((index) < 0 || (index) >= (nb_elems)) {\

            av_log(NULL, AV_LOG_ERROR, "Invalid %s index %d while processing metadata maps\n",\

                     (desc), (index));\

            exit_program(1);\

        }



        int in_file_index = meta_data_maps[i][1].file;

        if (in_file_index < 0)

            continue;

        METADATA_CHECK_INDEX(in_file_index, nb_input_files, "input file")



        files[0] = oc;

        files[1] = input_files[in_file_index].ctx;



        for (j = 0; j < 2; j++) {

            MetadataMap *map = &meta_data_maps[i][j];



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



        av_dict_copy(meta[0], *meta[1], AV_DICT_DONT_OVERWRITE);

    }



    /* copy global metadata by default */

    if (metadata_global_autocopy && nb_input_files)

        av_dict_copy(&oc->metadata, input_files[0].ctx->metadata,

                     AV_DICT_DONT_OVERWRITE);

    if (metadata_streams_autocopy)

        for (i = output_files[nb_output_files - 1].ost_index; i < nb_output_streams; i++) {

            InputStream *ist = &input_streams[output_streams[i].source_index];

            av_dict_copy(&output_streams[i].st->metadata, ist->st->metadata, AV_DICT_DONT_OVERWRITE);

        }



    frame_rate    = (AVRational){0, 0};

    frame_width   = 0;

    frame_height  = 0;

    audio_sample_rate = 0;

    audio_channels    = 0;

    audio_sample_fmt  = AV_SAMPLE_FMT_NONE;

    chapters_input_file = INT_MAX;

    limit_filesize = UINT64_MAX;



    av_freep(&meta_data_maps);

    nb_meta_data_maps = 0;

    metadata_global_autocopy   = 1;

    metadata_streams_autocopy  = 1;

    metadata_chapters_autocopy = 1;

    av_freep(&streamid_map);

    nb_streamid_map = 0;



    av_dict_free(&codec_names);



    av_freep(&forced_key_frames);

    reset_options(o);

}
