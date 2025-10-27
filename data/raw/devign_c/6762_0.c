static int dash_init(AVFormatContext *s)

{

    DASHContext *c = s->priv_data;

    int ret = 0, i;

    char *ptr;

    char basename[1024];



    if (c->single_file_name)

        c->single_file = 1;

    if (c->single_file)

        c->use_template = 0;



    av_strlcpy(c->dirname, s->filename, sizeof(c->dirname));

    ptr = strrchr(c->dirname, '/');

    if (ptr) {

        av_strlcpy(basename, &ptr[1], sizeof(basename));

        ptr[1] = '\0';

    } else {

        c->dirname[0] = '\0';

        av_strlcpy(basename, s->filename, sizeof(basename));

    }



    ptr = strrchr(basename, '.');

    if (ptr)

        *ptr = '\0';



    c->streams = av_mallocz(sizeof(*c->streams) * s->nb_streams);

    if (!c->streams)

        return AVERROR(ENOMEM);



    if ((ret = parse_adaptation_sets(s)) < 0)

        return ret;



    for (i = 0; i < s->nb_streams; i++) {

        OutputStream *os = &c->streams[i];

        AdaptationSet *as = &c->as[os->as_idx - 1];

        AVFormatContext *ctx;

        AVStream *st;

        AVDictionary *opts = NULL;

        char filename[1024];



        os->bit_rate = s->streams[i]->codecpar->bit_rate;

        if (os->bit_rate) {

            snprintf(os->bandwidth_str, sizeof(os->bandwidth_str),

                     " bandwidth=\"%d\"", os->bit_rate);

        } else {

            int level = s->strict_std_compliance >= FF_COMPLIANCE_STRICT ?

                        AV_LOG_ERROR : AV_LOG_WARNING;

            av_log(s, level, "No bit rate set for stream %d\n", i);

            if (s->strict_std_compliance >= FF_COMPLIANCE_STRICT)

                return AVERROR(EINVAL);

        }



        // copy AdaptationSet language and role from stream metadata

        dict_copy_entry(&as->metadata, s->streams[i]->metadata, "language");

        dict_copy_entry(&as->metadata, s->streams[i]->metadata, "role");



        ctx = avformat_alloc_context();

        if (!ctx)

            return AVERROR(ENOMEM);



        // choose muxer based on codec: webm for VP8/9 and opus, mp4 otherwise

        // note: os->format_name is also used as part of the mimetype of the

        //       representation, e.g. video/<format_name>

        if (s->streams[i]->codecpar->codec_id == AV_CODEC_ID_VP8 ||

            s->streams[i]->codecpar->codec_id == AV_CODEC_ID_VP9 ||

            s->streams[i]->codecpar->codec_id == AV_CODEC_ID_OPUS ||

            s->streams[i]->codecpar->codec_id == AV_CODEC_ID_VORBIS) {

            snprintf(os->format_name, sizeof(os->format_name), "webm");

        } else {

            snprintf(os->format_name, sizeof(os->format_name), "mp4");

        }

        ctx->oformat = av_guess_format(os->format_name, NULL, NULL);

        if (!ctx->oformat)

            return AVERROR_MUXER_NOT_FOUND;

        os->ctx = ctx;

        ctx->interrupt_callback = s->interrupt_callback;

        ctx->opaque             = s->opaque;

        ctx->io_close           = s->io_close;

        ctx->io_open            = s->io_open;



        if (!(st = avformat_new_stream(ctx, NULL)))

            return AVERROR(ENOMEM);

        avcodec_parameters_copy(st->codecpar, s->streams[i]->codecpar);

        st->sample_aspect_ratio = s->streams[i]->sample_aspect_ratio;

        st->time_base = s->streams[i]->time_base;

        st->avg_frame_rate = s->streams[i]->avg_frame_rate;

        ctx->avoid_negative_ts = s->avoid_negative_ts;

        ctx->flags = s->flags;



        if ((ret = avio_open_dyn_buf(&ctx->pb)) < 0)

            return ret;



        if (c->single_file) {

            if (c->single_file_name)

                ff_dash_fill_tmpl_params(os->initfile, sizeof(os->initfile), c->single_file_name, i, 0, os->bit_rate, 0);

            else

                snprintf(os->initfile, sizeof(os->initfile), "%s-stream%d.m4s", basename, i);

        } else {

            ff_dash_fill_tmpl_params(os->initfile, sizeof(os->initfile), c->init_seg_name, i, 0, os->bit_rate, 0);

        }

        snprintf(filename, sizeof(filename), "%s%s", c->dirname, os->initfile);

        ret = s->io_open(s, &os->out, filename, AVIO_FLAG_WRITE, NULL);

        if (ret < 0)

            return ret;

        os->init_start_pos = 0;



        if (!strcmp(os->format_name, "mp4")) {

            av_dict_set(&opts, "movflags", "frag_custom+dash+delay_moov", 0);

        } else {

            av_dict_set_int(&opts, "cluster_time_limit", c->min_seg_duration / 1000, 0);

            av_dict_set_int(&opts, "cluster_size_limit", 5 * 1024 * 1024, 0); // set a large cluster size limit

            av_dict_set_int(&opts, "dash", 1, 0);

            av_dict_set_int(&opts, "dash_track_number", i + 1, 0);

            av_dict_set_int(&opts, "live", 1, 0);

        }

        if ((ret = avformat_init_output(ctx, &opts)) < 0)

            return ret;

        os->ctx_inited = 1;

        avio_flush(ctx->pb);

        av_dict_free(&opts);



        av_log(s, AV_LOG_VERBOSE, "Representation %d init segment will be written to: %s\n", i, filename);



        // Flush init segment

        // except for mp4, since delay_moov is set and the init segment

        // is then flushed after the first packets

        if (strcmp(os->format_name, "mp4")) {

            flush_init_segment(s, os);

        }



        s->streams[i]->time_base = st->time_base;

        // If the muxer wants to shift timestamps, request to have them shifted

        // already before being handed to this muxer, so we don't have mismatches

        // between the MPD and the actual segments.

        s->avoid_negative_ts = ctx->avoid_negative_ts;

        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {

            AVRational avg_frame_rate = s->streams[i]->avg_frame_rate;

            if (avg_frame_rate.num > 0) {

                if (av_cmp_q(avg_frame_rate, as->min_frame_rate) < 0)

                    as->min_frame_rate = avg_frame_rate;

                if (av_cmp_q(as->max_frame_rate, avg_frame_rate) < 0)

                    as->max_frame_rate = avg_frame_rate;

            } else {

                as->ambiguous_frame_rate = 1;

            }

            c->has_video = 1;

        }



        set_codec_str(s, st->codecpar, os->codec_str, sizeof(os->codec_str));

        os->first_pts = AV_NOPTS_VALUE;

        os->max_pts = AV_NOPTS_VALUE;

        os->last_dts = AV_NOPTS_VALUE;

        os->segment_index = 1;

    }



    if (!c->has_video && c->min_seg_duration <= 0) {

        av_log(s, AV_LOG_WARNING, "no video stream and no min seg duration set\n");

        return AVERROR(EINVAL);

    }

    return 0;

}
