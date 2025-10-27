static int init_output_stream_streamcopy(OutputStream *ost)

{

    OutputFile *of = output_files[ost->file_index];

    InputStream *ist = get_input_stream(ost);

    AVCodecParameters *par_dst = ost->st->codecpar;

    AVCodecParameters *par_src = ost->ref_par;

    AVRational sar;

    int i, ret;

    uint32_t codec_tag = par_dst->codec_tag;



    av_assert0(ist && !ost->filter);



    avcodec_parameters_to_context(ost->enc_ctx, ist->st->codecpar);

    ret = av_opt_set_dict(ost->enc_ctx, &ost->encoder_opts);

    if (ret < 0) {

        av_log(NULL, AV_LOG_FATAL,

               "Error setting up codec context options.\n");

        return ret;

    }

    avcodec_parameters_from_context(par_src, ost->enc_ctx);



    if (!codec_tag) {

        unsigned int codec_tag_tmp;

        if (!of->ctx->oformat->codec_tag ||

            av_codec_get_id (of->ctx->oformat->codec_tag, par_src->codec_tag) == par_src->codec_id ||

            !av_codec_get_tag2(of->ctx->oformat->codec_tag, par_src->codec_id, &codec_tag_tmp))

            codec_tag = par_src->codec_tag;

    }



    ret = avcodec_parameters_copy(par_dst, par_src);

    if (ret < 0)

        return ret;



    par_dst->codec_tag = codec_tag;



    if (!ost->frame_rate.num)

        ost->frame_rate = ist->framerate;

    ost->st->avg_frame_rate = ost->frame_rate;



    ret = avformat_transfer_internal_stream_timing_info(of->ctx->oformat, ost->st, ist->st, copy_tb);

    if (ret < 0)

        return ret;



    // copy timebase while removing common factors

    ost->st->time_base = av_add_q(av_stream_get_codec_timebase(ost->st), (AVRational){0, 1});



    // copy disposition

    ost->st->disposition = ist->st->disposition;



    if (ist->st->nb_side_data) {

        ost->st->side_data = av_realloc_array(NULL, ist->st->nb_side_data,

                                              sizeof(*ist->st->side_data));

        if (!ost->st->side_data)

            return AVERROR(ENOMEM);



        ost->st->nb_side_data = 0;

        for (i = 0; i < ist->st->nb_side_data; i++) {

            const AVPacketSideData *sd_src = &ist->st->side_data[i];

            AVPacketSideData *sd_dst = &ost->st->side_data[ost->st->nb_side_data];



            if (ost->rotate_overridden && sd_src->type == AV_PKT_DATA_DISPLAYMATRIX)

                continue;



            sd_dst->data = av_malloc(sd_src->size);

            if (!sd_dst->data)

                return AVERROR(ENOMEM);

            memcpy(sd_dst->data, sd_src->data, sd_src->size);

            sd_dst->size = sd_src->size;

            sd_dst->type = sd_src->type;

            ost->st->nb_side_data++;

        }

    }



    ost->parser = av_parser_init(par_dst->codec_id);

    ost->parser_avctx = avcodec_alloc_context3(NULL);

    if (!ost->parser_avctx)

        return AVERROR(ENOMEM);



    switch (par_dst->codec_type) {

    case AVMEDIA_TYPE_AUDIO:

        if (audio_volume != 256) {

            av_log(NULL, AV_LOG_FATAL, "-acodec copy and -vol are incompatible (frames are not decoded)\n");

            exit_program(1);

        }

        if((par_dst->block_align == 1 || par_dst->block_align == 1152 || par_dst->block_align == 576) && par_dst->codec_id == AV_CODEC_ID_MP3)

            par_dst->block_align= 0;

        if(par_dst->codec_id == AV_CODEC_ID_AC3)

            par_dst->block_align= 0;

        break;

    case AVMEDIA_TYPE_VIDEO:

        if (ost->frame_aspect_ratio.num) { // overridden by the -aspect cli option

            sar =

                av_mul_q(ost->frame_aspect_ratio,

                         (AVRational){ par_dst->height, par_dst->width });

            av_log(NULL, AV_LOG_WARNING, "Overriding aspect ratio "

                   "with stream copy may produce invalid files\n");

            }

        else if (ist->st->sample_aspect_ratio.num)

            sar = ist->st->sample_aspect_ratio;

        else

            sar = par_src->sample_aspect_ratio;

        ost->st->sample_aspect_ratio = par_dst->sample_aspect_ratio = sar;

        ost->st->avg_frame_rate = ist->st->avg_frame_rate;

        ost->st->r_frame_rate = ist->st->r_frame_rate;

        break;

    }



    return 0;

}
