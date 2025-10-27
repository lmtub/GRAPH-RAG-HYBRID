static int open_file(AVFormatContext *avf, unsigned fileno)

{

    ConcatContext *cat = avf->priv_data;

    ConcatFile *file = &cat->files[fileno];

    int ret;



    if (cat->avf)

        avformat_close_input(&cat->avf);



    cat->avf = avformat_alloc_context();

    if (!cat->avf)

        return AVERROR(ENOMEM);



    cat->avf->flags |= avf->flags;

    cat->avf->interrupt_callback = avf->interrupt_callback;



    if ((ret = ff_copy_whiteblacklists(cat->avf, avf)) < 0)

        return ret;



    if ((ret = avformat_open_input(&cat->avf, file->url, NULL, NULL)) < 0 ||

        (ret = avformat_find_stream_info(cat->avf, NULL)) < 0) {

        av_log(avf, AV_LOG_ERROR, "Impossible to open '%s'\n", file->url);

        avformat_close_input(&cat->avf);

        return ret;

    }

    cat->cur_file = file;

    if (file->start_time == AV_NOPTS_VALUE)

        file->start_time = !fileno ? 0 :

                           cat->files[fileno - 1].start_time +

                           cat->files[fileno - 1].duration;

    file->file_start_time = (cat->avf->start_time == AV_NOPTS_VALUE) ? 0 : cat->avf->start_time;

    file->file_inpoint = (file->inpoint == AV_NOPTS_VALUE) ? file->file_start_time : file->inpoint;

    if (file->duration == AV_NOPTS_VALUE && file->outpoint != AV_NOPTS_VALUE)

        file->duration = file->outpoint - file->file_inpoint;



    if (cat->segment_time_metadata) {

        av_dict_set_int(&file->metadata, "lavf.concatdec.start_time", file->start_time, 0);

        if (file->duration != AV_NOPTS_VALUE)

            av_dict_set_int(&file->metadata, "lavf.concatdec.duration", file->duration, 0);

    }



    if ((ret = match_streams(avf)) < 0)

        return ret;

    if (file->inpoint != AV_NOPTS_VALUE) {

       if ((ret = avformat_seek_file(cat->avf, -1, INT64_MIN, file->inpoint, file->inpoint, 0)) < 0)

           return ret;

    }

    return 0;

}
