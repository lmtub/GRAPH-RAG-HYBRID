static int copy_chapters(InputFile *ifile, OutputFile *ofile, int copy_metadata)

{

    AVFormatContext *is = ifile->ctx;

    AVFormatContext *os = ofile->ctx;

    AVChapter **tmp;

    int i;



    tmp = av_realloc(os->chapters, sizeof(*os->chapters) * (is->nb_chapters + os->nb_chapters));

    if (!tmp)

        return AVERROR(ENOMEM);

    os->chapters = tmp;



    for (i = 0; i < is->nb_chapters; i++) {

        AVChapter *in_ch = is->chapters[i], *out_ch;

        int64_t ts_off   = av_rescale_q(ofile->start_time - ifile->ts_offset,

                                       AV_TIME_BASE_Q, in_ch->time_base);

        int64_t rt       = (ofile->recording_time == INT64_MAX) ? INT64_MAX :

                           av_rescale_q(ofile->recording_time, AV_TIME_BASE_Q, in_ch->time_base);





        if (in_ch->end < ts_off)

            continue;

        if (rt != INT64_MAX && in_ch->start > rt + ts_off)

            break;



        out_ch = av_mallocz(sizeof(AVChapter));

        if (!out_ch)

            return AVERROR(ENOMEM);



        out_ch->id        = in_ch->id;

        out_ch->time_base = in_ch->time_base;

        out_ch->start     = FFMAX(0,  in_ch->start - ts_off);

        out_ch->end       = FFMIN(rt, in_ch->end   - ts_off);



        if (copy_metadata)

            av_dict_copy(&out_ch->metadata, in_ch->metadata, 0);



        os->chapters[os->nb_chapters++] = out_ch;

    }

    return 0;

}
