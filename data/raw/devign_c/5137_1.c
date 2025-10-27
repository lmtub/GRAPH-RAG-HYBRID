static int set_segment_filename(AVFormatContext *s)

{

    SegmentContext *seg = s->priv_data;

    AVFormatContext *oc = seg->avf;

    size_t size;



    if (seg->segment_idx_wrap)

        seg->segment_idx %= seg->segment_idx_wrap;

    if (seg->use_strftime) {

        time_t now0;

        struct tm *tm, tmpbuf;

        time(&now0);

        tm = localtime_r(&now0, &tmpbuf);

        if (!strftime(oc->filename, sizeof(oc->filename), s->filename, tm)) {

            av_log(oc, AV_LOG_ERROR, "Could not get segment filename with strftime\n");

            return AVERROR(EINVAL);

        }

    } else if (av_get_frame_filename(oc->filename, sizeof(oc->filename),

                                     s->filename, seg->segment_idx) < 0) {

        av_log(oc, AV_LOG_ERROR, "Invalid segment filename template '%s'\n", s->filename);

        return AVERROR(EINVAL);

    }



    /* copy modified name in list entry */

    size = strlen(av_basename(oc->filename)) + 1;

    if (seg->entry_prefix)

        size += strlen(seg->entry_prefix);



    seg->cur_entry.filename = av_mallocz(size);

    if (!seg->cur_entry.filename)

        return AVERROR(ENOMEM);

    snprintf(seg->cur_entry.filename, size, "%s%s",

             seg->entry_prefix ? seg->entry_prefix : "",

             av_basename(oc->filename));



    return 0;

}
