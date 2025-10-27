static int check_recording_time(OutputStream *ost)

{

    OutputFile *of = output_files[ost->file_index];



    if (of->recording_time != INT64_MAX &&

        av_compare_ts(ost->sync_opts - ost->first_pts, ost->st->codec->time_base, of->recording_time,

                      AV_TIME_BASE_Q) >= 0) {

        ost->is_past_recording_time = 1;

        return 0;

    }

    return 1;

}
