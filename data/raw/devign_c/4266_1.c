static void ffm_set_write_index(AVFormatContext *s, int64_t pos,

                                int64_t file_size)

{

    av_opt_set_int(s, "server_attached", 1, AV_OPT_SEARCH_CHILDREN);

    av_opt_set_int(s, "write_index", pos, AV_OPT_SEARCH_CHILDREN);

    av_opt_set_int(s, "file_size", file_size, AV_OPT_SEARCH_CHILDREN);

}
