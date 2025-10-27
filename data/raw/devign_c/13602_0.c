static int format_name(char *buf, int buf_len, int index)

{

    const char *proto, *dir;

    char *orig_buf_dup = NULL, *mod_buf_dup = NULL;

    int ret = 0;



    if (!av_stristr(buf, "%v"))

        return ret;



    orig_buf_dup = av_strdup(buf);

    if (!orig_buf_dup) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    if (replace_int_data_in_filename(buf, buf_len, orig_buf_dup, 'v', index) < 1) {

        ret = AVERROR(EINVAL);

        goto fail;

    }



    proto = avio_find_protocol_name(orig_buf_dup);

    dir = av_dirname(orig_buf_dup);



    /* if %v is present in the file's directory, create sub-directory */

    if (av_stristr(dir, "%v") && proto && !strcmp(proto, "file")) {

        mod_buf_dup = av_strdup(buf);

        if (!mod_buf_dup) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        dir = av_dirname(mod_buf_dup);

        if (mkdir_p(dir) == -1 && errno != EEXIST) {

            ret = AVERROR(errno);

            goto fail;

        }

    }



fail:

    av_freep(&orig_buf_dup);

    av_freep(&mod_buf_dup);

    return ret;

}
