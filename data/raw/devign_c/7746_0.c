static int url_connect(struct variant *var, AVDictionary *opts)

{

    AVDictionary *tmp = NULL;

    int ret;



    av_dict_copy(&tmp, opts, 0);



    av_opt_set_dict(var->input, &tmp);



    if ((ret = ffurl_connect(var->input, NULL)) < 0) {

        ffurl_close(var->input);

        var->input = NULL;

    }



    av_dict_free(&tmp);

    return ret;

}
