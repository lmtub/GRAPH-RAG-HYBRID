static int url_connect(struct playlist *pls, AVDictionary *opts, AVDictionary *opts2)

{

    AVDictionary *tmp = NULL;

    int ret;



    av_dict_copy(&tmp, opts, 0);

    av_dict_copy(&tmp, opts2, 0);



    av_opt_set_dict(pls->input, &tmp);



    if ((ret = ffurl_connect(pls->input, NULL)) < 0) {

        ffurl_close(pls->input);

        pls->input = NULL;

    }



    av_dict_free(&tmp);

    return ret;

}
