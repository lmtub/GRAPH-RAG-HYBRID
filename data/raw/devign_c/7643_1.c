int av_bsf_list_parse_str(const char *str, AVBSFContext **bsf_lst)

{

    AVBSFList *lst;

    char *bsf_str, *buf, *dup, *saveptr;

    int ret;



    if (!str)

        return av_bsf_get_null_filter(bsf_lst);



    lst = av_bsf_list_alloc();

    if (!lst)

        return AVERROR(ENOMEM);



    if (!(dup = buf = av_strdup(str)))

        return AVERROR(ENOMEM);



    while (1) {

        bsf_str = av_strtok(buf, ",", &saveptr);

        if (!bsf_str)

            break;



        ret = bsf_parse_single(bsf_str, lst);

        if (ret < 0)

            goto end;



        buf = NULL;

    }



    ret = av_bsf_list_finalize(&lst, bsf_lst);

end:

    if (ret < 0)

        av_bsf_list_free(&lst);

    av_free(dup);

    return ret;

}
