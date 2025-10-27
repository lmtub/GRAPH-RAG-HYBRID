int ffurl_open(URLContext **puc, const char *filename, int flags,

               const AVIOInterruptCB *int_cb, AVDictionary **options)

{

    int ret = ffurl_alloc(puc, filename, flags, int_cb);

    if (ret < 0)

        return ret;

    if (options && (*puc)->prot->priv_data_class &&

        (ret = av_opt_set_dict((*puc)->priv_data, options)) < 0)

        goto fail;

    if ((ret = av_opt_set_dict(*puc, options)) < 0)

        goto fail;

    ret = ffurl_connect(*puc, options);

    if (!ret)

        return 0;

fail:

    ffurl_close(*puc);

    *puc = NULL;

    return ret;

}
