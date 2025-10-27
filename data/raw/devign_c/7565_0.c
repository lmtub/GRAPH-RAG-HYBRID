block_crypto_create_opts_init(QCryptoBlockFormat format,

                              QemuOpts *opts,

                              Error **errp)

{

    Visitor *v;

    QCryptoBlockCreateOptions *ret = NULL;

    Error *local_err = NULL;



    ret = g_new0(QCryptoBlockCreateOptions, 1);

    ret->format = format;



    v = opts_visitor_new(opts);



    visit_start_struct(v, NULL, NULL, 0, &local_err);

    if (local_err) {

        goto out;

    }



    switch (format) {

    case Q_CRYPTO_BLOCK_FORMAT_LUKS:

        visit_type_QCryptoBlockCreateOptionsLUKS_members(

            v, &ret->u.luks, &local_err);

        break;



    default:

        error_setg(&local_err, "Unsupported block format %d", format);

        break;

    }

    if (!local_err) {

        visit_check_struct(v, &local_err);

    }



    visit_end_struct(v, NULL);



 out:

    if (local_err) {

        error_propagate(errp, local_err);

        qapi_free_QCryptoBlockCreateOptions(ret);

        ret = NULL;

    }

    visit_free(v);

    return ret;

}
