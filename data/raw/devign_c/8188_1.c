block_crypto_open_opts_init(QCryptoBlockFormat format,

                            QemuOpts *opts,

                            Error **errp)

{

    OptsVisitor *ov;

    QCryptoBlockOpenOptions *ret = NULL;

    Error *local_err = NULL;



    ret = g_new0(QCryptoBlockOpenOptions, 1);

    ret->format = format;



    ov = opts_visitor_new(opts);



    visit_start_struct(opts_get_visitor(ov),

                       NULL, NULL, 0, &local_err);

    if (local_err) {

        goto out;

    }



    switch (format) {

    case Q_CRYPTO_BLOCK_FORMAT_LUKS:

        visit_type_QCryptoBlockOptionsLUKS_members(

            opts_get_visitor(ov), &ret->u.luks, &local_err);

        break;



    default:

        error_setg(&local_err, "Unsupported block format %d", format);

        break;

    }

    error_propagate(errp, local_err);

    local_err = NULL;



    visit_end_struct(opts_get_visitor(ov), &local_err);



 out:

    if (local_err) {

        error_propagate(errp, local_err);

        qapi_free_QCryptoBlockOpenOptions(ret);

        ret = NULL;

    }

    opts_visitor_cleanup(ov);

    return ret;

}
