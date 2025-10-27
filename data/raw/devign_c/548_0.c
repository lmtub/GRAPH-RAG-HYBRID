void hmp_info_tpm(Monitor *mon, const QDict *qdict)

{

    TPMInfoList *info_list, *info;

    Error *err = NULL;

    unsigned int c = 0;

    TPMPassthroughOptions *tpo;



    info_list = qmp_query_tpm(&err);

    if (err) {

        monitor_printf(mon, "TPM device not supported\n");

        error_free(err);

        return;

    }



    if (info_list) {

        monitor_printf(mon, "TPM device:\n");

    }



    for (info = info_list; info; info = info->next) {

        TPMInfo *ti = info->value;

        monitor_printf(mon, " tpm%d: model=%s\n",

                       c, TpmModel_lookup[ti->model]);



        monitor_printf(mon, "  \\ %s: type=%s",

                       ti->id, TpmTypeOptionsKind_lookup[ti->options->kind]);



        switch (ti->options->kind) {

        case TPM_TYPE_OPTIONS_KIND_PASSTHROUGH:

            tpo = ti->options->passthrough;

            monitor_printf(mon, "%s%s%s%s",

                           tpo->has_path ? ",path=" : "",

                           tpo->has_path ? tpo->path : "",

                           tpo->has_cancel_path ? ",cancel-path=" : "",

                           tpo->has_cancel_path ? tpo->cancel_path : "");

            break;

        case TPM_TYPE_OPTIONS_KIND_MAX:

            break;

        }

        monitor_printf(mon, "\n");

        c++;

    }

    qapi_free_TPMInfoList(info_list);

}
