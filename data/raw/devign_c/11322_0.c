void spapr_tce_free(sPAPRTCETable *tcet)

{

    QLIST_REMOVE(tcet, list);



    if (!kvm_enabled() ||

        (kvmppc_remove_spapr_tce(tcet->table, tcet->fd,

                                 tcet->window_size) != 0)) {

        g_free(tcet->table);

    }



    g_free(tcet);

}
