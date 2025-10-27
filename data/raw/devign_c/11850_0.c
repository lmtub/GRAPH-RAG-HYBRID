static void tpm_passthrough_worker_thread(gpointer data,

                                          gpointer user_data)

{

    TPMPassthruThreadParams *thr_parms = user_data;

    TPMPassthruState *tpm_pt = thr_parms->tb->s.tpm_pt;

    TPMBackendCmd cmd = (TPMBackendCmd)data;



    DPRINTF("tpm_passthrough: processing command type %d\n", cmd);



    switch (cmd) {

    case TPM_BACKEND_CMD_PROCESS_CMD:

        tpm_passthrough_unix_transfer(tpm_pt->tpm_fd,

                                      thr_parms->tpm_state->locty_data);



        thr_parms->recv_data_callback(thr_parms->tpm_state,

                                      thr_parms->tpm_state->locty_number);

        break;

    case TPM_BACKEND_CMD_INIT:

    case TPM_BACKEND_CMD_END:

    case TPM_BACKEND_CMD_TPM_RESET:

        /* nothing to do */

        break;

    }

}
