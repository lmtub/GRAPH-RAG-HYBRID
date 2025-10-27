static void tpm_backend_worker_thread(gpointer data, gpointer user_data)

{

    TPMBackend *s = TPM_BACKEND(user_data);

    TPMBackendClass *k = TPM_BACKEND_GET_CLASS(s);



    assert(k->handle_request != NULL);

    k->handle_request(s, (TPMBackendCmd *)data);



    qemu_bh_schedule(s->bh);

}
