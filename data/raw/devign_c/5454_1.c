TPMVersion tpm_tis_get_tpm_version(Object *obj)
{
    TPMState *s = TPM(obj);
    return tpm_backend_get_tpm_version(s->be_driver);