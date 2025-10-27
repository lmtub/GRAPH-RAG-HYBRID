int tpm_register_model(enum TpmModel model)

{

    int i;



    for (i = 0; i < TPM_MAX_MODELS; i++) {

        if (tpm_models[i] == -1) {

            tpm_models[i] = model;

            return 0;

        }

    }

    error_report("Could not register TPM model");

    return 1;

}
