int tpm_register_driver(const TPMDriverOps *tdo)

{

    int i;



    for (i = 0; i < TPM_MAX_DRIVERS; i++) {

        if (!be_drivers[i]) {

            be_drivers[i] = tdo;

            return 0;

        }

    }

    error_report("Could not register TPM driver");

    return 1;

}
