static void smbios_validate_table(void)

{

    if (smbios_type4_count && smbios_type4_count != smp_cpus) {

         fprintf(stderr,

                 "Number of SMBIOS Type 4 tables must match cpu count.\n");

        exit(1);

    }

}
