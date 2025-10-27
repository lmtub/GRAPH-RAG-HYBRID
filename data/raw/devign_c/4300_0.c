void do_smbios_option(const char *optarg)

{

#ifdef TARGET_I386

    if (smbios_entry_add(optarg) < 0) {

        fprintf(stderr, "Wrong smbios provided\n");

        exit(1);

    }

#endif

}
