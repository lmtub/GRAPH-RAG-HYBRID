static void smbios_build_type_0_fields(QemuOpts *opts)

{

    const char *val;

    unsigned char major, minor;



    val = qemu_opt_get(opts, "vendor");

    if (val) {

        smbios_add_field(0, offsetof(struct smbios_type_0, vendor_str),

                         val, strlen(val) + 1);

    }

    val = qemu_opt_get(opts, "version");

    if (val) {

        smbios_add_field(0, offsetof(struct smbios_type_0, bios_version_str),

                         val, strlen(val) + 1);

    }

    val = qemu_opt_get(opts, "date");

    if (val) {

        smbios_add_field(0, offsetof(struct smbios_type_0,

                                     bios_release_date_str),

                         val, strlen(val) + 1);

    }

    val = qemu_opt_get(opts, "release");

    if (val) {

        if (sscanf(val, "%hhu.%hhu", &major, &minor) != 2) {

            error_report("Invalid release");

            exit(1);

        }

        smbios_add_field(0, offsetof(struct smbios_type_0,

                                     system_bios_major_release),

                         &major, 1);

        smbios_add_field(0, offsetof(struct smbios_type_0,

                                     system_bios_minor_release),

                         &minor, 1);

    }

}
