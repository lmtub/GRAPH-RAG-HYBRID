static void smbios_build_type_1_fields(const char *t)

{

    char buf[1024];



    if (get_param_value(buf, sizeof(buf), "manufacturer", t))

        smbios_add_field(1, offsetof(struct smbios_type_1, manufacturer_str),

                         strlen(buf) + 1, buf);

    if (get_param_value(buf, sizeof(buf), "product", t))

        smbios_add_field(1, offsetof(struct smbios_type_1, product_name_str),

                         strlen(buf) + 1, buf);

    if (get_param_value(buf, sizeof(buf), "version", t))

        smbios_add_field(1, offsetof(struct smbios_type_1, version_str),

                         strlen(buf) + 1, buf);

    if (get_param_value(buf, sizeof(buf), "serial", t))

        smbios_add_field(1, offsetof(struct smbios_type_1, serial_number_str),

                         strlen(buf) + 1, buf);

    if (get_param_value(buf, sizeof(buf), "uuid", t)) {

        if (qemu_uuid_parse(buf, qemu_uuid) != 0) {

            fprintf(stderr, "Invalid SMBIOS UUID string\n");

            exit(1);

        }

    }

    if (get_param_value(buf, sizeof(buf), "sku", t))

        smbios_add_field(1, offsetof(struct smbios_type_1, sku_number_str),

                         strlen(buf) + 1, buf);

    if (get_param_value(buf, sizeof(buf), "family", t))

        smbios_add_field(1, offsetof(struct smbios_type_1, family_str),

                         strlen(buf) + 1, buf);

}
