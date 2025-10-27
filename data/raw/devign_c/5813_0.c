void smbios_add_field(int type, int offset, const void *data, size_t len)

{

    struct smbios_field *field;



    smbios_check_collision(type, SMBIOS_FIELD_ENTRY);



    if (!smbios_entries) {

        smbios_entries_len = sizeof(uint16_t);

        smbios_entries = g_malloc0(smbios_entries_len);

    }

    smbios_entries = g_realloc(smbios_entries, smbios_entries_len +

                                                  sizeof(*field) + len);

    field = (struct smbios_field *)(smbios_entries + smbios_entries_len);

    field->header.type = SMBIOS_FIELD_ENTRY;

    field->header.length = cpu_to_le16(sizeof(*field) + len);



    field->type = type;

    field->offset = cpu_to_le16(offset);

    memcpy(field->data, data, len);



    smbios_entries_len += sizeof(*field) + len;

    (*(uint16_t *)smbios_entries) =

            cpu_to_le16(le16_to_cpu(*(uint16_t *)smbios_entries) + 1);

}
