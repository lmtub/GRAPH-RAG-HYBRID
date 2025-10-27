static void *acpi_add_rom_blob(AcpiBuildState *build_state, GArray *blob,

                               const char *name)

{

    return rom_add_blob(name, blob->data, acpi_data_len(blob), -1, name,

                        acpi_build_update, build_state);

}
