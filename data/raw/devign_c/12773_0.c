void smbios_set_defaults(const char *manufacturer, const char *product,

                         const char *version)

{

    SMBIOS_SET_DEFAULT(type1.manufacturer, manufacturer);

    SMBIOS_SET_DEFAULT(type1.product, product);

    SMBIOS_SET_DEFAULT(type1.version, version);

}
