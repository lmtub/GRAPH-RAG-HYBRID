static void ich_ahci_register(void)

{

    type_register_static(&ich_ahci_info);

    type_register_static_alias(&ich_ahci_info, "ahci");

}
