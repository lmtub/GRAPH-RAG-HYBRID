static inline void s390_machine_initfn(Object *obj)

{

    object_property_add_bool(obj, "aes-key-wrap",

                             machine_get_aes_key_wrap,

                             machine_set_aes_key_wrap, NULL);

    object_property_set_description(obj, "aes-key-wrap",

            "enable/disable AES key wrapping using the CPACF wrapping key",


    object_property_set_bool(obj, true, "aes-key-wrap", NULL);



    object_property_add_bool(obj, "dea-key-wrap",

                             machine_get_dea_key_wrap,

                             machine_set_dea_key_wrap, NULL);

    object_property_set_description(obj, "dea-key-wrap",

            "enable/disable DEA key wrapping using the CPACF wrapping key",


    object_property_set_bool(obj, true, "dea-key-wrap", NULL);

    object_property_add_str(obj, "loadparm",

            machine_get_loadparm, machine_set_loadparm, NULL);

    object_property_set_description(obj, "loadparm",

            "Up to 8 chars in set of [A-Za-z0-9. ] (lower case chars converted"

            " to upper case) to pass to machine loader, boot manager,"

            " and guest kernel",









}