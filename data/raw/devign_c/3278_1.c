static void nvme_instance_init(Object *obj)

{

    object_property_add(obj, "bootindex", "int32",

                        nvme_get_bootindex,

                        nvme_set_bootindex, NULL, NULL, NULL);

    object_property_set_int(obj, -1, "bootindex", NULL);

}
