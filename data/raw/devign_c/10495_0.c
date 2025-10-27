static void nvdimm_init(Object *obj)

{

    object_property_add(obj, "label-size", "int",

                        nvdimm_get_label_size, nvdimm_set_label_size, NULL,

                        NULL, NULL);

}
