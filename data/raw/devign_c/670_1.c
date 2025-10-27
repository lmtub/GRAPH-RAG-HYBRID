static void pc_dimm_check_memdev_is_busy(Object *obj, const char *name,

                                      Object *val, Error **errp)

{

    MemoryRegion *mr;



    mr = host_memory_backend_get_memory(MEMORY_BACKEND(val), errp);

    if (memory_region_is_mapped(mr)) {

        char *path = object_get_canonical_path_component(val);

        error_setg(errp, "can't use already busy memdev: %s", path);

        g_free(path);

    } else {

        qdev_prop_allow_set_link_before_realize(obj, name, val, errp);

    }

}
