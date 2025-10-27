static void pc_dimm_check_memdev_is_busy(Object *obj, const char *name,

                                      Object *val, Error **errp)

{

    MemoryRegion *mr;

    Error *local_err = NULL;



    mr = host_memory_backend_get_memory(MEMORY_BACKEND(val), &local_err);

    if (local_err) {

        goto out;

    }

    if (memory_region_is_mapped(mr)) {

        char *path = object_get_canonical_path_component(val);

        error_setg(&local_err, "can't use already busy memdev: %s", path);

        g_free(path);

    } else {

        qdev_prop_allow_set_link_before_realize(obj, name, val, &local_err);

    }



out:

    error_propagate(errp, local_err);

}
