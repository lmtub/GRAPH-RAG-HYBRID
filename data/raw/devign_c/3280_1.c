static void spapr_cpu_core_realize(DeviceState *dev, Error **errp)

{

    sPAPRCPUCore *sc = SPAPR_CPU_CORE(OBJECT(dev));

    CPUCore *cc = CPU_CORE(OBJECT(dev));

    const char *typename = object_class_get_name(sc->cpu_class);

    size_t size = object_type_get_instance_size(typename);

    Error *local_err = NULL;

    Object *obj;

    int i;



    sc->threads = g_malloc0(size * cc->nr_threads);

    for (i = 0; i < cc->nr_threads; i++) {

        char id[32];

        void *obj = sc->threads + i * size;



        object_initialize(obj, size, typename);

        snprintf(id, sizeof(id), "thread[%d]", i);

        object_property_add_child(OBJECT(sc), id, obj, &local_err);

        if (local_err) {

            goto err;

        }


    }

    object_child_foreach(OBJECT(dev), spapr_cpu_core_realize_child, &local_err);

    if (local_err) {

        goto err;

    } else {

        return;

    }



err:

    while (--i >= 0) {

        obj = sc->threads + i * size;

        object_unparent(obj);

    }

    g_free(sc->threads);

    error_propagate(errp, local_err);

}