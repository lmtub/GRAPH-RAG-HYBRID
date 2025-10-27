static void memory_region_initfn(Object *obj)

{

    MemoryRegion *mr = MEMORY_REGION(obj);

    ObjectProperty *op;



    mr->ops = &unassigned_mem_ops;

    mr->enabled = true;

    mr->romd_mode = true;

    mr->global_locking = true;

    mr->destructor = memory_region_destructor_none;

    QTAILQ_INIT(&mr->subregions);

    QTAILQ_INIT(&mr->coalesced);



    op = object_property_add(OBJECT(mr), "container",

                             "link<" TYPE_MEMORY_REGION ">",

                             memory_region_get_container,

                             NULL, /* memory_region_set_container */

                             NULL, NULL, &error_abort);

    op->resolve = memory_region_resolve_container;



    object_property_add(OBJECT(mr), "addr", "uint64",

                        memory_region_get_addr,

                        NULL, /* memory_region_set_addr */

                        NULL, NULL, &error_abort);

    object_property_add(OBJECT(mr), "priority", "uint32",

                        memory_region_get_priority,

                        NULL, /* memory_region_set_priority */

                        NULL, NULL, &error_abort);

    object_property_add_bool(OBJECT(mr), "may-overlap",

                             memory_region_get_may_overlap,

                             NULL, /* memory_region_set_may_overlap */

                             &error_abort);

    object_property_add(OBJECT(mr), "size", "uint64",

                        memory_region_get_size,

                        NULL, /* memory_region_set_size, */

                        NULL, NULL, &error_abort);

}
