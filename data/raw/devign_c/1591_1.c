static void host_memory_backend_init(Object *obj)

{

    HostMemoryBackend *backend = MEMORY_BACKEND(obj);



    backend->merge = qemu_opt_get_bool(qemu_get_machine_opts(),

                                       "mem-merge", true);

    backend->dump = qemu_opt_get_bool(qemu_get_machine_opts(),

                                      "dump-guest-core", true);

    backend->prealloc = mem_prealloc;



    object_property_add_bool(obj, "merge",

                        host_memory_backend_get_merge,

                        host_memory_backend_set_merge, NULL);

    object_property_add_bool(obj, "dump",

                        host_memory_backend_get_dump,

                        host_memory_backend_set_dump, NULL);

    object_property_add_bool(obj, "prealloc",

                        host_memory_backend_get_prealloc,

                        host_memory_backend_set_prealloc, NULL);

    object_property_add(obj, "size", "int",

                        host_memory_backend_get_size,

                        host_memory_backend_set_size, NULL, NULL, NULL);

    object_property_add(obj, "host-nodes", "int",

                        host_memory_backend_get_host_nodes,

                        host_memory_backend_set_host_nodes, NULL, NULL, NULL);

    object_property_add_enum(obj, "policy", "HostMemPolicy",

                             HostMemPolicy_lookup,

                             host_memory_backend_get_policy,

                             host_memory_backend_set_policy, NULL);

}
