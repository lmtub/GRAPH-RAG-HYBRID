void fdt_build_clock_node(void *host_fdt, void *guest_fdt,

                         uint32_t host_phandle,

                         uint32_t guest_phandle)

{

    char *node_path = NULL;

    char *nodename;

    const void *r;

    int ret, node_offset, prop_len, path_len = 16;



    node_offset = fdt_node_offset_by_phandle(host_fdt, host_phandle);

    if (node_offset <= 0) {

        error_setg(&error_fatal,

                   "not able to locate clock handle %d in host device tree",

                   host_phandle);

    }

    node_path = g_malloc(path_len);

    while ((ret = fdt_get_path(host_fdt, node_offset, node_path, path_len))

            == -FDT_ERR_NOSPACE) {

        path_len += 16;

        node_path = g_realloc(node_path, path_len);

    }

    if (ret < 0) {

        error_setg(&error_fatal,

                   "not able to retrieve node path for clock handle %d",

                   host_phandle);

    }



    r = qemu_fdt_getprop(host_fdt, node_path, "compatible", &prop_len,

                         &error_fatal);

    if (strcmp(r, "fixed-clock")) {

        error_setg(&error_fatal,

                   "clock handle %d is not a fixed clock", host_phandle);

    }



    nodename = strrchr(node_path, '/');

    qemu_fdt_add_subnode(guest_fdt, nodename);



    copy_properties_from_host(clock_copied_properties,

                              ARRAY_SIZE(clock_copied_properties),

                              host_fdt, guest_fdt,

                              node_path, nodename);



    qemu_fdt_setprop_cell(guest_fdt, nodename, "phandle", guest_phandle);



    g_free(node_path);

}
