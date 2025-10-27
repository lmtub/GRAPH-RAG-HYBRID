int xenstore_domain_init1(const char *kernel, const char *ramdisk,

                          const char *cmdline)

{

    char *dom, uuid_string[42], vm[256], path[256];

    int i;



    snprintf(uuid_string, sizeof(uuid_string), UUID_FMT,

             qemu_uuid[0], qemu_uuid[1], qemu_uuid[2], qemu_uuid[3],

             qemu_uuid[4], qemu_uuid[5], qemu_uuid[6], qemu_uuid[7],

             qemu_uuid[8], qemu_uuid[9], qemu_uuid[10], qemu_uuid[11],

             qemu_uuid[12], qemu_uuid[13], qemu_uuid[14], qemu_uuid[15]);

    dom = xs_get_domain_path(xenstore, xen_domid);

    snprintf(vm,  sizeof(vm),  "/vm/%s", uuid_string);



    xenstore_domain_mkdir(dom);



    xenstore_write_str(vm, "image/ostype",  "linux");

    if (kernel)

        xenstore_write_str(vm, "image/kernel",  kernel);

    if (ramdisk)

        xenstore_write_str(vm, "image/ramdisk", ramdisk);

    if (cmdline)

        xenstore_write_str(vm, "image/cmdline", cmdline);



    /* name + id */

    xenstore_write_str(vm,  "name",   qemu_name ? qemu_name : "no-name");

    xenstore_write_str(vm,  "uuid",   uuid_string);

    xenstore_write_str(dom, "name",   qemu_name ? qemu_name : "no-name");

    xenstore_write_int(dom, "domid",  xen_domid);

    xenstore_write_str(dom, "vm",     vm);



    /* memory */

    xenstore_write_int(dom, "memory/target", ram_size >> 10);  // kB

    xenstore_write_int(vm, "memory",         ram_size >> 20);  // MB

    xenstore_write_int(vm, "maxmem",         ram_size >> 20);  // MB



    /* cpus */

    for (i = 0; i < smp_cpus; i++) {

	snprintf(path, sizeof(path), "cpu/%d/availability",i);

	xenstore_write_str(dom, path, "online");

    }

    xenstore_write_int(vm, "vcpu_avail",  smp_cpus);

    xenstore_write_int(vm, "vcpus",       smp_cpus);



    /* vnc password */

    xenstore_write_str(vm, "vncpassword", "" /* FIXME */);



    free(dom);

    return 0;

}
