UuidInfo *qmp_query_uuid(Error **errp)

{

    UuidInfo *info = g_malloc0(sizeof(*info));

    char uuid[64];



    snprintf(uuid, sizeof(uuid), UUID_FMT, qemu_uuid[0], qemu_uuid[1],

                   qemu_uuid[2], qemu_uuid[3], qemu_uuid[4], qemu_uuid[5],

                   qemu_uuid[6], qemu_uuid[7], qemu_uuid[8], qemu_uuid[9],

                   qemu_uuid[10], qemu_uuid[11], qemu_uuid[12], qemu_uuid[13],

                   qemu_uuid[14], qemu_uuid[15]);



    info->UUID = g_strdup(uuid);

    return info;

}
