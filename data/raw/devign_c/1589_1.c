static void qemu_rdma_dump_gid(const char *who, struct rdma_cm_id *id)

{

    char sgid[33];

    char dgid[33];

    inet_ntop(AF_INET6, &id->route.addr.addr.ibaddr.sgid, sgid, sizeof sgid);

    inet_ntop(AF_INET6, &id->route.addr.addr.ibaddr.dgid, dgid, sizeof dgid);

    DPRINTF("%s Source GID: %s, Dest GID: %s\n", who, sgid, dgid);

}
